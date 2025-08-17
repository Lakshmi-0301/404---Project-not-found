import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        df_recent = df.groupby('Country').last().reset_index()
        
        df_recent['Trade_Volume'] = df_recent['Trade (% of GDP)'] * df_recent['GDP (current US$)'] / 100
        df_recent['Export_Volume'] = df_recent['Exports of goods and services (% of GDP)'] * df_recent['GDP (current US$)'] / 100
        df_recent['Import_Volume'] = df_recent['Imports of goods and services (% of GDP)'] * df_recent['GDP (current US$)'] / 100
        
        numeric_cols = df_recent.select_dtypes(include=[np.number]).columns
        df_recent[numeric_cols] = df_recent[numeric_cols].fillna(df_recent[numeric_cols].median())
        
        return df_recent
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def select_top_countries(df, n_countries=25):
    scaler = StandardScaler()
    
    factors = ['GDP (current US$)', 'Trade (% of GDP)', 'Trade_Volume']
    available_factors = [f for f in factors if f in df.columns]
    
    if available_factors:
        importance_matrix = scaler.fit_transform(df[available_factors])
        df['Importance_Score'] = importance_matrix.mean(axis=1)
    else:
        df['Importance_Score'] = df['GDP (current US$)'].fillna(0)
    
    top_countries = df.nlargest(n_countries, 'Importance_Score').copy()
    
    return top_countries

def create_trade_network(df):
    trade_features = [
        'Exports of goods and services (% of GDP)',
        'Imports of goods and services (% of GDP)', 
        'Trade (% of GDP)',
        'GDP per capita (current US$)',
        'GDP growth (annual %)'
    ]
    
    available_features = [f for f in trade_features if f in df.columns]
    
    if len(available_features) < 3:
        st.warning("Limited trade features available. Using GDP and basic trade metrics.")
        available_features = ['GDP (current US$)', 'Trade (% of GDP)']
    
    feature_matrix = df[available_features].fillna(df[available_features].median())
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    similarity_matrix = cosine_similarity(normalized_features)
    
    G = nx.Graph()
    
    for i, country in enumerate(df['Country']):
        G.add_node(country, 
                  gdp=df.iloc[i]['GDP (current US$)'],
                  trade_gdp=df.iloc[i]['Trade (% of GDP)'],
                  exports=df.iloc[i].get('Export_Volume', 0),
                  imports=df.iloc[i].get('Import_Volume', 0),
                  pop=df.iloc[i].get('pop_total_population___both_sexes', 0))
    
    threshold = np.percentile(similarity_matrix, 80)
    
    countries = df['Country'].tolist()
    for i in range(len(countries)):
        for j in range(i+1, len(countries)):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                weight = similarity * np.log1p(min(df.iloc[i]['GDP (current US$)'], 
                                                 df.iloc[j]['GDP (current US$)']))
                G.add_edge(countries[i], countries[j], 
                          weight=weight, 
                          similarity=similarity)
    
    return G

def calculate_network_metrics(G):
    if G.number_of_nodes() == 0:
        return {}, pd.DataFrame()
        
    metrics = {}
    
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    try:
        metrics['degree_centrality'] = nx.degree_centrality(G)
    except:
        metrics['degree_centrality'] = {node: 0 for node in G.nodes()}
    
    try:
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
    except:
        metrics['betweenness_centrality'] = {node: 0 for node in G.nodes()}
    
    try:
        metrics['closeness_centrality'] = nx.closeness_centrality(G, distance='weight')
    except:
        metrics['closeness_centrality'] = {node: 0 for node in G.nodes()}
    
    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector_centrality'] = {node: 0 for node in G.nodes()}
    
    try:
        metrics['pagerank'] = nx.pagerank(G, weight='weight', max_iter=1000)
    except:
        metrics['pagerank'] = {node: 1/G.number_of_nodes() for node in G.nodes()}
    
    centrality_df = pd.DataFrame({
        'Country': list(G.nodes()),
        'Degree_Centrality': [metrics['degree_centrality'][node] for node in G.nodes()],
        'Betweenness_Centrality': [metrics['betweenness_centrality'][node] for node in G.nodes()],
        'Closeness_Centrality': [metrics['closeness_centrality'][node] for node in G.nodes()],
        'Eigenvector_Centrality': [metrics['eigenvector_centrality'][node] for node in G.nodes()],
        'PageRank': [metrics['pagerank'][node] for node in G.nodes()]
    })
    
    scaler = MinMaxScaler()
    centrality_cols = ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality', 'PageRank']
    centrality_df['Composite_Centrality'] = scaler.fit_transform(centrality_df[centrality_cols]).mean(axis=1)
    
    return metrics, centrality_df

def simulate_network_disruption(G, node_to_remove):
    G_disrupted = G.copy()
    
    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()
    original_components = nx.number_connected_components(G)
    
    if node_to_remove in G_disrupted:
        neighbors = list(G_disrupted.neighbors(node_to_remove))
        G_disrupted.remove_node(node_to_remove)
    else:
        neighbors = []
    
    new_nodes = G_disrupted.number_of_nodes()
    new_edges = G_disrupted.number_of_edges()
    new_components = nx.number_connected_components(G_disrupted)
    
    disruption_impact = {
        'removed_node': node_to_remove,
        'neighbors_affected': len(neighbors),
        'nodes_lost': original_nodes - new_nodes,
        'edges_lost': original_edges - new_edges,
        'components_before': original_components,
        'components_after': new_components,
        'fragmentation_increase': new_components - original_components,
        'connectivity_loss': (original_edges - new_edges) / original_edges if original_edges > 0 else 0,
        'network_efficiency_before': nx.global_efficiency(G),
        'network_efficiency_after': nx.global_efficiency(G_disrupted) if G_disrupted.number_of_nodes() > 0 else 0
    }
    
    disruption_impact['efficiency_loss'] = disruption_impact['network_efficiency_before'] - disruption_impact['network_efficiency_after']
    
    return G_disrupted, disruption_impact, neighbors

def create_network_visualization(G, centrality_df, layout_type='spring'):
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes to display", x=0.5, y=0.5, showarrow=False)
        return fig
    
    try:
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
    except Exception as e:
        pos = nx.spring_layout(G, seed=42)
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_sizes = [centrality_df[centrality_df['Country'] == node]['Composite_Centrality'].iloc[0] * 50 + 10 
                  for node in G.nodes()]
    
    node_colors = [centrality_df[centrality_df['Country'] == node]['Degree_Centrality'].iloc[0] 
                   for node in G.nodes()]
    
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.02,
                title="Degree Centrality"
            ),
            line=dict(width=2, color='white')
        ),
        hovertext=[f"{node}<br>Composite Centrality: {centrality_df[centrality_df['Country']==node]['Composite_Centrality'].iloc[0]:.3f}" 
                   for node in G.nodes()],
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text="Global Trade Network Graph", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=5, r=5, t=40),
        annotations=[dict(
            text="Node size = Composite Centrality, Color = Degree Centrality",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def main():
    st.header('Global Trade Network Analysis')
    
    st.write("""
    This application analyzes global trade relationships as a network, identifying central countries 
    and simulating the impact of network disruptions.
    """)
    
    file_path = r"datasets/processed/integrated_tren_dataset.csv"
    
    with st.spinner("Loading and processing dataset..."):
        df = load_and_prepare_data(file_path)
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
    
    n_countries = 25
    layout_type = "circular"
    
    with st.spinner("Selecting top countries and building network..."):
        top_countries_df = select_top_countries(df, n_countries)
        G = create_trade_network(top_countries_df)
        network_metrics, centrality_df = calculate_network_metrics(G)
    
    st.subheader('Network Overview')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries (Nodes)", network_metrics['nodes'])
    with col2:
        st.metric("Trade Connections (Edges)", network_metrics['edges'])
    with col3:
        st.metric("Network Density", f"{network_metrics['density']:.3f}")
    with col4:
        st.metric("Connected Components", nx.number_connected_components(G))
    
    st.subheader('Trade Network Visualization')
    
    network_fig = create_network_visualization(G, centrality_df, layout_type)
    st.plotly_chart(network_fig, use_container_width=True)
    
    st.subheader('Network Centrality Analysis')
    
    n_central_countries = st.slider("Number of Top Central Countries to Display", min_value=5, max_value=25, value=10, step=1, key="central_countries")
    
    top_central = centrality_df.nlargest(n_central_countries, 'Composite_Centrality')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Most Central Countries")
        display_centrality = top_central[['Country', 'Composite_Centrality', 'Degree_Centrality', 'Betweenness_Centrality']].copy()
        display_centrality['Composite_Centrality'] = display_centrality['Composite_Centrality'].round(3)
        display_centrality['Degree_Centrality'] = display_centrality['Degree_Centrality'].round(3)
        display_centrality['Betweenness_Centrality'] = display_centrality['Betweenness_Centrality'].round(3)
        st.dataframe(display_centrality, use_container_width=True)
    
    with col2:
        fig_cent = px.bar(
            top_central.head(8),
            x='Composite_Centrality',
            y='Country',
            orientation='h',
            title='Top 8 Countries by Composite Centrality',
            color='Composite_Centrality',
            color_continuous_scale='Blues'
        )
        fig_cent.update_layout(height=400)
        st.plotly_chart(fig_cent, use_container_width=True)
    
    st.write("Centrality Measures Comparison")
    n_heatmap_countries = st.slider("Number of Countries for Heatmap", min_value=5, max_value=25, value=10, step=1, key="heatmap_countries")
    top_heatmap = centrality_df.nlargest(n_heatmap_countries, 'Composite_Centrality')
    centrality_matrix = top_heatmap[['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality', 'PageRank']].T
    centrality_matrix.columns = top_heatmap['Country']
    
    fig_heatmap = px.imshow(
        centrality_matrix,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title=f"Centrality Measures Heatmap (Top {n_heatmap_countries} Countries)"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.subheader('Network Disruption Simulation')
    
    st.write("Select a country to remove from the network and analyze the impact:")
    
    country_to_remove = st.selectbox(
        "Country to Remove",
        options=sorted(centrality_df['Country'].tolist()),
        index=0
    )
    
    if st.button("Simulate Network Disruption", type="primary"):
        with st.spinner(f"Simulating removal of {country_to_remove}..."):
            G_disrupted, disruption_impact, affected_neighbors = simulate_network_disruption(G, country_to_remove)
        
        st.subheader(f"Impact of Removing {country_to_remove}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Neighbors Affected", disruption_impact['neighbors_affected'])
            st.metric("Edges Lost", disruption_impact['edges_lost'])
        with col2:
            st.metric("Connectivity Loss", f"{disruption_impact['connectivity_loss']*100:.1f}%")
            st.metric("Remaining Nodes", G_disrupted.number_of_nodes())
        with col3:
            st.metric("Efficiency Loss", f"{disruption_impact['efficiency_loss']:.4f}")
        
        if affected_neighbors:
            st.write("Directly Affected Trading Partners:")
            st.write(", ".join(affected_neighbors))
        
        if G_disrupted.number_of_nodes() > 0:
            centrality_df_disrupted = centrality_df[centrality_df['Country'] != country_to_remove].copy()
            fig_disrupted = create_network_visualization(G_disrupted, centrality_df_disrupted, layout_type)
            fig_disrupted.update_layout(title=f"Trade Network After Removing {country_to_remove}")
            st.plotly_chart(fig_disrupted, use_container_width=True)
    
    st.subheader('Network Vulnerability Analysis')
    
    vulnerability_results = []
    top_5_central = centrality_df.nlargest(5, 'Composite_Centrality')['Country'].tolist()
    
    for country in top_5_central:
        G_temp, impact, _ = simulate_network_disruption(G, country)
        vulnerability_results.append({
            'Country': country,
            'Composite_Centrality': centrality_df[centrality_df['Country']==country]['Composite_Centrality'].iloc[0],
            'Connectivity_Loss': impact['connectivity_loss'],
            'Efficiency_Loss': impact['efficiency_loss'],
            'Neighbors_Affected': impact['neighbors_affected'],
            'Fragmentation_Impact': impact['fragmentation_increase']
        })
    
    vulnerability_df = pd.DataFrame(vulnerability_results)
    
    fig_vuln = px.scatter(
        vulnerability_df,
        x='Composite_Centrality',
        y='Efficiency_Loss',
        size='Neighbors_Affected',
        hover_name='Country',
        title='Network Centrality vs Disruption Impact',
        labels={
            'Composite_Centrality': 'Composite Centrality Score',
            'Efficiency_Loss': 'Network Efficiency Loss'
        }
    )
    st.plotly_chart(fig_vuln, use_container_width=True)
    
    st.subheader('Key Insights')
    
    most_central = centrality_df.iloc[centrality_df['Composite_Centrality'].idxmax()]
    highest_impact = vulnerability_df.iloc[vulnerability_df['Efficiency_Loss'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Network Structure:")
        st.markdown(f"""
        - Most Central Country: {most_central['Country']} (Score: {most_central['Composite_Centrality']:.3f})
        - Network Density: {network_metrics['density']:.3f} (Scale: 0-1)
        - Average Clustering: {nx.average_clustering(G):.3f}
        - Network Diameter: {nx.diameter(G) if nx.is_connected(G) else "Network not fully connected"}
        """)
    
    with col2:
        st.write("Vulnerability Assessment:")
        st.markdown(f"""
        - Highest Impact Removal: {highest_impact['Country']} ({highest_impact['Efficiency_Loss']:.4f} efficiency loss)
        - Most Connected Node: {centrality_df.iloc[centrality_df['Degree_Centrality'].idxmax()]['Country']}
        - Key Bridge Country: {centrality_df.iloc[centrality_df['Betweenness_Centrality'].idxmax()]['Country']}
        - Network Robustness: {"High" if network_metrics['density'] > 0.3 else "Medium" if network_metrics['density'] > 0.15 else "Low"}
        """)
    
    st.subheader('Export Results')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        centrality_csv = centrality_df.to_csv(index=False)
        st.download_button(
            "Download Centrality Analysis",
            centrality_csv,
            "trade_network_centrality.csv",
            "text/csv"
        )
    
    with col2:
        vulnerability_csv = vulnerability_df.to_csv(index=False)
        st.download_button(
            "Download Vulnerability Analysis",
            vulnerability_csv,
            "network_vulnerability_analysis.csv",
            "text/csv"
        )
    
    with col3:
        network_summary = {
            'Total_Countries': network_metrics['nodes'],
            'Total_Connections': network_metrics['edges'],
            'Network_Density': network_metrics['density'],
            'Most_Central_Country': most_central['Country'],
            'Highest_Impact_Removal': highest_impact['Country']
        }
        summary_df = pd.DataFrame([network_summary])
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            "Download Network Summary",
            summary_csv,
            "network_summary.csv",
            "text/csv"
        )
    
    with st.expander("Methodology & Technical Details"):
        st.markdown("""
        Network Construction:
        - Countries selected based on GDP, trade volume, and importance scores
        - Edges created using cosine similarity of trade profiles (top 20% connections)
        - Edge weights based on similarity and economic size
        
        Centrality Measures:
        - Degree Centrality: Number of direct connections
        - Betweenness Centrality: Control over shortest paths (bridge countries)
        - Closeness Centrality: Average distance to all other nodes
        - Eigenvector Centrality: Connections to other well-connected nodes
        - PageRank: Google's algorithm adapted for trade networks
        - Composite Score: Normalized average of all centrality measures
        
        Disruption Simulation:
        - Node removal with cascade analysis
        - Network efficiency calculated using global efficiency metric
        - Fragmentation measured by connected components increase
        
        Limitations:
        - Simplified bilateral trade model
        - Static network (no temporal dynamics)
        - Economic similarity proxy for trade relationships
        - Limited by available data granularity
        """)

if __name__ == "__main__":
    main()