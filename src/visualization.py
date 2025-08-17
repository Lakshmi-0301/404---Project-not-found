import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import numpy as np

def load_data(file=None, default_path="datasets/processed/integrated_tren_dataset.csv"):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(default_path)
        
        available_cols = df.columns.tolist()
        
        numeric_cols = [col for col in [
            'Trade (% of GDP)', 'disaster_count', 'disaster_damage_usd_thousands',
            'agr_yield_total', 'disaster_total_affected', 'GDP (current US$)',
            'GDP per capita (current US$)', 'Population growth (annual %)'
        ] if col in available_cols]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_heatmap(df, country, output_dir='heatmaps'):
    os.makedirs(output_dir, exist_ok=True)
    
    country_df = df[df['Country'] == country].copy()
    
    if country_df.empty:
        st.warning(f"No data found for country: {country}")
        return
    
    cols_for_heatmap = ['Year', 'Trade (% of GDP)', 'disaster_total_affected', 
                       'disaster_count', 'disaster_damage_usd_thousands']
    available_cols = [col for col in cols_for_heatmap if col in country_df.columns]
    
    if len(available_cols) < 3:
        st.warning(f"Insufficient data columns for {country}")
        return
    
    corr_data = country_df[available_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    plt.title(f'{country}: Correlation Matrix - Trade & Disaster Metrics')
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(f'{output_dir}/{country}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_trade_network(df, year=2020, output_dir='trade_networks'):
    """Create trade network based on actual trade data"""
    os.makedirs(output_dir, exist_ok=True)
    
    year_df = df[df['Year'] == year][['Country', 'Trade (% of GDP)']].dropna()
    
    if year_df.empty:
        st.warning(f"No trade data available for year {year}")
        return
    
    G = nx.Graph()
    
    for _, row in year_df.iterrows():
        country = row['Country']
        trade_vol = row['Trade (% of GDP)']
        G.add_node(country, trade_volume=trade_vol)
    
    major_traders = year_df.nlargest(10, 'Trade (% of GDP)')['Country'].tolist()
    
    for i, country1 in enumerate(major_traders):
        for country2 in major_traders[i+1:]:
            trade1 = year_df[year_df['Country'] == country1]['Trade (% of GDP)'].iloc[0]
            trade2 = year_df[year_df['Country'] == country2]['Trade (% of GDP)'].iloc[0]
            weight = (trade1 + trade2) / 2
            G.add_edge(country1, country2, weight=weight)
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    node_sizes = [G.nodes[country]['trade_volume'] * 10 for country in G.nodes()]
    
    if G.edges():
        edge_widths = [G[u][v]['weight'] / 20 for u, v in G.edges()]
    else:
        edge_widths = []
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    if G.edges():
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Trade Network Visualization (Year {year})\nNode size = Trade volume, Edge width = Trade relationship')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(f'{output_dir}/trade_network_{year}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_shock_map(df, year=2020, output_dir='shock_maps'):
    os.makedirs(output_dir, exist_ok=True)
    
    coords = {
        'Afghanistan': (33.9, 67.7), 'Argentina': (-34.6, -58.4), 'Australia': (-25.3, 133.8),
        'Bangladesh': (23.7, 90.4), 'Belgium': (50.5, 4.5), 'Canada': (56.1, -106.3),
        'China': (35.9, 104.2), 'Croatia': (45.1, 15.2), 'France': (46.2, 2.2),
        'Germany': (51.2, 10.5), 'India': (20.6, 78.9), 'Iran': (32.4, 53.7),
        'Iraq': (33.2, 43.7), 'Israel': (31.0, 34.8), 'Italy': (41.9, 12.6),
        'Japan': (36.2, 138.3), 'Pakistan': (30.4, 69.3), 'Portugal': (39.4, -8.2),
        'Russia': (61.5, 105.3), 'Saudi Arabia': (24.0, 45.1), 'Spain': (40.5, -3.7),
        'Sri Lanka': (7.9, 80.8), 'Sweden': (64.0, 20.9), 'United Kingdom': (55.4, -3.4),
        'United States': (37.1, -95.7)
    }
    
    year_df = df[df['Year'] == year].copy()
    
    year_df['lat'] = year_df['Country'].map(lambda x: coords.get(x, (0, 0))[0])
    year_df['lon'] = year_df['Country'].map(lambda x: coords.get(x, (0, 0))[1])
    
    year_df = year_df[(year_df['lat'] != 0) | (year_df['lon'] != 0)]
    
    year_df['disaster_count'] = year_df['disaster_count'].fillna(0)
    year_df['disaster_damage_usd_thousands'] = year_df['disaster_damage_usd_thousands'].fillna(0)
    
    fig = px.scatter_geo(
        year_df, 
        lat='lat', 
        lon='lon', 
        size='disaster_count',
        color='disaster_damage_usd_thousands', 
        hover_name='Country',
        hover_data=['disaster_count', 'disaster_damage_usd_thousands'],
        title=f'Global Disaster Risk Assessment (Year {year})',
        color_continuous_scale='Reds',
        size_max=50
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            coastlinecolor="rgb(204, 204, 204)",
            projection_type='equirectangular'
        ),
        title_x=0.5
    )
    
    st.plotly_chart(fig)
    fig.write_html(f'{output_dir}/interactive_shock_map_{year}.html')
    fig.write_image(f'{output_dir}/shock_map_{year}.png', width=1200, height=700)

def identify_vulnerabilities(df, output_dir='vulnerabilities'):
    os.makedirs(output_dir, exist_ok=True)
    
    vulnerabilities = {}
    countries = df['Country'].unique()
    
    for country in countries:
        country_df = df[df['Country'] == country]
        
        if country_df.empty:
            continue
        
        metrics = {}
        
        if 'Trade (% of GDP)' in country_df.columns:
            trade_mean = country_df['Trade (% of GDP)'].mean()
            metrics['Trade Dependency'] = trade_mean if not pd.isna(trade_mean) else 0
        
        if 'disaster_total_affected' in country_df.columns:
            disaster_mean = country_df['disaster_total_affected'].mean()
            metrics['Disaster Vulnerability'] = disaster_mean if not pd.isna(disaster_mean) else 0
        
        if 'GDP growth (annual %)' in country_df.columns:
            gdp_var = country_df['GDP growth (annual %)'].var()
            metrics['Economic Volatility'] = gdp_var if not pd.isna(gdp_var) else 0
        
        if 'agr_yield_total' in country_df.columns:
            agr_yield = country_df['agr_yield_total'].mean()
            if not pd.isna(agr_yield) and agr_yield > 0:
                metrics['Agricultural Vulnerability'] = 1 / (agr_yield / 1000000)  
            else:
                metrics['Agricultural Vulnerability'] = 0
        
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        vulnerabilities[country] = sorted_metrics[:3]
    
    with open(f'{output_dir}/vulnerability_analysis.txt', 'w') as f:
        f.write("VULNERABILITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for country, vulns in vulnerabilities.items():
            f.write(f"{country}:\n")
            for i, (vuln_type, score) in enumerate(vulns, 1):
                f.write(f"  {i}. {vuln_type}: {score:.4f}\n")
            f.write("\n")
    
    return vulnerabilities

def create_summary_dashboard(df, vulnerabilities, output_dir='dashboard'):
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'Total Countries': df['Country'].nunique(),
        'Years Covered': f"{df['Year'].min()} - {df['Year'].max()}",
        'Total Disasters': df['disaster_count'].sum() if 'disaster_count' in df.columns else 0,
        'Most Vulnerable (Trade)': df.loc[df['Trade (% of GDP)'].idxmax(), 'Country'] if 'Trade (% of GDP)' in df.columns else 'N/A'
    }
    
    for key, value in stats.items():
        st.write(f"**{key}**: {value}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'Trade (% of GDP)' in df.columns:
        trade_by_country = df.groupby('Country')['Trade (% of GDP)'].mean().nlargest(10)
        trade_by_country.plot(kind='bar', ax=ax1)
        ax1.set_title('Top 10 Countries by Trade Dependency')
        ax1.set_ylabel('Trade (% of GDP)')
        ax1.tick_params(axis='x', rotation=45)
    
    if 'disaster_count' in df.columns:
        disaster_trends = df.groupby('Year')['disaster_count'].sum()
        disaster_trends.plot(kind='line', ax=ax2, marker='o')
        ax2.set_title('Global Disaster Count Trends')
        ax2.set_ylabel('Total Disasters')
        ax2.grid(True, alpha=0.3)
    
    if 'GDP growth (annual %)' in df.columns:
        df['GDP growth (annual %)'].hist(bins=30, ax=ax3, alpha=0.7)
        ax3.set_title('Distribution of GDP Growth Rates')
        ax3.set_xlabel('GDP Growth (%)')
        ax3.set_ylabel('Frequency')
    
    vuln_scores = {}
    for country, vulns in vulnerabilities.items():
        total_score = sum([score for _, score in vulns])
        vuln_scores[country] = total_score
    
    top_vulnerable = dict(sorted(vuln_scores.items(), key=lambda x: x[1], reverse=True)[:10])
    ax4.barh(list(top_vulnerable.keys()), list(top_vulnerable.values()))
    ax4.set_title('Top 10 Most Vulnerable Countries')
    ax4.set_xlabel('Vulnerability Score')
    
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(f'{output_dir}/summary_stats.txt', 'w') as f:
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 30 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

def main():
    st.write("### Trade Resilience Visualization")

    df = load_data()
    if df is None:
        st.stop()
    
    countries = df['Country'].unique()
    selected_country = st.selectbox("Select a country for correlation heatmap", countries, key="country_selectbox")
    
    st.write("#### Correlation Heatmap")
    create_heatmap(df, selected_country)
    
    st.write("#### Trade Network Visualization (Year 2020)")
    create_trade_network(df, year=2020)
    
    st.write("#### Disaster Risk Map (Year 2020)")
    create_shock_map(df, year=2020)
    
    st.write("#### Vulnerability Analysis")
    vulnerabilities = identify_vulnerabilities(df)

    create_summary_dashboard(df, vulnerabilities)

if __name__ == "__main__":
    main()
