import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        df.columns = df.columns.str.strip()
        
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        df_recent = df[df['Year'] >= 2015].copy()
        
        df_recent['Export_Import_Ratio'] = (
            df_recent['Exports of goods and services (% of GDP)'] / 
            df_recent['Imports of goods and services (% of GDP)']
        )
        
        numeric_cols = df_recent.select_dtypes(include=[np.number]).columns
        df_recent[numeric_cols] = df_recent.groupby('Country')[numeric_cols].transform(
            lambda x: x.fillna(x.mean())
        )
        
        return df_recent
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_trade_vulnerability(df):
    """Calculate trade vulnerability scores for each country"""
    latest_data = df.groupby('Country').last().reset_index()
    
    latest_data['Trade_Openness'] = latest_data['Trade (% of GDP)']
    latest_data['Import_Dependency'] = latest_data['Imports of goods and services (% of GDP)']
    latest_data['Export_Reliance'] = latest_data['Exports of goods and services (% of GDP)']
    
    scaler = StandardScaler()
    
    vulnerability_features = ['Trade_Openness', 'Import_Dependency', 'Export_Reliance']
    latest_data['Vulnerability_Score'] = scaler.fit_transform(
        latest_data[vulnerability_features]
    ).mean(axis=1) * 20 + 50
    
    latest_data['Vulnerability_Score'] = np.clip(latest_data['Vulnerability_Score'], 0, 100)
    
    return latest_data

def simulate_china_export_shock(df, shock_percentage=0.25):

    china_data = df[df['Country'] == 'China'].copy()
    if china_data.empty:
        st.warning("China not found in dataset. Using synthetic data for simulation.")
        china_exports_gdp = 20.0  
        china_gdp = 17e12  
    else:
        latest_china = china_data.iloc[-1]
        china_exports_gdp = latest_china['Exports of goods and services (% of GDP)']
        china_gdp = latest_china['GDP (current US$)']
    
    china_export_drop = china_gdp * (china_exports_gdp / 100) * shock_percentage
    
    vulnerability_data = calculate_trade_vulnerability(df)
    
    impact_results = []
    
    for _, country in vulnerability_data.iterrows():
        if country['Country'] == 'China':
            gdp_loss_pct = shock_percentage * (china_exports_gdp / 100) * 0.8  # 80% pass-through
        else:
            
            trade_dependency_factor = country['Import_Dependency'] / 100
            vulnerability_factor = country['Vulnerability_Score'] / 100
            
            china_trade_share = min(0.3, trade_dependency_factor * 0.5)  # Max 30% trade with China
            
            direct_impact = china_trade_share * shock_percentage * 0.6  # 60% pass-through
            indirect_impact = vulnerability_factor * 0.02 * shock_percentage  # Spillover effects
            
            gdp_loss_pct = (direct_impact + indirect_impact) * 100
        
        impact_results.append({
            'Country': country['Country'],
            'GDP_Loss_Percentage': gdp_loss_pct,
            'GDP_Current_USD': country['GDP (current US$)'],
            'GDP_Loss_USD': country['GDP (current US$)'] * (gdp_loss_pct / 100),
            'Trade_GDP_Ratio': country['Trade (% of GDP)'],
            'Vulnerability_Score': country['Vulnerability_Score'],
            'Import_Dependency': country['Import_Dependency']
        })
    
    return pd.DataFrame(impact_results)

def main():
    st.markdown("### China Export Shock: Global Trade Impact Analysis", 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes the cascading effects of a hypothetical 25% drop in China's exports in 2028,
    identifying the countries that would suffer the greatest GDP losses.
    """)
    
    st.sidebar.markdown("###  Data Configuration")
    
    file_path = r"datasets/processed/integrated_tren_dataset.csv"
    
    with st.spinner("Loading and processing dataset..."):
        df = load_and_prepare_data(file_path)
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
        
    st.sidebar.markdown("###  Simulation Parameters")
    shock_percentage = st.sidebar.slider(
        "China Export Drop Percentage", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.25, 
        step=0.05,
        format="%.2f"
    )
    
    top_n_countries = st.sidebar.slider(
        "Number of Most Affected Countries to Display", 
        min_value=3, 
        max_value=15, 
        value=5
    )
    
    with st.spinner("Running trade shock simulation..."):
        impact_results = simulate_china_export_shock(df, shock_percentage)
    
    impact_results = impact_results.sort_values('GDP_Loss_Percentage', ascending=False)
    
    st.markdown("#### Key Impact Metrics")    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "China Export Drop",
            f"{shock_percentage*100:.1f}%",
            f"${impact_results[impact_results['Country']=='China']['GDP_Loss_USD'].iloc[0]/1e12:.2f}T USD"
        )
    
    with col2:
        avg_loss = impact_results['GDP_Loss_Percentage'].mean()
        st.metric(
            "Average Global GDP Loss",
            f"{avg_loss:.2f}%"
        )
    
    with col3:
        total_loss = impact_results['GDP_Loss_USD'].sum()
        st.metric(
            "Total Global GDP Loss",
            f"${total_loss/1e12:.2f}T USD"
        )
    
    with col4:
        most_affected = impact_results.iloc[0]
        st.metric(
            "Most Affected Country",
            most_affected['Country'],
            f"-{most_affected['GDP_Loss_Percentage']:.2f}%"
        )
    
    st.markdown('<div class="sub-header"> Top Most Affected Countries</div>', unsafe_allow_html=True)
    
    top_affected = impact_results.head(top_n_countries)
    
    display_df = top_affected[['Country', 'GDP_Loss_Percentage', 'GDP_Loss_USD', 'Trade_GDP_Ratio', 'Vulnerability_Score']].copy()
    display_df['GDP_Loss_USD'] = display_df['GDP_Loss_USD'].apply(lambda x: f"${x/1e9:.2f}B")
    display_df['GDP_Loss_Percentage'] = display_df['GDP_Loss_Percentage'].apply(lambda x: f"{x:.3f}%")
    display_df['Trade_GDP_Ratio'] = display_df['Trade_GDP_Ratio'].apply(lambda x: f"{x:.1f}%")
    display_df['Vulnerability_Score'] = display_df['Vulnerability_Score'].apply(lambda x: f"{x:.1f}")
    
    display_df.columns = ['Country', 'GDP Loss (%)', 'GDP Loss (USD)', 'Trade/GDP Ratio', 'Vulnerability Score']
    
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("### Impact Visualizations")
    
    fig1 = px.bar(
        top_affected, 
        x='Country', 
        y='GDP_Loss_Percentage',
        title=f'Top {top_n_countries} Countries by GDP Loss Percentage',
        labels={'GDP_Loss_Percentage': 'GDP Loss (%)', 'Country': 'Country'},
        color='GDP_Loss_Percentage',
        color_continuous_scale='Reds'
    )
    fig1.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(
        impact_results, 
        x='Vulnerability_Score', 
        y='GDP_Loss_Percentage',
        size='GDP_Current_USD',
        hover_name='Country',
        title='Trade Vulnerability vs GDP Impact',
        labels={
            'Vulnerability_Score': 'Trade Vulnerability Score (0-100)',
            'GDP_Loss_Percentage': 'GDP Loss (%)'
        },
        color='GDP_Loss_Percentage',
        color_continuous_scale='RdYlBu_r'
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.histogram(
        impact_results,
        x='GDP_Loss_Percentage',
        nbins=20,
        title='Distribution of GDP Loss Across Countries',
        labels={'GDP_Loss_Percentage': 'GDP Loss (%)', 'count': 'Number of Countries'}
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    fig4 = px.scatter(
        impact_results,
        x='Trade_GDP_Ratio',
        y='GDP_Loss_Percentage',
        hover_name='Country',
        title='Trade Openness vs GDP Impact',
        labels={
            'Trade_GDP_Ratio': 'Trade as % of GDP',
            'GDP_Loss_Percentage': 'GDP Loss (%)'
        },
        size='GDP_Current_USD',
        color='Import_Dependency',
        color_continuous_scale='Viridis'
    )
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("### Economic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Findings:**")
        st.markdown(f"""
        - **Most Affected Country:** {impact_results.iloc[0]['Country']} (-{impact_results.iloc[0]['GDP_Loss_Percentage']:.3f}% GDP)
        - **Average Global Impact:** {impact_results['GDP_Loss_Percentage'].mean():.3f}% GDP loss
        - **Countries with >1% Loss:** {len(impact_results[impact_results['GDP_Loss_Percentage'] > 1])}
        - **Total Economic Loss:** ${impact_results['GDP_Loss_USD'].sum()/1e12:.2f} Trillion USD
        """)
    
    with col2:
        st.markdown("**Risk Factors:**")
        high_trade_countries = len(impact_results[impact_results['Trade_GDP_Ratio'] > 100])
        st.markdown(f"""
        - **High Trade Dependency:** {high_trade_countries} countries with Trade/GDP > 100%
        - **Vulnerability Distribution:** Mean score {impact_results['Vulnerability_Score'].mean():.1f}/100
        - **Import Dependent Economies:** {len(impact_results[impact_results['Import_Dependency'] > 50])} countries
        - **Cascading Effect Range:** {impact_results['GDP_Loss_Percentage'].min():.3f}% to {impact_results['GDP_Loss_Percentage'].max():.3f}%
        """)
    
    st.markdown('<div class="sub-header"> Download Results</div>', unsafe_allow_html=True)
    
    csv_data = impact_results.to_csv(index=False)
    st.download_button(
        label=" Download Full Impact Analysis (CSV)",
        data=csv_data,
        file_name=f"china_export_shock_impact_{shock_percentage*100:.0f}percent.csv",
        mime="text/csv"
    )
    
    with st.expander("Model Methodology"):
        st.markdown("""
        **Trade Impact Model Assumptions:**
        
        1. **Direct Impact on China:** GDP loss = Export drop × Export/GDP ratio × 0.8 (pass-through)
        
        2. **Indirect Impact on Other Countries:**
           - Trade dependency factor based on import dependency
           - Estimated China trade share (max 30% of total trade)
           - Direct impact: China trade share × shock × 0.6 (pass-through)
           - Indirect spillover: Vulnerability score × 0.02 × shock
        
        3. **Vulnerability Score Calculation:**
           - Based on trade openness, import dependency, and export reliance
           - Normalized to 0-100 scale using StandardScaler
        
        4. **Limitations:**
           - Simplified bilateral trade relationships
           - Assumes uniform pass-through rates
           - No sectoral analysis or supply chain complexity
           - Static model without dynamic adjustments
        """)

if __name__ == "__main__":
    main()