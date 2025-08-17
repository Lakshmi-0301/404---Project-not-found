import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import altair as alt

DEFAULT_DATA_PATH = Path("datasets/processed/integrated_tren_dataset.csv")

AGRI_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "imports_gdp": [
        "Imports of goods and services (% of GDP)",
        "imports_gdp_pct",
        "Imports_GDP_pct"
    ],
    "agr_production_total": ["agr_production_total"],
    "agr_area_harvested_total": ["agr_area_harvested_total"],
    "pop_total_population": ["pop_total_population___both_sexes"],
    "gdp_current": ["GDP (current US$)"],
    "food_inflation": ["Inflation, consumer prices (annual %)"],
}

@st.cache_data(show_spinner=False)
def load_agri_data() -> pd.DataFrame:
    try:
        if DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            st.error(f"Dataset not found at {DEFAULT_DATA_PATH}. Please provide the correct dataset.")
            return pd.DataFrame()
        
        df = df.drop_duplicates()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str)
        if 'Year' in df.columns or 'year' in df.columns:
            year_col = 'Year' if 'Year' in df.columns else 'year'
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def resolve_agri_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in AGRI_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_food_security_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                              latest_year_only: bool = True) -> pd.DataFrame:
    country_col = column_mapping['country']
    year_col = column_mapping['year']
    
    if not country_col or not year_col:
        st.error("Country or Year columns not found in data")
        return pd.DataFrame()
    
    if latest_year_only:
        df_latest = df.groupby(country_col)[year_col].transform('max')
        df_work = df[df[year_col] == df_latest].copy()
    else:
        df_work = df.copy()
    
    results = []
    
    for country in df_work[country_col].unique():
        if pd.isna(country) or country == 'nan':
            continue
            
        country_data = df_work[df_work[country_col] == country].iloc[0]
        
        import_intensity = 0
        if column_mapping['imports_gdp'] and not pd.isna(country_data.get(column_mapping['imports_gdp'])):
            import_intensity = float(country_data[column_mapping['imports_gdp']])
        
        agri_production = 0
        if column_mapping['agr_production_total'] and not pd.isna(country_data.get(column_mapping['agr_production_total'])):
            agri_production = float(country_data[column_mapping['agr_production_total']])
        
        population = 0
        if column_mapping['pop_total_population'] and not pd.isna(country_data.get(column_mapping['pop_total_population'])):
            population = float(country_data[column_mapping['pop_total_population']])
        
        food_inflation = 0
        if column_mapping['food_inflation'] and not pd.isna(country_data.get(column_mapping['food_inflation'])):
            food_inflation = float(country_data[column_mapping['food_inflation']])
        
 
        top_partner = "Unknown"
        top_partner_share = 0
        partner_concentration = 0
        
        if import_intensity > 50:
            top_partner_share = 0.35 
            partner_concentration = 0.20
        elif import_intensity > 30:
            top_partner_share = 0.30
            partner_concentration = 0.15
        else:
            top_partner_share = 0.25
            partner_concentration = 0.10
        
        import_component = min(import_intensity / 2, 40)
        concentration_component = partner_concentration * 100
        production_component = max(0, (1000000 - agri_production) / 20000)
        inflation_component = max(0, food_inflation * 2)
        
        fsi = (0.35 * import_component +
               0.35 * concentration_component +
               0.20 * production_component +
               0.10 * inflation_component)
        
        results.append({
            'Country': country,
            'Import_Intensity_GDP': import_intensity,
            'Top_Partner_Share': top_partner_share,
            'Partner_Concentration_HHI': partner_concentration,
            'Agri_Production': agri_production,
            'Population': population,
            'Food_Inflation': food_inflation,
            'Food_Security_Index': fsi,
            'Risk_Category': 'High' if fsi > 35 else 'Medium' if fsi > 25 else 'Low'
        })
    
    return pd.DataFrame(results).sort_values('Food_Security_Index', ascending=False)

def simulate_food_security_shock(df: pd.DataFrame, fsi_df: pd.DataFrame, 
                               column_mapping: Dict[str, Optional[str]], 
                               shock_countries: List[str], shock_magnitude: float = 0.5) -> pd.DataFrame:
    results = []
    
    for country in shock_countries:
        if country not in fsi_df['Country'].values:
            continue
            
        country_info = fsi_df[fsi_df['Country'] == country].iloc[0]
        
        import_intensity = country_info['Import_Intensity_GDP']
        top_partner_share = country_info['Top_Partner_Share']
        population = country_info['Population']
        
        import_loss_pct = top_partner_share * shock_magnitude
        food_price_increase = import_loss_pct * import_intensity / 100 * 10
        
        if import_intensity > 50:
            multiplier = 2.0
        elif import_intensity > 30:
            multiplier = 1.5
        else:
            multiplier = 1.2
            
        total_price_impact = food_price_increase * multiplier
        
        baseline_inflation = 2.0
        if column_mapping['food_inflation']:
            country_data = df[df[column_mapping['country']] == country]
            if not country_data.empty:
                latest_inflation = country_data[column_mapping['food_inflation']].dropna()
                if not latest_inflation.empty:
                    baseline_inflation = float(latest_inflation.iloc[-1])
        
        shocked_inflation = baseline_inflation + total_price_impact
        
        recovery_years = max(1, min(5, int(total_price_impact / 5)))
        
        results.append({
            'Country': country,
            'Baseline_Food_Inflation': baseline_inflation,
            'Import_Loss_Percent': import_loss_pct * 100,
            'Food_Price_Impact': total_price_impact,
            'Shocked_Food_Inflation': shocked_inflation,
            'Estimated_Recovery_Years': recovery_years,
            'Food_Security_Index': country_info['Food_Security_Index']
        })
    
    return pd.DataFrame(results)

def create_food_security_visualizations(fsi_df: pd.DataFrame, shock_results: pd.DataFrame = None):
    st.subheader("Food Security Index Rankings")
    
    if not fsi_df.empty:
        top_10 = fsi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Food_Security_Index:Q', title='Food Security Index'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Risk_Category:N',
                          scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                        range=['green', 'orange', 'red']),
                          title='Risk Level'),
            tooltip=['Country:N', 'Food_Security_Index:Q', 
                    'Top_Partner_Share:Q', 'Import_Intensity_GDP:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Food Security Risk"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Import Intensity vs Partner Concentration")
        
        scatter = alt.Chart(fsi_df).mark_circle(size=100).encode(
            x=alt.X('Import_Intensity_GDP:Q', title='Import Intensity (% of GDP)'),
            y=alt.Y('Partner_Concentration_HHI:Q', title='Partner Concentration (HHI)'),
            color=alt.Color('Food_Security_Index:Q',
                          scale=alt.Scale(scheme='viridis'),
                          title='FSI Score'),
            size=alt.Size('Top_Partner_Share:Q',
                         scale=alt.Scale(range=[50, 400]),
                         title='Top Partner Share'),
            tooltip=['Country:N', 'Food_Security_Index:Q', 
                    'Top_Partner_Share:Q', 'Import_Intensity_GDP:Q']
        ).properties(
            width=600,
            height=400,
            title="Food Security Risk Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if shock_results is not None and not shock_results.empty:
        st.subheader("Food Security Shock Simulation")
        
        shock_chart = alt.Chart(shock_results).mark_bar().encode(
            x=alt.X('Country:N', title='Country'),
            y=alt.Y('Food_Price_Impact:Q', title='Food Price Impact (% increase)'),
            color=alt.Color('Food_Price_Impact:Q',
                          scale=alt.Scale(scheme='reds'),
                          title='Impact Severity'),
            tooltip=['Country:N', 'Food_Price_Impact:Q',
                    'Shocked_Food_Inflation:Q', 'Import_Loss_Percent:Q']
        ).properties(
            width=600,
            height=300,
            title="Estimated Food Price Impact from 50% Agricultural Import Ban"
        )
        
        st.altair_chart(shock_chart, use_container_width=True)

def main():
    st.title("Food Security Risk Analysis")
    st.caption("Identifying countries vulnerable to agricultural import disruptions")
 
    with st.spinner("Loading agricultural data..."):
        df = load_agri_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_agri_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_data_only = st.sidebar.checkbox("Use latest year data only", value=True)
    shock_magnitude = st.sidebar.slider(
        "Import ban magnitude (%)",
        min_value=10, max_value=80, value=50, step=5
    ) / 100
    
    st.header("Food Security Index (FSI) Analysis")
    
    with st.spinner("Computing Food Security Index..."):
        fsi_results = compute_food_security_index(df, column_mapping, latest_data_only)
    
    if fsi_results.empty:
        st.error("Could not compute Food Security Index. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", len(fsi_results))
    
    with col2:
        high_risk = len(fsi_results[fsi_results['Risk_Category'] == 'High'])
        st.metric("High Risk Countries", high_risk, delta=f"{high_risk/len(fsi_results)*100:.1f}%")
    
    with col3:
        avg_fsi = fsi_results['Food_Security_Index'].mean()
        st.metric("Average FSI Score", f"{avg_fsi:.1f}")
    
    with col4:
        max_import_intensity = fsi_results['Import_Intensity_GDP'].max()
        st.metric("Max Import Intensity", f"{max_import_intensity:.1f}% of GDP")
    
    st.subheader("Top 3 Most Vulnerable Countries")
    
    top_3 = fsi_results.head(3)
    
    for idx, (_, country_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {country_data['Country']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("FSI Score", f"{country_data['Food_Security_Index']:.1f}")
                st.metric("Risk Level", country_data['Risk_Category'])
            
            with c2:
                st.metric("Import Intensity", f"{country_data['Import_Intensity_GDP']:.1f}% GDP")
            
            with c3:
                st.metric("Partner Share", f"{country_data['Top_Partner_Share']*100:.1f}%")
                st.metric("Concentration (HHI)", f"{country_data['Partner_Concentration_HHI']:.3f}")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Country', 'Food_Security_Index', 'Risk_Category',
                   'Import_Intensity_GDP', 'Top_Partner_Share']
    
    formatted_results = fsi_results[display_cols].copy()
    formatted_results['Top_Partner_Share'] = formatted_results['Top_Partner_Share'].apply(lambda x: f"{x*100:.1f}%")
    formatted_results['Import_Intensity_GDP'] = formatted_results['Import_Intensity_GDP'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Food_Security_Index'] = formatted_results['Food_Security_Index'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Food_Security_Index": "FSI Score",
            "Risk_Category": "Risk Level",
            "Import_Intensity_GDP": "Imports/GDP",
            "Top_Partner_Share": "Top Partner Share"
        },
        use_container_width=True
    )
    
    st.header("Food Security Shock Simulation")
    st.markdown("Simulate the impact of major agricultural import bans")
    
    available_countries = fsi_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries to simulate import ban:",
        available_countries,
        default=default_selection,
        help="Countries to include in the food security shock simulation"
    )
    
    if selected_countries:
        with st.spinner("Running food security shock simulation..."):
            shock_results = simulate_food_security_shock(
                df, fsi_results, column_mapping, selected_countries, shock_magnitude
            )
        
        if not shock_results.empty:
            st.subheader(f"Impact of {shock_magnitude*100:.0f}% Agricultural Import Ban")
            
            avg_impact = shock_results['Food_Price_Impact'].mean()
            worst_impact = shock_results['Food_Price_Impact'].max()
            worst_country = shock_results.loc[shock_results['Food_Price_Impact'].idxmax(), 'Country']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price Impact", f"{avg_impact:.2f}%")
            with col2:
                st.metric("Worst Price Impact", f"{worst_impact:.2f}%")
            with col3:
                st.metric("Most Affected", worst_country)
            
            for _, result in shock_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Country']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "Food Inflation",
                            f"{result['Shocked_Food_Inflation']:.2f}%",
                            f"{result['Food_Price_Impact']:.2f}%"
                        )
                    with c2:
                        st.metric("Import Loss", f"{result['Import_Loss_Percent']:.1f}%")
                    with c3:
                        st.metric("Recovery Time", f"{result['Estimated_Recovery_Years']:.0f} years")
                    with c4:
                        st.metric("FSI Score", f"{result['Food_Security_Index']:.1f}")
    
    create_food_security_visualizations(fsi_results, shock_results if 'shock_results' in locals() else None)
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Food Security Index (FSI) Calculation
        
        **Components (weighted):**
        - **Import Intensity (35%)**: Imports as % of GDP (proxy for agricultural imports)
        - **Partner Concentration (35%)**: Estimated Herfindahl-Hirschman Index based on import intensity
        - **Agricultural Production (20%)**: Domestic agricultural production levels
        - **Food Inflation (10%)**: Consumer price inflation (proxy for food inflation)
        
        **Risk Categories:**
        - **High Risk**: FSI > 35 (Heavy import dependence with high concentration)
        - **Medium Risk**: FSI 25-35 (Moderate import dependence)
        - **Low Risk**: FSI < 25 (Diversified import sources)
        
        ### Food Security Shock Simulation Assumptions
        
        1. **Direct Impact**: Import loss = Estimated top partner share × Shock magnitude × Import intensity
        2. **Multiplier Effects**: Import-dependent economies have higher multipliers (1.2x to 2.0x)
        3. **Recovery**: Estimated based on impact severity (1-5 years)
        4. **Baseline Inflation**: Uses historical data or assumes 2.0% baseline
        
        ### Data Sources & Limitations
        
        - Dataset lacks specific agricultural import partner data; uses import intensity to estimate concentration
        - Assumes 2-3 dominant partners based on import intensity thresholds
        - Real implementation requires detailed bilateral agricultural trade data
        - Multiplier effects are simplified estimates based on economic literature
        """)

if __name__ == "__main__":
    main()