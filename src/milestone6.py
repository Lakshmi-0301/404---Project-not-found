import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import altair as alt

DEFAULT_DATA_PATH = Path("datasets/processed/integrated_tren_dataset.csv")

EXPORT_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "exports_gdp": [
        "Exports of goods and services (% of GDP)",
        "exports_gdp_pct",
        "Exports_GDP_pct"
    ],
    "trade_gdp": [
        "Trade (% of GDP)",
        "trade_gdp_pct",
        "Trade_GDP_pct"
    ],
    "population": ["pop_total_population___both_sexes"],
}

@st.cache_data(show_spinner=False)
def load_export_data() -> pd.DataFrame:
    try:
        if DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            return pd.DataFrame()
        
        df = df.drop_duplicates()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str)
        if 'Year' in df.columns or 'year' in df.columns:
            year_col = 'Year' if 'Year' in df.columns else 'year'
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        return df
    except Exception as e:
        return pd.DataFrame()

def resolve_export_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in EXPORT_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_export_dependency_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
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
    
    median_ages = {
        "Japan": 50.2, "Italy": 48.3, "Germany": 48.4, "South Korea": 46.3,
        "Spain": 46.4, "Portugal": 46.7, "Netherlands": 43.4, "Belgium": 42.3,
        "Canada": 42.6, "United Kingdom": 41.2, "United States": 38.9, "China": 39.8,
        "Mexico": 30.7, "India": 29.5, "Afghanistan": 19.5
    }
    
    for country in df_work[country_col].unique():
        if pd.isna(country) or country == 'nan':
            continue
            
        country_data = df_work[df_work[country_col] == country].iloc[0]
        
        export_intensity = 0
        if column_mapping['exports_gdp'] and not pd.isna(country_data.get(column_mapping['exports_gdp'])):
            export_intensity = float(country_data[column_mapping['exports_gdp']])
        elif column_mapping['trade_gdp'] and not pd.isna(country_data.get(column_mapping['trade_gdp'])):
            export_intensity = float(country_data[column_mapping['trade_gdp']]) * 0.5
        
        population = 0
        if column_mapping['population'] and not pd.isna(country_data.get(column_mapping['population'])):
            population = float(country_data[column_mapping['population']])
        
        median_age = median_ages.get(country, 40.0)
        labor_vulnerability = max(0, (median_age - 40) / 5)
        
        edi = (0.6 * min(export_intensity / 2, 50) + 
               0.4 * min(labor_vulnerability * 10, 50)) 
        
        results.append({
            'Country': country,
            'Export_Intensity_GDP': export_intensity,
            'Median_Age': median_age,
            'Population': population,
            'Export_Dependency_Index': edi,
            'Risk_Category': 'High' if edi > 30 else 'Medium' if edi > 20 else 'Low'
        })
    
    return pd.DataFrame(results).sort_values('Export_Dependency_Index', ascending=False)

def simulate_productivity_impact(df: pd.DataFrame, edi_df: pd.DataFrame, 
                                column_mapping: Dict[str, Optional[str]], 
                                shock_countries: List[str], age_increases: List[int] = [5, 10, 15]) -> pd.DataFrame:
    results = []
    
    for country in shock_countries:
        if country not in edi_df['Country'].values:
            continue
            
        country_info = edi_df[edi_df['Country'] == country].iloc[0]
        
        export_intensity = country_info['Export_Intensity_GDP']
        baseline_median_age = country_info['Median_Age']
        
        def productivity(age):
            return max(0, 1 - 0.001 * (age - 45) ** 2)
        
        baseline_productivity = productivity(baseline_median_age)
        
        for age_increase in age_increases:
            shocked_age = baseline_median_age + age_increase
            shocked_productivity = productivity(shocked_age)
            productivity_impact = ((shocked_productivity - baseline_productivity) / baseline_productivity) * 100
            
            export_impact = productivity_impact * (export_intensity / 100)
            
            results.append({
                'Country': country,
                'Age_Increase_Years': age_increase,
                'Baseline_Median_Age': baseline_median_age,
                'Shocked_Median_Age': shocked_age,
                'Productivity_Impact_Percent': productivity_impact,
                'Export_Output_Impact_Percent': export_impact,
                'Export_Dependency_Index': country_info['Export_Dependency_Index']
            })
    
    return pd.DataFrame(results)

def create_export_visualizations(edi_df: pd.DataFrame, shock_results: pd.DataFrame = None):
    st.subheader("Export Dependency Index Rankings")
    
    if not edi_df.empty:
        top_10 = edi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Export_Dependency_Index:Q', title='Export Dependency Index'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Risk_Category:N',
                          scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                        range=['green', 'orange', 'red']),
                          title='Risk Level'),
            tooltip=['Country:N', 'Export_Dependency_Index:Q', 'Export_Intensity_GDP:Q', 'Median_Age:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Export Dependency Risk"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Export Intensity vs Median Age")
        
        scatter = alt.Chart(edi_df).mark_circle(size=100).encode(
            x=alt.X('Export_Intensity_GDP:Q', title='Export Intensity (% of GDP)'),
            y=alt.Y('Median_Age:Q', title='Median Age'),
            color=alt.Color('Export_Dependency_Index:Q',
                          scale=alt.Scale(scheme='viridis'),
                          title='EDI Score'),
            size=alt.Size('Population:Q',
                         scale=alt.Scale(range=[50, 400]),
                         title='Population'),
            tooltip=['Country:N', 'Export_Dependency_Index:Q', 'Export_Intensity_GDP:Q', 'Median_Age:Q']
        ).properties(
            width=600,
            height=400,
            title="Export Dependency Risk Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if shock_results is not None and not shock_results.empty:
        st.subheader("Productivity Impact Simulation")
        
        for age_increase in [5, 10, 15]:
            age_data = shock_results[shock_results['Age_Increase_Years'] == age_increase]
            if not age_data.empty:
                st.write(f"Rendering graph for Median Age Increase of {age_increase} Years with {len(age_data)} countries")  # Debug output
                shock_chart = alt.Chart(age_data).mark_bar().encode(
                    x=alt.X('Country:N', title='Country'),
                    y=alt.Y('Productivity_Impact_Percent:Q', title='Productivity Impact (% Change)'),
                    color=alt.Color('Productivity_Impact_Percent:Q',
                                  scale=alt.Scale(scheme='blues'),
                                  title='Impact (%)'),
                    tooltip=[
                        'Country:N',
                        alt.Tooltip('Age_Increase_Years:N', title='Age Increase (Years)'),
                        alt.Tooltip('Productivity_Impact_Percent:Q', title='Productivity Impact (%)'),
                        alt.Tooltip('Export_Output_Impact_Percent:Q', title='Export Output Impact (%)'),
                        alt.Tooltip('Shocked_Median_Age:Q', title='New Median Age')
                    ]
                ).properties(
                    width=600,
                    height=300,
                    title=f"Productivity Impacts for Median Age Increase of {age_increase} Years"
                )
                
                st.altair_chart(shock_chart, use_container_width=True)
            else:
                st.write(f"No data available for Median Age Increase of {age_increase} Years")  # Debug output

def main():
    st.title("Export Sector Risk Analysis: Ageing Demographics")
    st.caption("Identifying countries vulnerable to labor shortages in export sectors")
 
    with st.spinner("Loading export data..."):
        df = load_export_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_export_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_data_only = st.sidebar.checkbox("Use latest year data only", value=True)
    
    st.header("Export Dependency Index (EDI) Analysis")
    
    with st.spinner("Computing Export Dependency Index..."):
        edi_results = compute_export_dependency_index(df, column_mapping, latest_data_only)
    
    if edi_results.empty:
        st.error("Could not compute Export Dependency Index. Please check your data.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Countries Analyzed", len(edi_results))
    
    with col2:
        high_risk = len(edi_results[edi_results['Risk_Category'] == 'High'])
        st.metric("High Risk Countries", high_risk, delta=f"{high_risk/len(edi_results)*100:.1f}%")
    
    with col3:
        avg_edi = edi_results['Export_Dependency_Index'].mean()
        st.metric("Average EDI Score", f"{avg_edi:.1f}")
    
    st.subheader("Top 3 Most Vulnerable Countries")
    
    top_3 = edi_results.head(3)
    
    for idx, (_, country_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {country_data['Country']}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("EDI Score", f"{country_data['Export_Dependency_Index']:.1f}")
                st.metric("Risk Level", country_data['Risk_Category'])
            
            with c2:
                st.metric("Export Intensity", f"{country_data['Export_Intensity_GDP']:.1f}% GDP")
                st.metric("Median Age", f"{country_data['Median_Age']:.1f}")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Country', 'Export_Dependency_Index', 'Risk_Category', 'Export_Intensity_GDP', 'Median_Age']
    
    formatted_results = edi_results[display_cols].copy()
    formatted_results['Export_Intensity_GDP'] = formatted_results['Export_Intensity_GDP'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Export_Dependency_Index'] = formatted_results['Export_Dependency_Index'].apply(lambda x: f"{x:.1f}")
    formatted_results['Median_Age'] = formatted_results['Median_Age'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Export_Dependency_Index": "EDI Score",
            "Risk_Category": "Risk Level",
            "Export_Intensity_GDP": "Exports/GDP",
            "Median_Age": "Median Age"
        },
        use_container_width=True
    )
    
    st.header("Productivity Impact Simulation")
    st.markdown("Simulate productivity impacts from median age increases in export sectors")
    
    available_countries = edi_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries to simulate age increase:",
        available_countries,
        default=default_selection,
        help="Countries to include in the productivity impact simulation",
        key="milestone6_country_select"
    )
    
    if selected_countries:
        with st.spinner("Running productivity impact simulation..."):
            shock_results = simulate_productivity_impact(
                df, edi_results, column_mapping, selected_countries
            )
        
        if not shock_results.empty:
            st.subheader("Impact of Median Age Increases")
            
            avg_impact_5 = shock_results[shock_results['Age_Increase_Years'] == 5]['Productivity_Impact_Percent'].mean()
            avg_impact_10 = shock_results[shock_results['Age_Increase_Years'] == 10]['Productivity_Impact_Percent'].mean()
            avg_impact_15 = shock_results[shock_results['Age_Increase_Years'] == 15]['Productivity_Impact_Percent'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Impact (+5 Years)", f"{avg_impact_5:.2f}%")
            with col2:
                st.metric("Avg Impact (+10 Years)", f"{avg_impact_10:.2f}%")
            with col3:
                st.metric("Avg Impact (+15 Years)", f"{avg_impact_15:.2f}%")
            
            st.subheader("Simulation Results")
            formatted_shock = shock_results.copy()
            formatted_shock['Productivity_Impact_Percent'] = formatted_shock['Productivity_Impact_Percent'].apply(lambda x: f"{x:.2f}%")
            formatted_shock['Export_Output_Impact_Percent'] = formatted_shock['Export_Output_Impact_Percent'].apply(lambda x: f"{x:.2f}%")
            formatted_shock['Baseline_Median_Age'] = formatted_shock['Baseline_Median_Age'].apply(lambda x: f"{x:.1f}")
            formatted_shock['Shocked_Median_Age'] = formatted_shock['Shocked_Median_Age'].apply(lambda x: f"{x:.1f}")
            formatted_shock['Export_Dependency_Index'] = formatted_shock['Export_Dependency_Index'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(
                formatted_shock[['Country', 'Age_Increase_Years', 'Baseline_Median_Age', 
                               'Shocked_Median_Age', 'Productivity_Impact_Percent', 
                               'Export_Output_Impact_Percent', 'Export_Dependency_Index']],
                column_config={
                    "Country": "Country",
                    "Age_Increase_Years": "Age Increase (Years)",
                    "Baseline_Median_Age": "Baseline Median Age",
                    "Shocked_Median_Age": "Shocked Median Age",
                    "Productivity_Impact_Percent": "Productivity Impact (%)",
                    "Export_Output_Impact_Percent": "Export Output Impact (%)",
                    "Export_Dependency_Index": "EDI Score"
                },
                use_container_width=True
            )
    
    create_export_visualizations(edi_results, shock_results if 'shock_results' in locals() else None)
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Export Dependency Index (EDI) Calculation
        
        **Components (weighted):**
        - **Export Intensity (60%)**: Exports as % of GDP (or half of Trade % GDP if exports unavailable)
        - **Labor Vulnerability (40%)**: Based on median age, with higher ages indicating greater risk of labor shortages
        
        **Risk Categories:**
        - **High Risk**: EDI > 30 (High export reliance and ageing workforce)
        - **Medium Risk**: EDI 20-30 (Moderate export reliance or ageing)
        - **Low Risk**: EDI < 20 (Diversified economy or younger workforce)
        
        ### Productivity Impact Simulation Assumptions
        
        1. **Productivity Model**: Quadratic function (1 - 0.001 × (age - 45)²), peaking at age 45
        2. **Impact Calculation**: Productivity change scales with export intensity for export output impact
        3. **Median Age**: Uses estimated 2025 values for key countries; defaults to global average (40) for others
        4. **Limitations**: Dataset lacks direct median age data; uses external estimates for key countries
        
        ### Data Sources & Limitations
        
        - Median age data sourced from UN/World Bank 2025 estimates
        - Assumes labor-intensive export sectors (e.g., manufacturing) are most affected
        - Productivity model is simplified; real impacts vary by sector and mitigation strategies
        """)

if __name__ == "__main__":
    main()