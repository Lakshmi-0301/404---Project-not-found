import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import altair as alt

DEFAULT_DATA_PATH = Path("datasets/processed/integrated_tren_dataset.csv")

SCENARIO_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "gdp_current": ["GDP (current US$)"],
    "gdp_growth": ["GDP growth (annual %)"],
    "poverty_headcount": ["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
    "trade_dependency_index": ["Trade_Dependency_Index"],
    "resilience_score": ["Resilience_Score"],
    "shock_impact_score": ["Shock_Impact_Score"],
    "disaster_count": ["disaster_count"],
}

@st.cache_data(show_spinner=False)
def load_scenario_data() -> pd.DataFrame:
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
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def resolve_scenario_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in SCENARIO_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_baseline_projection(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                                projection_year: int = 2030) -> pd.DataFrame:
    country_col = column_mapping['country']
    year_col = column_mapping['year']
    gdp_col = column_mapping['gdp_current']
    growth_col = column_mapping['gdp_growth']
    poverty_col = column_mapping['poverty_headcount']
    
    if not all([country_col, year_col, gdp_col, growth_col, poverty_col]):
        st.error("Required columns not found in data")
        return pd.DataFrame()
    
    df_latest = df.loc[df.groupby(country_col)[year_col].idxmax()]
    
    results = []
    for _, row in df_latest.iterrows():
        country = row[country_col]
        latest_year = row[year_col]
        current_gdp = row[gdp_col]
        current_poverty = row[poverty_col]
    
        country_data = df[df[country_col] == country]
        avg_growth = country_data[growth_col].mean() / 100
        
        years_to_project = projection_year - latest_year
        
        baseline_gdp_2030 = current_gdp * (1 + avg_growth) ** years_to_project
        growth_factor = (baseline_gdp_2030 / current_gdp) - 1
        baseline_poverty_2030 = max(0, current_poverty * (1 - 0.5 * growth_factor))
        
        trade_dep = row.get(column_mapping['trade_dependency_index'], 0)
        resilience = row.get(column_mapping['resilience_score'], 0.5)
        shock_impact = row.get(column_mapping['shock_impact_score'], 0.5)
        
        results.append({
            'Country': country,
            'Baseline_GDP_2030': baseline_gdp_2030,
            'Baseline_Poverty_2030': baseline_poverty_2030,
            'Avg_Historical_Growth': avg_growth * 100,
            'Trade_Dependency': trade_dep,
            'Resilience_Score': resilience,
            'Shock_Impact': shock_impact,
        })
    
    return pd.DataFrame(results).sort_values('Baseline_GDP_2030', ascending=False)

def simulate_scenarios(df: pd.DataFrame, baseline_df: pd.DataFrame, 
                       column_mapping: Dict[str, Optional[str]], 
                       selected_countries: List[str],
                       best_intensity: float = 0.3, worst_intensity: float = 0.3) -> pd.DataFrame:
    results = []
    
    for country in selected_countries:
        if country not in baseline_df['Country'].values:
            continue
        
        baseline_info = baseline_df[baseline_df['Country'] == country].iloc[0]
        country_data = df[df[column_mapping['country']] == country].iloc[-1]
        
        trade_dep = baseline_info['Trade_Dependency']
        resilience = baseline_info['Resilience_Score']
        shock_impact = baseline_info['Shock_Impact']
        baseline_gdp = baseline_info['Baseline_GDP_2030']
        baseline_poverty = baseline_info['Baseline_Poverty_2030']
        
        trade_dep_best = trade_dep * (1 - best_intensity)
        resilience_best = min(1.0, resilience + best_intensity * 0.8)
        growth_adjust_best = best_intensity * 0.05
        gdp_best = baseline_gdp * (1 + growth_adjust_best)
        poverty_best = max(0, baseline_poverty * (1 - best_intensity * 0.7))
        
        trade_dep_worst = trade_dep * (1 + worst_intensity)
        resilience_worst = max(0, resilience - worst_intensity * 0.8)
        disaster_freq = float(country_data.get(column_mapping['disaster_count'], 0))
        shock_adjust = worst_intensity + (disaster_freq / 10) * worst_intensity
        growth_adjust_worst = -shock_adjust * 0.1
        gdp_worst = baseline_gdp * (1 + growth_adjust_worst)
        poverty_worst = baseline_poverty * (1 + shock_adjust * 0.8)
        
        results.append({
            'Country': country,
            'Best_GDP_2030': gdp_best,
            'Best_Poverty_2030': poverty_best,
            'Worst_GDP_2030': gdp_worst,
            'Worst_Poverty_2030': poverty_worst,
            'GDP_Delta_Best': (gdp_best - baseline_gdp) / baseline_gdp * 100,
            'GDP_Delta_Worst': (gdp_worst - baseline_gdp) / baseline_gdp * 100,
            'Poverty_Delta_Best': poverty_best - baseline_poverty,
            'Poverty_Delta_Worst': poverty_worst - baseline_poverty,
        })
    
    return pd.DataFrame(results)

def create_scenario_visualizations(baseline_df: pd.DataFrame, scenario_results: pd.DataFrame = None):
    st.subheader("Baseline 2030 Projections")
    
    if not baseline_df.empty:
        top_10 = baseline_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Baseline_GDP_2030:Q', title='Baseline GDP 2030 (US$)'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Avg_Historical_Growth:Q',
                            scale=alt.Scale(scheme='blues'),
                            title='Avg Growth (%)'),
            tooltip=['Country:N', 'Baseline_GDP_2030:Q', 'Baseline_Poverty_2030:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Baseline GDP 2030"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Resilience vs Trade Dependency")
        
        scatter = alt.Chart(baseline_df).mark_circle(size=100).encode(
            x=alt.X('Trade_Dependency:Q', title='Trade Dependency Index'),
            y=alt.Y('Resilience_Score:Q', title='Resilience Score'),
            color=alt.Color('Shock_Impact:Q',
                            scale=alt.Scale(scheme='reds'),
                            title='Shock Impact'),
            size=alt.Size('Avg_Historical_Growth:Q',
                          scale=alt.Scale(range=[50, 400]),
                          title='Avg Growth (%)'),
            tooltip=['Country:N', 'Trade_Dependency:Q', 'Resilience_Score:Q']
        ).properties(
            width=600,
            height=400,
            title="Baseline Scenario Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if scenario_results is not None and not scenario_results.empty:
        st.subheader("Scenario Comparisons")
        
        # Melt for stacked bar
        melted = scenario_results.melt(id_vars=['Country'], 
                                       value_vars=['Best_GDP_2030', 'Worst_GDP_2030'],
                                       var_name='Scenario', value_name='GDP_2030')
        
        scenario_chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('GDP_2030:Q', title='GDP 2030 (US$)'),
            y=alt.Y('Country:N', title='Country'),
            color=alt.Color('Scenario:N',
                            scale=alt.Scale(domain=['Best_GDP_2030', 'Worst_GDP_2030'],
                                            range=['green', 'red']),
                            title='Scenario'),
            tooltip=['Country:N', 'Scenario:N', 'GDP_2030:Q']
        ).properties(
            width=600,
            height=300,
            title="GDP 2030 under Best and Worst Cases"
        )
        
        st.altair_chart(scenario_chart, use_container_width=True)
        
        # Poverty chart
        melted_pov = scenario_results.melt(id_vars=['Country'], 
                                           value_vars=['Best_Poverty_2030', 'Worst_Poverty_2030'],
                                           var_name='Scenario', value_name='Poverty_2030')
        
        pov_chart = alt.Chart(melted_pov).mark_bar().encode(
            x=alt.X('Poverty_2030:Q', title='Poverty Rate 2030 (%)'),
            y=alt.Y('Country:N', title='Country'),
            color=alt.Color('Scenario:N',
                            scale=alt.Scale(domain=['Best_Poverty_2030', 'Worst_Poverty_2030'],
                                            range=['green', 'red']),
                            title='Scenario'),
            tooltip=['Country:N', 'Scenario:N', 'Poverty_2030:Q']
        ).properties(
            width=600,
            height=300,
            title="Poverty Rate 2030 under Best and Worst Cases"
        )
        
        st.altair_chart(pov_chart, use_container_width=True)

def main():
    st.title("2030 Economic Scenario Projections")
    st.caption("Modeling GDP and poverty rates under best and worst case scenarios")
 
    with st.spinner("Loading scenario data..."):
        df = load_scenario_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_scenario_columns(df)

    st.sidebar.subheader("Projection Parameters")
    projection_year = st.sidebar.slider("Projection Year", 2025, 2050, 2030)
    best_intensity = st.sidebar.slider("Best Case Intensity", 0.1, 0.5, 0.3, 0.05)
    worst_intensity = st.sidebar.slider("Worst Case Intensity", 0.1, 0.5, 0.3, 0.05)
    
    st.header("Baseline 2030 Projections")
    
    with st.spinner("Computing baseline projections..."):
        baseline_results = compute_baseline_projection(df, column_mapping, projection_year)
    
    if baseline_results.empty:
        st.error("Could not compute baseline projections. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Projected", len(baseline_results))
    
    with col2:
        avg_gdp = baseline_results['Baseline_GDP_2030'].mean() / 1e9
        st.metric("Avg GDP 2030", f"${avg_gdp:.1f}B")
    
    with col3:
        avg_poverty = baseline_results['Baseline_Poverty_2030'].mean()
        st.metric("Avg Poverty 2030", f"{avg_poverty:.1f}%")
    
    with col4:
        avg_growth = baseline_results['Avg_Historical_Growth'].mean()
        st.metric("Avg Historical Growth", f"{avg_growth:.1f}%")
    
    st.subheader("Top 3 Largest Economies in 2030 (Baseline)")
    
    top_3 = baseline_results.head(3)
    
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {row['Country']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("GDP 2030", f"${row['Baseline_GDP_2030']/1e9:.1f}B")
            with c2:
                st.metric("Poverty 2030", f"{row['Baseline_Poverty_2030']:.1f}%")
            with c3:
                st.metric("Avg Growth", f"{row['Avg_Historical_Growth']:.1f}%")
    
    st.subheader("Complete Baseline Projections")
    
    display_cols = ['Country', 'Baseline_GDP_2030', 'Baseline_Poverty_2030', 'Avg_Historical_Growth']
    
    formatted_results = baseline_results[display_cols].copy()
    formatted_results['Baseline_GDP_2030'] = formatted_results['Baseline_GDP_2030'].apply(lambda x: f"${x/1e9:.1f}B")
    formatted_results['Baseline_Poverty_2030'] = formatted_results['Baseline_Poverty_2030'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Avg_Historical_Growth'] = formatted_results['Avg_Historical_Growth'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Baseline_GDP_2030": "GDP 2030",
            "Baseline_Poverty_2030": "Poverty 2030",
            "Avg_Historical_Growth": "Avg Growth"
        },
        use_container_width=True
    )
    
    st.header("Scenario Simulations")
    st.markdown("Compare best and worst case scenarios for 2030")
    
    available_countries = baseline_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries for scenarios:",
        available_countries,
        default=default_selection,
        help="Countries to simulate best and worst cases"
    )
    
    if selected_countries:
        with st.spinner("Running scenario simulations..."):
            scenario_results = simulate_scenarios(
                df, baseline_results, column_mapping, selected_countries, 
                best_intensity, worst_intensity
            )
        
        if not scenario_results.empty:
            st.subheader("Scenario Results")
            
            avg_gdp_delta_best = scenario_results['GDP_Delta_Best'].mean()
            avg_gdp_delta_worst = scenario_results['GDP_Delta_Worst'].mean()
            avg_pov_delta_best = scenario_results['Poverty_Delta_Best'].mean()
            avg_pov_delta_worst = scenario_results['Poverty_Delta_Worst'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg GDP Delta Best", f"{avg_gdp_delta_best:.1f}%")
            with col2:
                st.metric("Avg GDP Delta Worst", f"{avg_gdp_delta_worst:.1f}%")
            with col3:
                st.metric("Avg Poverty Delta Best", f"{avg_pov_delta_best:.1f}pp")
            with col4:
                st.metric("Avg Poverty Delta Worst", f"{avg_pov_delta_worst:.1f}pp")
            
            for _, result in scenario_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Country']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Best GDP 2030", f"${result['Best_GDP_2030']/1e9:.1f}B")
                        st.metric("Best Poverty 2030", f"{result['Best_Poverty_2030']:.1f}%")
                    with c2:
                        st.metric("Worst GDP 2030", f"${result['Worst_GDP_2030']/1e9:.1f}B")
                        st.metric("Worst Poverty 2030", f"{result['Worst_Poverty_2030']:.1f}%")
                    with c3:
                        st.metric("GDP Delta Best", f"{result['GDP_Delta_Best']:.1f}%")
                        st.metric("Poverty Delta Best", f"{result['Poverty_Delta_Best']:.1f}pp")
                    with c4:
                        st.metric("GDP Delta Worst", f"{result['GDP_Delta_Worst']:.1f}%")
                        st.metric("Poverty Delta Worst", f"{result['Poverty_Delta_Worst']:.1f}pp")
    
    create_scenario_visualizations(baseline_results, scenario_results if 'scenario_results' in locals() else None)
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Baseline Projection Calculation
        
        - Use historical average GDP growth rate per country
        - Project GDP using compound growth: GDP_2030 = Current_GDP * (1 + avg_growth)^years
        - Project poverty using elasticity -0.5: Poverty_2030 = Current_Poverty * (1 - 0.5 * total_growth)
        
        ### Best Case Scenario (Trade Diversification + Resilience Investments)
        
        - Reduce trade dependency by intensity
        - Increase resilience by 80% of intensity
        - Boost GDP growth by 5% per intensity point
        - Reduce poverty by 70% of intensity
        
        ### Worst Case Scenario (Recurring Disasters + Trade Concentration)
        
        - Increase trade dependency by intensity
        - Decrease resilience by 80% of intensity
        - Penalty to GDP growth based on intensity + disaster frequency
        - Increase poverty based on shock adjustment
        
        ### Data Sources & Limitations
        
        - Projections based on historical averages from dataset
        - Assumes constant growth rates; no external shocks modeled in baseline
        - Simplified multipliers for scenarios based on economic literature approximations
        - Poverty elasticity fixed at -0.5; may vary by country
        """)

if __name__ == "__main__":
    main()