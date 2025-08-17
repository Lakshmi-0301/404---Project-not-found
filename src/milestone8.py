import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import altair as alt

DEFAULT_DATA_PATH = Path("datasets/processed/integrated_tren_dataset.csv")

TRADE_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "exports_gdp": ["Exports of goods and services (% of GDP)"],
    "imports_gdp": ["Imports of goods and services (% of GDP)", "imports_gdp_pct", "Imports_GDP_pct"],
    "trade_gdp": ["Trade (% of GDP)"],
    "gdp_current": ["GDP (current US$)"],
    "trade_dependency_index": ["Trade_Dependency_Index"],
    "resilience_score": ["Resilience_Score"],
}

def extract_trade_partner_shares_from_data(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    country_col = column_mapping.get('country')
    year_col = column_mapping.get('year')
    exports_gdp_col = column_mapping.get('exports_gdp')
    imports_gdp_col = column_mapping.get('imports_gdp')
    trade_gdp_col = column_mapping.get('trade_gdp')
    gdp_col = column_mapping.get('gdp_current')
    
    if not all([country_col, year_col, exports_gdp_col, gdp_col]):
        st.error("Required columns not found in dataset for trade partner estimation")
        return {}
    
    df_latest = df[df[year_col] == df.groupby(country_col)[year_col].transform('max')]
    
    df_latest = df_latest.copy()
    df_latest['trade_intensity'] = df_latest[exports_gdp_col].fillna(0) + df_latest[imports_gdp_col].fillna(0)
    df_latest['gdp_billions'] = df_latest[gdp_col].fillna(0) / 1e9
    
    major_economies = df_latest[df_latest['gdp_billions'] > 500]['Country'].tolist()
    high_trade_economies = df_latest[df_latest['trade_intensity'] > 50]['Country'].tolist()
    
    trade_partner_shares = {}
    
    known_relationships = [
        ("Canada", "United States", 0.75, 0.15),
        ("United States", "Mexico", 0.15, 0.25),
        
        ("China", "United States", 0.18, 0.08),   
        ("China", "Japan", 0.08, 0.18),           
        ("Japan", "China", 0.22, 0.10),           
        
        ("Germany", "France", 0.08, 0.12),
        ("France", "Germany", 0.15, 0.08),
        ("Germany", "United Kingdom", 0.06, 0.10), 
        
        ("South Korea", "China", 0.25, 0.08),
        ("Australia", "China", 0.35, 0.05),
        ("Brazil", "China", 0.28, 0.02),
        ("India", "United States", 0.18, 0.05),
    ]
    
    available_countries = set(df_latest[country_col].unique())
    
    for country_a, country_b, share_a_to_b, share_b_to_a in known_relationships:
        if country_a in available_countries and country_b in available_countries:
            trade_partner_shares[(country_a, country_b)] = {
                "export_A_to_B_share": share_a_to_b,
                "export_B_to_A_share": share_b_to_a
            }
    
    for country in high_trade_economies[:10]:
        if country in major_economies:
            for partner in major_economies:
                if country != partner and (country, partner) not in trade_partner_shares:
                    country_data = df_latest[df_latest[country_col] == country].iloc[0]
                    partner_data = df_latest[df_latest[country_col] == partner].iloc[0]
                    
                    if country_data['gdp_billions'] < partner_data['gdp_billions']:
                        share_to_partner = min(0.25, country_data['trade_intensity'] / 200)
                        share_from_partner = min(0.15, partner_data['trade_intensity'] / 300)
                    else:
                        share_to_partner = min(0.15, country_data['trade_intensity'] / 300)
                        share_from_partner = min(0.25, partner_data['trade_intensity'] / 200)
                    
                    if share_to_partner > 0.05 or share_from_partner > 0.05:
                        trade_partner_shares[(country, partner)] = {
                            "export_A_to_B_share": share_to_partner,
                            "export_B_to_A_share": share_from_partner
                        }
    
    st.info(f"Generated {len(trade_partner_shares)} trade relationships from data analysis")
    
    return trade_partner_shares

def generate_bilateral_pairs(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> List[Dict]:

    country_col = column_mapping.get('country')
    year_col = column_mapping.get('year')
    exports_gdp_col = column_mapping.get('exports_gdp')
    gdp_col = column_mapping.get('gdp_current')
    
    if not all([country_col, year_col, exports_gdp_col, gdp_col]):
        st.error("Required columns not found in dataset for bilateral trade estimation")
        return []
    
    trade_partner_shares = extract_trade_partner_shares_from_data(df, column_mapping)
    
    if not trade_partner_shares:
        st.warning("No trade relationships could be derived from the data")
        return []
    
    df_latest = df[df[year_col] == df.groupby(country_col)[year_col].transform('max')]
    
    bilateral_pairs = []
    for (country_A, country_B), shares in trade_partner_shares.items():
        data_A = df_latest[df_latest[country_col] == country_A]
        data_B = df_latest[df_latest[country_col] == country_B]
        
        if data_A.empty or data_B.empty:
            continue 
        
        data_A = data_A.iloc[0]
        data_B = data_B.iloc[0]
        
        exports_A = data_A[exports_gdp_col] if pd.notna(data_A[exports_gdp_col]) else 0
        exports_B = data_B[exports_gdp_col] if pd.notna(data_B[exports_gdp_col]) else 0
        gdp_A = data_A[gdp_col] if pd.notna(data_A[gdp_col]) else 0
        gdp_B = data_B[gdp_col] if pd.notna(data_B[gdp_col]) else 0
        
        if gdp_A == 0 or gdp_B == 0:
            continue
        
        total_exports_A = (float(exports_A) * float(gdp_A) / 100) / 1e9
        total_exports_B = (float(exports_B) * float(gdp_B) / 100) / 1e9
        
        export_A_to_B = total_exports_A * shares["export_A_to_B_share"]
        export_B_to_A = total_exports_B * shares["export_B_to_A_share"]
        trade_volume = export_A_to_B + export_B_to_A
        
        if trade_volume > 1.0:
            bilateral_pairs.append({
                "pair": (country_A, country_B),
                "export_A_to_B": export_A_to_B,
                "export_B_to_A": export_B_to_A,
                "trade_volume": trade_volume
            })
    
    return bilateral_pairs

@st.cache_data(show_spinner=False)
def load_trade_data() -> pd.DataFrame:
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
        
        df = df[df[year_col] >= 2015].copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby('Country')[numeric_cols].transform(
            lambda x: x.fillna(x.mean())
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def resolve_trade_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in TRADE_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_mutual_benefit_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                                 latest_year_only: bool = True) -> pd.DataFrame:
    country_col = column_mapping['country']
    year_col = column_mapping['year']
    
    if not country_col or not year_col:
        st.error("Country or Year columns not found in data")
        return pd.DataFrame()
    
    if latest_year_only:
        df_latest = df[df[year_col] == df.groupby(country_col)[year_col].transform('max')]
    else:
        df_latest = df.copy()
    
    BILATERAL_PAIRS = generate_bilateral_pairs(df, column_mapping)
    
    if not BILATERAL_PAIRS:
        st.error("No trade pairs could be generated from the data")
        return pd.DataFrame()
    
    results = []
    
    for pair_data in BILATERAL_PAIRS:
        country_A, country_B = pair_data["pair"]
        export_A_to_B = pair_data["export_A_to_B"]
        export_B_to_A = pair_data["export_B_to_A"]
        trade_volume = pair_data["trade_volume"]
        
        data_A = df_latest[df_latest[country_col] == country_A]
        data_B = df_latest[df_latest[country_col] == country_B]
        
        if data_A.empty or data_B.empty:
            continue
        
        data_A = data_A.iloc[0]
        data_B = data_B.iloc[0]
        
        gdp_A = float(data_A.get(column_mapping['gdp_current'], 1e9)) / 1e9
        gdp_B = float(data_B.get(column_mapping['gdp_current'], 1e9)) / 1e9
        
        if gdp_A == 0 or gdp_B == 0:
            continue
        
        export_pct_A = (export_A_to_B / gdp_A) * 100
        export_pct_B = (export_B_to_A / gdp_B) * 100
        
        mutual_benefit = np.sqrt(export_pct_A * export_pct_B)
        mutual_benefit = np.clip(mutual_benefit, 0, 100)
        
        dependency_A = float(data_A.get(column_mapping['trade_dependency_index'], 0))
        dependency_B = float(data_B.get(column_mapping['trade_dependency_index'], 0))
        
        results.append({
            'Pair': f"{country_A} - {country_B}",
            'Trade_Volume_Billion': float(trade_volume),
            'Export_Pct_A': float(export_pct_A),
            'Export_Pct_B': float(export_pct_B),
            'Mutual_Benefit_Index': float(mutual_benefit),
            'Dependency_A': float(dependency_A),
            'Dependency_B': float(dependency_B),
            'Risk_Category': 'High' if mutual_benefit > 5 else 'Medium' if mutual_benefit > 2 else 'Low'
        })
    
    results_df = pd.DataFrame(results).sort_values('Mutual_Benefit_Index', ascending=False)
    
    numeric_cols = ['Trade_Volume_Billion', 'Export_Pct_A', 'Export_Pct_B', 'Mutual_Benefit_Index', 
                    'Dependency_A', 'Dependency_B']
    results_df[numeric_cols] = results_df[numeric_cols].fillna(0.0)
    
    return results_df

def simulate_trade_collapse(df: pd.DataFrame, mbi_df: pd.DataFrame, 
                            column_mapping: Dict[str, Optional[str]], 
                            shock_pairs: List[str], shock_magnitude: float = 1.0) -> pd.DataFrame:
    results = []
    
    for pair_str in shock_pairs:
        if pair_str not in mbi_df['Pair'].values:
            st.warning(f"Skipping pair {pair_str}: Not found in MBI results")
            continue
        
        pair_info = mbi_df[mbi_df['Pair'] == pair_str].iloc[0]
        
        country_A, country_B = pair_str.split(' - ')
        
        data_A = df[df[column_mapping['country']] == country_A]
        data_B = df[df[column_mapping['country']] == country_B]
        
        if data_A.empty or data_B.empty:
            st.warning(f"Skipping pair {pair_str}: Data not found")
            continue
        
        data_A = data_A.iloc[-1]
        data_B = data_B.iloc[-1]
        
        import_intensity_A = float(data_A.get(column_mapping['imports_gdp'], 0))
        import_intensity_B = float(data_B.get(column_mapping['imports_gdp'], 0))
        
        export_pct_A = pair_info['Export_Pct_A']
        export_pct_B = pair_info['Export_Pct_B']
        
        multiplier_A = 1.0
        if import_intensity_A > 50:
            multiplier_A = 2.0
        elif import_intensity_A > 30:
            multiplier_A = 1.5
        
        multiplier_B = 1.0
        if import_intensity_B > 50:
            multiplier_B = 2.0
        elif import_intensity_B > 30:
            multiplier_B = 1.5
        
        gdp_impact_A = -export_pct_A * shock_magnitude * multiplier_A
        gdp_impact_B = -export_pct_B * shock_magnitude * multiplier_B
        
        recovery_years_A = max(1, min(5, int(abs(gdp_impact_A) / 5)))
        recovery_years_B = max(1, min(5, int(abs(gdp_impact_B) / 5)))
        
        results.append({
            'Pair': pair_str,
            'Country_A': country_A,
            'Country_B': country_B,
            'GDP_Impact_A': float(gdp_impact_A),
            'GDP_Impact_B': float(gdp_impact_B),
            'Multiplier_A': float(multiplier_A),
            'Multiplier_B': float(multiplier_B),
            'Recovery_Years_A': int(recovery_years_A),
            'Recovery_Years_B': int(recovery_years_B),
            'Mutual_Benefit_Index': float(pair_info['Mutual_Benefit_Index'])
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        numeric_cols = ['GDP_Impact_A', 'GDP_Impact_B', 'Multiplier_A', 'Multiplier_B', 
                        'Recovery_Years_A', 'Recovery_Years_B', 'Mutual_Benefit_Index']
        results_df[numeric_cols] = results_df[numeric_cols].fillna(0.0)
    
    return results_df

def create_trade_visualizations(mbi_df: pd.DataFrame, shock_results: pd.DataFrame = None):
    st.subheader("Mutual Benefit Index Rankings")
    
    if not mbi_df.empty:
        top_10 = mbi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Mutual_Benefit_Index:Q', title='Mutual Benefit Index'),
            y=alt.Y('Pair:N', sort='-x', title='Country Pair'),
            color=alt.Color('Risk_Category:N',
                            scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                            range=['green', 'orange', 'red']),
                            title='Risk Level'),
            tooltip=['Pair:N', 'Mutual_Benefit_Index:Q', 
                     'Trade_Volume_Billion:Q', 'Export_Pct_A:Q', 'Export_Pct_B:Q']
        ).properties(
            width=600,
            height=400,
            title="Top Country Pairs by Mutual Trade Benefit"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Trade Volume vs Mutual Benefit")
        
        scatter = alt.Chart(mbi_df).mark_circle(size=100).encode(
            x=alt.X('Trade_Volume_Billion:Q', title='Trade Volume (Billion USD)'),
            y=alt.Y('Mutual_Benefit_Index:Q', title='Mutual Benefit Index'),
            color=alt.Color('Mutual_Benefit_Index:Q',
                            scale=alt.Scale(scheme='viridis'),
                            title='MBI Score'),
            size=alt.Size('Trade_Volume_Billion:Q',
                          scale=alt.Scale(range=[50, 400]),
                          title='Trade Volume'),
            tooltip=['Pair:N', 'Mutual_Benefit_Index:Q', 
                     'Trade_Volume_Billion:Q', 'Export_Pct_A:Q', 'Export_Pct_B:Q']
        ).properties(
            width=600,
            height=400,
            title="Trade Relationship Benefit Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if shock_results is not None and not shock_results.empty:
        st.subheader("Trade Collapse Simulation")
        
        melted = shock_results.melt(
            id_vars=['Pair', 'Country_A', 'Country_B'], 
            value_vars=['GDP_Impact_A', 'GDP_Impact_B'],
            var_name='Country_Role', 
            value_name='GDP_Impact'
        )
        melted['Country'] = melted.apply(
            lambda x: x['Country_A'] if x['Country_Role'] == 'GDP_Impact_A' else x['Country_B'], 
            axis=1
        )
        
        shock_chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('GDP_Impact:Q', title='GDP Impact (%)'),
            y=alt.Y('Pair:N', title='Country Pair'),
            color=alt.Color('Country:N',
                            scale=alt.Scale(scheme='category10'),
                            title='Country'),
            tooltip=['Pair:N', 'Country:N', 'GDP_Impact:Q']
        ).properties(
            width=600,
            height=300,
            title=f"Estimated GDP Impact from Trade Collapse"
        )
        
        st.altair_chart(shock_chart, use_container_width=True)

def main():
    st.markdown("## Trade Relationship Mutual Benefit Analysis")
    st.caption("Identifying country pairs with highest mutual trade benefit and simulating collapse impacts")
 
    with st.spinner("Loading trade data..."):
        df = load_trade_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_trade_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_year_only = st.sidebar.checkbox("Use latest year data only", value=True)
    shock_magnitude = st.sidebar.slider(
        "Trade Collapse Magnitude (%)",
        min_value=10, max_value=100, value=100, step=10
    ) / 100
    
    st.markdown("## Mutual Benefit Index (MBI) Analysis")
    
    with st.spinner("Computing Mutual Benefit Index..."):
        try:
            mbi_results = compute_mutual_benefit_index(df, column_mapping, latest_year_only)
        except Exception as e:
            st.error(f"Error computing Mutual Benefit Index: {e}")
            st.stop()
    
    if mbi_results.empty:
        st.error("Could not compute Mutual Benefit Index. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pairs Analyzed", len(mbi_results))
    
    with col2:
        high_risk = len(mbi_results[mbi_results['Risk_Category'] == 'High'])
        st.metric("High Risk Pairs", high_risk, delta=f"{high_risk/len(mbi_results)*100:.1f}%")
    
    with col3:
        avg_mbi = mbi_results['Mutual_Benefit_Index'].mean()
        st.metric("Average MBI Score", f"{avg_mbi:.1f}")
    
    with col4:
        max_trade_volume = mbi_results['Trade_Volume_Billion'].max()
        st.metric("Max Trade Volume", f"${max_trade_volume:.0f}B")
    
    st.subheader("Top 3 Most Beneficial Trade Pairs")
    
    top_3 = mbi_results.head(3)
    
    for idx, (_, pair_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {pair_data['Pair']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MBI Score", f"{pair_data['Mutual_Benefit_Index']:.1f}")
                st.metric("Risk Level", pair_data['Risk_Category'])
            
            with c2:
                st.metric("Trade Volume", f"${pair_data['Trade_Volume_Billion']:.0f}B")
            
            with c3:
                st.metric("Export Pct A", f"{pair_data['Export_Pct_A']:.1f}%")
                st.metric("Export Pct B", f"{pair_data['Export_Pct_B']:.1f}%")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Pair', 'Mutual_Benefit_Index', 'Risk_Category', 
                    'Trade_Volume_Billion', 'Export_Pct_A', 'Export_Pct_B']
    
    formatted_results = mbi_results[display_cols].copy()
    formatted_results['Trade_Volume_Billion'] = formatted_results['Trade_Volume_Billion'].fillna(0.0).apply(lambda x: f"${x:.0f}B")
    formatted_results['Export_Pct_A'] = formatted_results['Export_Pct_A'].fillna(0.0).apply(lambda x: f"{x:.1f}%")
    formatted_results['Export_Pct_B'] = formatted_results['Export_Pct_B'].fillna(0.0).apply(lambda x: f"{x:.1f}%")
    formatted_results['Mutual_Benefit_Index'] = formatted_results['Mutual_Benefit_Index'].fillna(0.0).apply(lambda x: f"{x:.1f}")
    formatted_results['Risk_Category'] = formatted_results['Risk_Category'].fillna('Low')
    
    st.dataframe(
        formatted_results,
        column_config={
            "Pair": "Country Pair",
            "Mutual_Benefit_Index": "MBI Score",
            "Risk_Category": "Risk Level",
            "Trade_Volume_Billion": "Trade Volume",
            "Export_Pct_A": "Export % GDP A",
            "Export_Pct_B": "Export % GDP B"
        },
        use_container_width=True
    )
    
    st.header("Trade Route Collapse Simulation")
    st.markdown("Simulate the GDP impact if the trade route between pairs collapses")
    
    available_pairs = mbi_results['Pair'].tolist()
    default_selection = top_3['Pair'].tolist()
    
    selected_pairs = st.multiselect(
        "Select pairs to simulate trade collapse:",
        available_pairs,
        default=default_selection,
        help="Country pairs to include in the trade collapse simulation"
    )
    
    if selected_pairs:
        with st.spinner("Running trade collapse simulation..."):
            shock_results = simulate_trade_collapse(
                df, mbi_results, column_mapping, selected_pairs, shock_magnitude
            )
        
        if not shock_results.empty:
            st.subheader(f"Impact of {shock_magnitude*100:.0f}% Trade Route Collapse")
            
            avg_impact_a = shock_results['GDP_Impact_A'].mean()
            avg_impact_b = shock_results['GDP_Impact_B'].mean()
            worst_impact = min(shock_results['GDP_Impact_A'].min(), shock_results['GDP_Impact_B'].min())
            worst_pair = shock_results.loc[
                shock_results[['GDP_Impact_A', 'GDP_Impact_B']].min(axis=1).idxmin(), 'Pair'
            ]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg GDP Impact A", f"{avg_impact_a:.2f}%")
            with col2:
                st.metric("Avg GDP Impact B", f"{avg_impact_b:.2f}%")
            with col3:
                st.metric("Worst Impact", f"{worst_impact:.2f}%", delta=f"Pair: {worst_pair}")
            
            for _, result in shock_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Pair']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "GDP Impact A",
                            f"{result['GDP_Impact_A']:.2f}%",
                            f"{result['Country_A']}, Multiplier {result['Multiplier_A']:.1f}"
                        )
                    with c2:
                        st.metric(
                            "GDP Impact B",
                            f"{result['GDP_Impact_B']:.2f}%",
                            f"{result['Country_B']}, Multiplier {result['Multiplier_B']:.1f}"
                        )
                    with c3:
                        st.metric("Recovery A", f"{result['Recovery_Years_A']:.0f} years")
                        st.metric("Recovery B", f"{result['Recovery_Years_B']:.0f} years")
                    with c4:
                        st.metric("MBI Score", f"{result['Mutual_Benefit_Index']:.1f}")
    
    create_trade_visualizations(mbi_results, shock_results if 'shock_results' in locals() else None)
    
    if 'shock_results' in locals() and not shock_results.empty:
        st.subheader("Download Results")
        csv_data = shock_results.to_csv(index=False)
        st.download_button(
            label="Download Trade Collapse Analysis (CSV)",
            data=csv_data,
            file_name=f"trade_collapse_impact_{shock_magnitude*100:.0f}percent.csv",
            mime="text/csv"
        )
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Mutual Benefit Index (MBI) Calculation
        
        **Components:**
        - Geometric mean of estimated export percentages of GDP between country pairs
        - Exports from A to B and B to A estimated from total exports and derived trade partner shares
        - Scaled to reflect mutual dependency (0-100)
        
        **Risk Categories:**
        - **High Risk**: MBI > 5 (High mutual dependency)
        - **Medium Risk**: MBI 2-5 (Moderate mutual dependency)
        - **Low Risk**: MBI < 2 (Low mutual dependency)
        
        ### Dynamic Trade Partner Detection
        
        **Data-Driven Approach:**
        - Analyzes countries with high trade dependency indices and GDP values
        - Incorporates known major trade relationships based on economic patterns
        - Estimates bilateral trade shares using:
          * Relative economic size (GDP)
          * Trade intensity (exports + imports as % of GDP)
          * Geographic and economic proximity patterns
        
        **Estimated Trade Relationships Include:**
        - Major North American partnerships (US-Canada, US-Mexico)
        - Key Asian trade corridors (China-US, China-Japan, Korea-China)
        - European economic partnerships (Germany-France, Germany-UK)
        - Resource trade relationships (Australia-China, Brazil-China)
        - Service and technology trade (India-US)
        
        ### Trade Collapse Simulation
        
        **Assumptions:**
        - **Direct Impact**: GDP loss = Estimated export to partner (% of GDP) × Shock magnitude
        - **Multiplier Effects**: 1.0-2.0 based on import intensity (>50%: 2.0, >30%: 1.5)
        - **Recovery Time**: 1-5 years based on impact severity (1 year per 5% GDP loss)
        - **Bilateral Trade Estimation**: Uses total exports (% of GDP) from the dataset and dynamically estimated trade partner shares based on economic analysis
        
        **Data Sources:**
        - Primary data from integrated dataset with trade, GDP, and economic indicators
        - Trade relationship patterns derived from economic size and trade intensity analysis
        - Bilateral trade shares estimated using heuristic models based on typical trade patterns
        
        **Limitations:**
        - Bilateral trade values are estimated using economic heuristics, not actual bilateral trade data
        - Trade relationship estimates are based on typical patterns and may not reflect exact current relationships
        - Assumes static trade relationships without dynamic adjustments
        - Simplified multiplier effects may not capture full economic spillovers
        - Limited to countries present in the dataset
        """)

if __name__ == "__main__":
    main()