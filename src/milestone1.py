import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import altair as alt

DEFAULT_DATA_PATH = Path("datasets/processed/integrated_tren_dataset_with_indexes.csv")

TRADE_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "exports_gdp": [
        "Exports of goods and services (% of GDP)",
        "exports_gdp_pct",
        "Exports_GDP_pct"
    ],
    "imports_gdp": [
        "Imports of goods and services (% of GDP)", 
        "imports_gdp_pct",
        "Imports_GDP_pct"
    ],
    "trade_gdp": [
        "Trade (% of GDP)",
        "trade_gdp_pct",
        "Trade_GDP_pct"
    ],
    "gdp_growth": [
        "GDP growth (annual %)",
        "gdp_growth_pct",
        "GDP_growth_pct"
    ],
    "current_account": [
        "Current account balance (% of GDP)",
        "current_account_balance_gdp_pct",
        "CAB_GDP_pct"
    ],
    "fdi_gdp": [
        "Foreign direct investment, net inflows (% of GDP)",
        "fdi_net_inflows_gdp_pct",
        "FDI_GDP_pct"
    ]
}

MOCK_TRADE_PARTNERS = {
    "Germany": {"China": 0.18, "United States": 0.12, "Netherlands": 0.11, "France": 0.08, "Italy": 0.06},
    "Mexico": {"United States": 0.78, "China": 0.06, "Canada": 0.03, "Germany": 0.02, "Japan": 0.02},
    "South Korea": {"China": 0.25, "United States": 0.13, "Japan": 0.09, "Vietnam": 0.08, "Hong Kong": 0.07},
    "Singapore": {"Malaysia": 0.12, "China": 0.11, "United States": 0.10, "Hong Kong": 0.08, "Indonesia": 0.07},
    "Netherlands": {"Germany": 0.24, "Belgium": 0.11, "United Kingdom": 0.09, "France": 0.08, "United States": 0.06},
    "Canada": {"United States": 0.73, "China": 0.05, "United Kingdom": 0.03, "Japan": 0.02, "Mexico": 0.02},
    "Ireland": {"United States": 0.28, "United Kingdom": 0.11, "Germany": 0.10, "Belgium": 0.08, "France": 0.06},
    "Belgium": {"Germany": 0.17, "France": 0.14, "Netherlands": 0.12, "United Kingdom": 0.08, "United States": 0.06},
    "Czech Republic": {"Germany": 0.32, "Slovakia": 0.08, "Poland": 0.06, "China": 0.05, "Austria": 0.04},
    "Hungary": {"Germany": 0.28, "Austria": 0.05, "Romania": 0.05, "Slovakia": 0.05, "Italy": 0.04},
    "Slovakia": {"Germany": 0.22, "Czech Republic": 0.12, "Poland": 0.07, "Hungary": 0.06, "Austria": 0.06},
    "Luxembourg": {"Germany": 0.25, "Belgium": 0.17, "France": 0.12, "United States": 0.08, "Netherlands": 0.07},
    "Estonia": {"Finland": 0.16, "Sweden": 0.10, "Latvia": 0.09, "Germany": 0.08, "Russia": 0.07},
    "Lithuania": {"Russia": 0.14, "Latvia": 0.10, "Poland": 0.09, "Germany": 0.08, "Estonia": 0.06},
    "Malta": {"Italy": 0.12, "Germany": 0.08, "France": 0.07, "United Kingdom": 0.06, "Spain": 0.05},
}

@st.cache_data(show_spinner=False)
def load_trade_data(uploaded_file=None) -> pd.DataFrame:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            st.warning("No data file found. Using sample data for demonstration.")
            return generate_sample_data()
        
        df = df.drop_duplicates()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str)
        if 'Year' in df.columns or 'year' in df.columns:
            year_col = 'Year' if 'Year' in df.columns else 'year'
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.warning("Using sample data for demonstration.")
        return generate_sample_data()

def generate_sample_data() -> pd.DataFrame:
    countries = list(MOCK_TRADE_PARTNERS.keys()) + ["United States", "China", "Japan", "United Kingdom", "France", "Italy", "Spain"]
    years = [2020, 2021, 2022, 2023]
    
    data = []
    for country in countries:
        for year in years:
            np.random.seed(hash(country + str(year)) % 1000)
            
            base_trade = 60 if country in ["Singapore", "Netherlands", "Ireland"] else 40
            exports_gdp = base_trade + np.random.normal(0, 10)
            imports_gdp = base_trade + np.random.normal(0, 10)
            
            data.append({
                'Country': country,
                'Year': year,
                'Exports of goods and services (% of GDP)': max(10, exports_gdp),
                'Imports of goods and services (% of GDP)': max(10, imports_gdp),
                'Trade (% of GDP)': max(20, exports_gdp + imports_gdp),
                'GDP growth (annual %)': np.random.normal(2.5, 2.0),
                'Current account balance (% of GDP)': np.random.normal(0, 3),
                'Foreign direct investment, net inflows (% of GDP)': np.random.normal(3, 2)
            })
    
    return pd.DataFrame(data)

def resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in TRADE_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_trade_dependency_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
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
        
        trade_intensity = 0
        if column_mapping['trade_gdp'] and not pd.isna(country_data.get(column_mapping['trade_gdp'])):
            trade_intensity = float(country_data[column_mapping['trade_gdp']])
        elif column_mapping['exports_gdp'] and column_mapping['imports_gdp']:
            exp_gdp = country_data.get(column_mapping['exports_gdp'], 0)
            imp_gdp = country_data.get(column_mapping['imports_gdp'], 0)
            if not pd.isna(exp_gdp) and not pd.isna(imp_gdp):
                trade_intensity = float(exp_gdp) + float(imp_gdp)
        
        top_partner_share = 0
        top_partner = "Unknown"
        partner_concentration = 0
        
        if country in MOCK_TRADE_PARTNERS:
            partners = MOCK_TRADE_PARTNERS[country]
            top_partner = max(partners.keys(), key=partners.get)
            top_partner_share = partners[top_partner]
            partner_concentration = sum(share**2 for share in partners.values())
        else:
            if trade_intensity > 100:  
                top_partner_share = np.random.uniform(0.15, 0.35)
                partner_concentration = 0.15
            elif trade_intensity > 60:
                top_partner_share = np.random.uniform(0.10, 0.25)
                partner_concentration = 0.12
            else:
                top_partner_share = np.random.uniform(0.05, 0.15)
                partner_concentration = 0.08
        
        ca_balance = 0
        if column_mapping['current_account']:
            ca_val = country_data.get(column_mapping['current_account'])
            if not pd.isna(ca_val):
                ca_balance = float(ca_val)
        
        fdi_inflows = 0
        if column_mapping['fdi_gdp']:
            fdi_val = country_data.get(column_mapping['fdi_gdp'])
            if not pd.isna(fdi_val):
                fdi_inflows = float(fdi_val)
        
        trade_component = min(trade_intensity / 2, 50)  
        concentration_component = partner_concentration * 100  
        vulnerability_component = max(0, -ca_balance / 2) + max(0, (20 - fdi_inflows))  
        
        tdi = (0.4 * trade_component + 
               0.4 * concentration_component + 
               0.2 * vulnerability_component)
        
        results.append({
            'Country': country,
            'Trade_Intensity_GDP': trade_intensity,
            'Top_Partner': top_partner,
            'Top_Partner_Share': top_partner_share,
            'Partner_Concentration_HHI': partner_concentration,
            'Current_Account_GDP': ca_balance,
            'FDI_Inflows_GDP': fdi_inflows,
            'Trade_Dependency_Index': tdi,
            'Risk_Category': 'High' if tdi > 30 else 'Medium' if tdi > 20 else 'Low'
        })
    
    return pd.DataFrame(results).sort_values('Trade_Dependency_Index', ascending=False)

def simulate_trade_shock_impact(tdi_df: pd.DataFrame, trade_df: pd.DataFrame, 
                              column_mapping: Dict[str, Optional[str]], 
                              shock_countries: List[str], shock_magnitude: float = 0.4) -> pd.DataFrame:

    results = []
    
    for country in shock_countries:
        if country not in tdi_df['Country'].values:
            continue
            
        country_info = tdi_df[tdi_df['Country'] == country].iloc[0]
        
        trade_intensity = country_info['Trade_Intensity_GDP']
        top_partner_share = country_info['Top_Partner_Share']
        exports_share = trade_intensity * 0.5  
        
        export_loss_pct = top_partner_share * shock_magnitude
        direct_gdp_impact = -(exports_share / 100) * export_loss_pct
        
        if trade_intensity > 100:
            multiplier = 1.8
        elif trade_intensity > 60:
            multiplier = 1.5
        else:
            multiplier = 1.2
            
        total_gdp_impact = direct_gdp_impact * multiplier
        
        baseline_growth = 2.5  
        if column_mapping['gdp_growth']:
            country_trade_data = trade_df[trade_df[column_mapping['country']] == country]
            if not country_trade_data.empty:
                latest_growth = country_trade_data[column_mapping['gdp_growth']].dropna()
                if not latest_growth.empty:
                    baseline_growth = float(latest_growth.iloc[-1])
        
        shocked_growth = baseline_growth + total_gdp_impact
        
        recovery_years = max(1, min(5, int(abs(total_gdp_impact))))
        
        results.append({
            'Country': country,
            'Top_Partner': country_info['Top_Partner'],
            'Baseline_GDP_Growth_2026': baseline_growth,
            'Direct_GDP_Impact': direct_gdp_impact,
            'Total_GDP_Impact': total_gdp_impact,
            'Shocked_GDP_Growth_2026': shocked_growth,
            'Export_Loss_Percent': export_loss_pct * 100,
            'Estimated_Recovery_Years': recovery_years,
            'Trade_Dependency_Index': country_info['Trade_Dependency_Index']
        })
    
    return pd.DataFrame(results)

def create_visualizations(tdi_df: pd.DataFrame, shock_results: pd.DataFrame = None):    
    st.subheader("Trade Dependency Index Rankings")
    
    if not tdi_df.empty:
        top_10 = tdi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Trade_Dependency_Index:Q', title='Trade Dependency Index'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Risk_Category:N', 
                          scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                        range=['green', 'orange', 'red']),
                          title='Risk Level'),
            tooltip=['Country:N', 'Trade_Dependency_Index:Q', 'Top_Partner:N', 
                    'Top_Partner_Share:Q', 'Trade_Intensity_GDP:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Trade Dependency Risk"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Trade Intensity vs Partner Concentration")
        
        scatter = alt.Chart(tdi_df).mark_circle(size=100).encode(
            x=alt.X('Trade_Intensity_GDP:Q', title='Trade Intensity (% of GDP)'),
            y=alt.Y('Partner_Concentration_HHI:Q', title='Partner Concentration (HHI)'),
            color=alt.Color('Trade_Dependency_Index:Q', 
                          scale=alt.Scale(scheme='viridis'),
                          title='TDI Score'),
            size=alt.Size('Top_Partner_Share:Q', 
                         scale=alt.Scale(range=[50, 400]),
                         title='Top Partner Share'),
            tooltip=['Country:N', 'Trade_Dependency_Index:Q', 'Top_Partner:N', 
                    'Top_Partner_Share:Q', 'Trade_Intensity_GDP:Q']
        ).properties(
            width=600,
            height=400,
            title="Trade Dependency Risk Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if shock_results is not None and not shock_results.empty:
        st.subheader("Trade Shock Impact Simulation")
        
        shock_chart = alt.Chart(shock_results).mark_bar().encode(
            x=alt.X('Country:N', title='Country'),
            y=alt.Y('Total_GDP_Impact:Q', title='GDP Impact (percentage points)'),
            color=alt.Color('Total_GDP_Impact:Q', 
                          scale=alt.Scale(scheme='reds'),
                          title='Impact Severity'),
            tooltip=['Country:N', 'Top_Partner:N', 'Total_GDP_Impact:Q', 
                    'Shocked_GDP_Growth_2026:Q', 'Export_Loss_Percent:Q']
        ).properties(
            width=600,
            height=300,
            title="Estimated GDP Impact from 40% Trade Partner Import Reduction"
        )
        
        st.altair_chart(shock_chart, use_container_width=True)

def main():
    """Main function for the trade dependency analysis"""
    
    st.title("Trade Dependency Risk Analysis")
    st.caption("Identifying countries most vulnerable to single-partner trade collapse")
 
    uploaded_file = "datasets/processed/integrated_tren_dataset.csv"

    with st.spinner("Loading trade data..."):
        df = load_trade_data(uploaded_file)
  
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_data_only = st.sidebar.checkbox("Use latest year data only", value=True)
    shock_magnitude = st.sidebar.slider(
        "Trade shock magnitude (%)", 
        min_value=10, max_value=80, value=40, step=5
    ) / 100
    
    st.header("Trade Dependency Index (TDI) Analysis")
    
    with st.spinner("Computing Trade Dependency Index..."):
        tdi_results = compute_trade_dependency_index(df, column_mapping, latest_data_only)
    
    if tdi_results.empty:
        st.error("Could not compute Trade Dependency Index. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Countries Analyzed", 
            len(tdi_results)
        )
    
    with col2:
        high_risk = len(tdi_results[tdi_results['Risk_Category'] == 'High'])
        st.metric(
            "High Risk Countries", 
            high_risk,
            delta=f"{high_risk/len(tdi_results)*100:.1f}%"
        )
    
    with col3:
        avg_tdi = tdi_results['Trade_Dependency_Index'].mean()
        st.metric(
            "Average TDI Score", 
            f"{avg_tdi:.1f}"
        )
    
    with col4:
        max_trade_intensity = tdi_results['Trade_Intensity_GDP'].max()
        st.metric(
            "Max Trade Intensity", 
            f"{max_trade_intensity:.1f}% of GDP"
        )
    
    st.subheader("Top 3 Most Vulnerable Countries")
    
    top_3 = tdi_results.head(3)
    
    for idx, (_, country_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {country_data['Country']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("TDI Score", f"{country_data['Trade_Dependency_Index']:.1f}")
                st.metric("Risk Level", country_data['Risk_Category'])
            
            with c2:
                st.metric("Trade Intensity", f"{country_data['Trade_Intensity_GDP']:.1f}% GDP")
                st.metric("Top Partner", country_data['Top_Partner'])
            
            with c3:
                st.metric("Partner Share", f"{country_data['Top_Partner_Share']*100:.1f}%")
                st.metric("Concentration (HHI)", f"{country_data['Partner_Concentration_HHI']:.3f}")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Country', 'Trade_Dependency_Index', 'Risk_Category', 
                   'Trade_Intensity_GDP', 'Top_Partner', 'Top_Partner_Share']
    
    formatted_results = tdi_results[display_cols].copy()
    formatted_results['Top_Partner_Share'] = formatted_results['Top_Partner_Share'].apply(lambda x: f"{x*100:.1f}%")
    formatted_results['Trade_Intensity_GDP'] = formatted_results['Trade_Intensity_GDP'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Trade_Dependency_Index'] = formatted_results['Trade_Dependency_Index'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Trade_Dependency_Index": "TDI Score",
            "Risk_Category": "Risk Level",
            "Trade_Intensity_GDP": "Trade/GDP",
            "Top_Partner": "Main Partner",
            "Top_Partner_Share": "Partner Share"
        },
        use_container_width=True
    )
    
    st.header("Trade Shock Simulation")
    st.markdown("Simulate the impact of a major trade partner reducing imports")
    
    available_countries = tdi_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries to simulate trade shock:",
        available_countries,
        default=default_selection,
        help="Countries to include in the trade shock simulation"
    )
    
    if selected_countries:
        with st.spinner("Running trade shock simulation..."):
            shock_results = simulate_trade_shock_impact(
                tdi_results, df, column_mapping, selected_countries, shock_magnitude
            )
        
        if not shock_results.empty:
            st.subheader(f"Impact of {shock_magnitude*100:.0f}% Import Reduction by Top Trade Partner")
            
            avg_impact = shock_results['Total_GDP_Impact'].mean()
            worst_impact = shock_results['Total_GDP_Impact'].min()
            worst_country = shock_results.loc[shock_results['Total_GDP_Impact'].idxmin(), 'Country']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average GDP Impact", f"{avg_impact:.2f} pp")
            with col2:
                st.metric("Worst Impact", f"{worst_impact:.2f} pp")
            with col3:
                st.metric("Most Affected", worst_country)
            
            for _, result in shock_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Country']} - Partner: {result['Top_Partner']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "GDP Growth 2026",
                            f"{result['Shocked_GDP_Growth_2026']:.2f}%",
                            f"{result['Total_GDP_Impact']:.2f} pp"
                        )
                    with c2:
                        st.metric("Export Loss", f"{result['Export_Loss_Percent']:.1f}%")
                    with c3:
                        st.metric("Recovery Time", f"{result['Estimated_Recovery_Years']:.0f} years")
                    with c4:
                        st.metric("TDI Score", f"{result['Trade_Dependency_Index']:.1f}")
    
    create_visualizations(tdi_results, shock_results if 'shock_results' in locals() else None)
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Trade Dependency Index (TDI) Calculation
        
        **Components (weighted):**
        - **Trade Intensity (40%)**: Total trade as % of GDP
        - **Partner Concentration (40%)**: Herfindahl-Hirschman Index of trade partner concentration  
        - **Economic Vulnerability (20%)**: Current account balance and FDI dependency
        
        **Risk Categories:**
        - **High Risk**: TDI > 30 (Heavy trade dependence with high concentration)
        - **Medium Risk**: TDI 20-30 (Moderate trade dependence)
        - **Low Risk**: TDI < 20 (Diversified trade relationships)
        
        ### Trade Shock Simulation Assumptions
        
        1. **Direct Impact**: Export loss = Top partner share × Shock magnitude × Export intensity
        2. **Multiplier Effects**: Trade-dependent economies have higher multipliers (1.2x to 1.8x)
        3. **Recovery**: Estimated based on impact severity (1-5 years)
        4. **Baseline Growth**: Uses historical data or assumes 2.5% baseline
        
        ### Data Sources & Limitations
        
        - Trade partner data uses representative estimates for major economies
        - Smaller economies use modeled relationships based on trade intensity
        - Real implementation would require detailed bilateral trade databases
        - Multiplier effects are simplified estimates based on economic literature
        """)

if __name__ == "__main__":
    main()