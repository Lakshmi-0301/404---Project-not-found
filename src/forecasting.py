import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import altair as alt

DEFAULT_DATA_PATH = Path("cleaned_datasets/merged_cleaned_data.csv")

CANDIDATE_COLS: Dict[str, List[str]] = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "gdp_growth": [
        "GDP growth (annual %)",
        "gdp_growth_pct",
    ],
    "poverty_rate": [
        "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)",
        "poverty_headcount_pct",
    ],
    "trade_gdp": [
        "Trade (% of GDP)",
        "trade_gdp_pct",
    ],
    "exports_gdp": [
        "Exports of goods and services (% of GDP)",
        "exports_gdp_pct",
    ],
    "imports_gdp": [
        "Imports of goods and services (% of GDP)",
        "imports_gdp_pct",
    ],
    "inflation": [
        "Inflation, consumer prices (annual %)",
        "inflation_pct",
    ],
    "cab_gdp": [
        "Current account balance (% of GDP)",
        "current_account_balance_gdp_pct",
    ],
    "ext_debt_gni": [
        "External debt stocks (% of GNI)",
        "external_debt_gni_pct",
    ],
    "fdi_gdp": [
        "Foreign direct investment, net inflows (% of GDP)",
        "fdi_net_inflows_gdp_pct",
    ],
    "disasters": [
        "Disaster_Count",
        "disasters",
    ],
}

@dataclass
class Targets:
    gdp_growth: Optional[float]
    poverty_rate: Optional[float]
    trade_resilience: Optional[float]

@st.cache_data(show_spinner=False)
def load_data(uploaded: Optional[Path]) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif DEFAULT_DATA_PATH.exists():
        df = pd.read_csv(DEFAULT_DATA_PATH)
    else:
        st.error("No data provided. Upload a CSV or place one at cleaned_datasets/merged_cleaned_data.csv")
        return pd.DataFrame()

    if any(c in df.columns for c in CANDIDATE_COLS["year"]):
        ycol = [c for c in CANDIDATE_COLS["year"] if c in df.columns][0]
        df[ycol] = pd.to_numeric(df[ycol], errors='coerce')
    if any(c in df.columns for c in CANDIDATE_COLS["country"]):
        ccol = CANDIDATE_COLS["country"][0]
        df[ccol] = df[ccol].astype(str)

    df = df.drop_duplicates().dropna(subset=[CANDIDATE_COLS["country"][0], ycol])
    return df

@st.cache_data(show_spinner=False)
def resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for key, candidates in CANDIDATE_COLS.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        mapping[key] = found
    return mapping


def fit_linear_trend(df: pd.DataFrame, country: str, year_col: str, target_col: str) -> Optional[LinearRegression]:
    sub = df[(df[CANDIDATE_COLS["country"][0]] == country) & (~df[target_col].isna())]
    if sub.empty:
        return None
    X = sub[[year_col]].values
    y = sub[target_col].values
    if len(sub) < 3:  
        return None
    model = LinearRegression()
    model.fit(X, y)
    return model


def compute_trade_resilience_row(row, m):
    val = {
        'trade_gdp': row.get(m['trade_gdp']) if m['trade_gdp'] else np.nan,
        'cab_gdp': row.get(m['cab_gdp']) if m['cab_gdp'] else np.nan,
        'ext_debt_gni': row.get(m['ext_debt_gni']) if m['ext_debt_gni'] else np.nan,
        'inflation': row.get(m['inflation']) if m['inflation'] else np.nan,
        'fdi_gdp': row.get(m['fdi_gdp']) if m['fdi_gdp'] else np.nan,
        'disasters': row.get(m['disasters']) if m['disasters'] else np.nan,
    }
    return val

@st.cache_data(show_spinner=False)
def build_trade_resilience_series(df: pd.DataFrame, m: Dict[str, Optional[str]]) -> pd.Series:
    needed = [m[k] for k in ["trade_gdp","cab_gdp","ext_debt_gni","inflation","fdi_gdp","disasters"] if m[k] is not None]
    if not needed:
        return pd.Series(np.nan, index=df.index)

    X = df[needed].copy()
    for col in needed:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X.fillna(X.median())), columns=needed, index=df.index)

    weights = {
        m['trade_gdp']: 0.20 if m['trade_gdp'] in Z.columns else 0.0,
        m['cab_gdp']:   0.25 if m['cab_gdp'] in Z.columns else 0.0,
        m['ext_debt_gni']: -0.20 if m['ext_debt_gni'] in Z.columns else 0.0,
        m['inflation']: -0.15 if m['inflation'] in Z.columns else 0.0,
        m['fdi_gdp']:   0.15 if m['fdi_gdp'] in Z.columns else 0.0,
        m['disasters']: -0.05 if m['disasters'] in Z.columns else 0.0,
    }

    score = sum(Z[col] * w for col, w in weights.items() if col in Z.columns)

    country_col = CANDIDATE_COLS['country'][0]
    res = []
    for country, grp in score.groupby(df[country_col]):
        mn, mx = grp.min(), grp.max()
        if np.isclose(mx, mn, equal_nan=True):
            res.append(pd.Series(50.0, index=grp.index))  # neutral
        else:
            res.append((grp - mn) / (mx - mn) * 100)
    return pd.concat(res).sort_index()



def apply_scenarios(base: Targets, social_increase: float, diversification_strength: float, crisis_severity: float) -> Dict[str, Targets]:
    """Apply simple, transparent scenario rules on top of the baseline.
       social_increase: 0.0–1.0 (e.g., 0.1 = +10%)
       diversification_strength: 0.0–1.0
       crisis_severity: 0.0–1.0
    """
    
    b = Targets(base.gdp_growth, base.poverty_rate, base.trade_resilience)

    social_d_gdp_pp = 0.3  
    social_pov_elasticity = -0.5  
    social_resilience_pts = 2.0   

    div_d_gdp_pp = 0.2
    div_resilience_pts = 5.0
    div_pov_rel_change = -0.02   
    crisis_d_gdp_pp = -3.0
    crisis_resilience_pts = -10.0
    crisis_pov_rel_change = +0.10  

    def rel(x, factor):
        return None if x is None else x * (1.0 + factor)

    scenarios = {
        "Baseline": b,
        "Increased social spending": Targets(
            None if b.gdp_growth is None else b.gdp_growth + social_d_gdp_pp * (social_increase*10),
            rel(b.poverty_rate, social_pov_elasticity * social_increase*10),
            None if b.trade_resilience is None else b.trade_resilience + social_resilience_pts * (social_increase*10),
        ),
        "Trade diversification": Targets(
            None if b.gdp_growth is None else b.gdp_growth + div_d_gdp_pp * (diversification_strength*10),
            rel(b.poverty_rate, div_pov_rel_change * diversification_strength*10),
            None if b.trade_resilience is None else b.trade_resilience + div_resilience_pts * (diversification_strength*10),
        ),
        "Global crisis": Targets(
            None if b.gdp_growth is None else b.gdp_growth + crisis_d_gdp_pp * (crisis_severity*10),
            rel(b.poverty_rate, crisis_pov_rel_change * crisis_severity*10),
            None if b.trade_resilience is None else b.trade_resilience + crisis_resilience_pts * (crisis_severity*10),
        ),
    }
    return scenarios


st.set_page_config(page_title="Modeling & Forecasting (2030)", layout="wide")

def run():
    """Main function to be called from app.py for the forecasting functionality"""
    st.title("Modeling & Forecasting to 2030")
    st.caption("Predict GDP growth, poverty, and trade resilience under multiple policy and risk scenarios.")

    
    uploaded_file = "datasets/processed/integrated_tren_dataset.csv"
    data_path = None
    if uploaded_file is not None:
        data_path = uploaded_file
    elif DEFAULT_DATA_PATH.exists():
        data_path = DEFAULT_DATA_PATH

    df = load_data(data_path)
    if df.empty:
        st.stop()

    colmap = resolve_columns(df)

    st.sidebar.header(" Filters")
    country_col = colmap['country']
    year_col = colmap['year']

    countries = sorted(df[country_col].dropna().unique().tolist())
    country = st.sidebar.selectbox("Country", countries)

    if 'trade_resilience_score' not in df.columns:
        try:
            tr_series = build_trade_resilience_series(df, colmap)
            df['trade_resilience_score'] = tr_series
        except Exception as e:
            st.warning(f"Couldn't compute trade resilience score automatically: {e}")
            df['trade_resilience_score'] = np.nan

    gdp_col = colmap['gdp_growth']
    pov_col = colmap['poverty_rate']

    models = {}
    for key, tcol in {
        'gdp_growth': gdp_col,
        'poverty_rate': pov_col,
        'trade_resilience': 'trade_resilience_score',
    }.items():
        if tcol is None:
            models[key] = None
            continue
        df[tcol] = pd.to_numeric(df[tcol], errors='coerce')
        models[key] = fit_linear_trend(df, country, year_col, tcol)

    def predict_or_nan(model):
        if model is None:
            return np.nan
        return float(model.predict(np.array([[2030]]) )[0])

    baseline = Targets(
        gdp_growth=predict_or_nan(models['gdp_growth']),
        poverty_rate=predict_or_nan(models['poverty_rate']),
        trade_resilience=predict_or_nan(models['trade_resilience']),
    )

    st.subheader(f"Baseline projection for 2030 – {country}")
    c1, c2, c3 = st.columns(3)
    c1.metric("GDP growth (annual %)", f"{baseline.gdp_growth:.2f}" if np.isfinite(baseline.gdp_growth) else "–")
    c2.metric("Poverty rate (%)", f"{baseline.poverty_rate:.2f}" if np.isfinite(baseline.poverty_rate) else "–")
    c3.metric("Trade resilience (0–100)", f"{baseline.trade_resilience:.1f}" if np.isfinite(baseline.trade_resilience) else "–")

    st.sidebar.header("Scenarios")
    social_increase = st.sidebar.slider("Increase social spending (vs. status quo)", 0.0, 0.5, 0.1, 0.01, help="0.10 = +10%")
    diversification_strength = st.sidebar.slider("Trade diversification intensity", 0.0, 0.5, 0.2, 0.01)
    crisis_severity = st.sidebar.slider("Global crisis severity", 0.0, 0.5, 0.2, 0.01)

    scenarios = apply_scenarios(baseline, social_increase, diversification_strength, crisis_severity)

    st.subheader("Scenario outcomes for 2030")
    show_cols = st.multiselect("Select outcomes to display", ["GDP growth", "Poverty rate", "Trade resilience"],
                               default=["GDP growth", "Poverty rate", "Trade resilience"])

    rows = []
    for name, t in scenarios.items():
        rows.append({
            "Scenario": name,
            "GDP growth": t.gdp_growth,
            "Poverty rate": t.poverty_rate,
            "Trade resilience": t.trade_resilience,
        })
    plot_df = pd.DataFrame(rows)

    for idx, row in plot_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Scenario']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("GDP growth (pp)", f"{row['GDP growth']:.2f}" if np.isfinite(row['GDP growth']) else "–")
            m2.metric("Poverty rate (%)", f"{row['Poverty rate']:.2f}" if np.isfinite(row['Poverty rate']) else "–")
            m3.metric("Trade resilience", f"{row['Trade resilience']:.1f}" if np.isfinite(row['Trade resilience']) else "–")

    long_df = plot_df.melt(id_vars=['Scenario'], value_vars=[c for c in ["GDP growth","Poverty rate","Trade resilience"] if c in show_cols],
                           var_name='Indicator', value_name='Value')
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Value'])

    if not long_df.empty:
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                x=alt.X('Scenario:N', sort=None),
                y=alt.Y('Value:Q'),
                color='Indicator:N',
                column=alt.Column('Indicator:N', header=alt.Header(title=None))
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Historical context")
    hist_cols = {
        "GDP growth (annual %)": gdp_col,
        "Poverty rate (%)": pov_col,
        "Trade resilience (0–100)": 'trade_resilience_score',
    }
    hc = st.multiselect("Select historical series", [k for k,v in hist_cols.items() if v is not None],
                        default=[k for k,v in hist_cols.items() if v is not None][:2])

    if hc:
        hist_df = df[df[country_col] == country][[year_col] + [hist_cols[k] for k in hc if hist_cols[k] is not None]].copy()
        hist_df = hist_df.rename(columns={hist_cols[k]: k for k in hc if hist_cols[k] is not None})
        hist_long = hist_df.melt(id_vars=[year_col], var_name='Series', value_name='Value').dropna()
        line = (
            alt.Chart(hist_long)
            .mark_line(point=True)
            .encode(x=alt.X(f'{year_col}:Q'), y='Value:Q', color='Series:N')
            .properties(height=350)
        )
        st.altair_chart(line, use_container_width=True)

    st.divider()
    with st.expander("Assumptions & Notes", expanded=False):
        st.markdown(
            """
            - **Baseline** uses a simple per-country linear trend (Year → indicator) fit to historical data.
            - **Trade resilience** is a composite (z-scored) index built from available columns:
              - + Trade (% of GDP), + Current account balance (% of GDP), + FDI (% of GDP),
              - − External debt (% of GNI), − Inflation, − Disaster count.
              - Scaled 0–100 within each country for interpretability.
            - **Scenario rules** (per +10% social spending / unit diversification / unit crisis):
              - Social spending: +0.3 pp to GDP growth, −5% to poverty rate (relative), +2 pts to resilience.
              - Trade diversification: +0.2 pp to GDP growth, −2% to poverty rate (relative), +5 pts to resilience.
              - Global crisis: −3.0 pp to GDP growth, +10% to poverty rate (relative), −10 pts to resilience.
            - These are **transparent placeholders**; adjust in the code to reflect evidence-based elasticities.
            """
        )

def main():
    """Standalone main function for running forecasting.py directly"""
    run()

if __name__ == "__main__":
    main()