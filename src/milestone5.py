from __future__ import annotations
import os
from typing import Optional, Dict, Any, Tuple, List
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
import plotly.graph_objects as go

DEFAULT_DATA_PATH = "datasets/processed/integrated_tren_dataset.csv"
MIN_YEARS_FOR_TREND = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ").replace(".", " ")

def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:

    cols = df.columns.tolist()
    nmap = {c: _norm(c) for c in cols}

    def find_by_keywords(keywords: List[str]):
        for c, nc in nmap.items():
            if all(k in nc for k in keywords):
                return c
        return None

    country_col = find_by_keywords(["country"]) or find_by_keywords(["nation"]) or find_by_keywords(["economy"]) or find_by_keywords(["country name"])

    year_col = find_by_keywords(["year"]) or find_by_keywords(["yr"])

    youth_col = None
    for c, nc in nmap.items():
        if "youth" in nc and ("unemploy" in nc or "unemployment" in nc):
            youth_col = c
            break
    if youth_col is None:
        for c, nc in nmap.items():
            if ("15 24" in nc or "15-24" in nc or "15_24" in nc) and "unemploy" in nc:
                youth_col = c
                break

    gdp_growth_col = find_by_keywords(["gdp growth"]) or find_by_keywords(["gdp (annual %)"]) or find_by_keywords(["gdp growth (annual %)", "gdp growth (annual percent)"])
    if gdp_growth_col is None:
        for c, nc in nmap.items():
            if "gdp" in nc and "growth" in nc:
                gdp_growth_col = c
                break

    return {
        "country": country_col,
        "year": year_col,
        "youth_unemployment": youth_col,
        "gdp_growth": gdp_growth_col,
    }

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    path = path or DEFAULT_DATA_PATH
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    raise FileNotFoundError(f"Data file not found at {path}")

def estimate_gdp_to_youth_unemp(df: pd.DataFrame, cols: Dict[str, str]) -> Tuple[float, Any]:
    ycol = cols["youth_unemployment"]
    xcol = cols["gdp_growth"]
    sub = df[[ycol, xcol]].dropna().copy()
    if sub.shape[0] < 10:
        return -0.8, None  
    X = sub[[xcol]].values
    y = sub[ycol].values
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return slope, (model, intercept)


def project_baseline_2030(df: pd.DataFrame, cols: Dict[str, str], target_year: int = 2030) -> pd.DataFrame:
    country_col = cols["country"]
    year_col = cols["year"]
    ycol = cols["youth_unemployment"]

    rows = []
    for c, g in df.groupby(country_col):
        gg = g[[year_col, ycol]].dropna().copy()
        if gg.shape[0] >= MIN_YEARS_FOR_TREND:
            X = gg[[year_col]].values
            y = gg[ycol].values
            model = LinearRegression().fit(X, y)
            baseline = float(model.predict([[target_year]])[0])
            slope = float(model.coef_[0])
            last_year = int(gg[year_col].max())
            last_value = float(gg[gg[year_col] == last_year][ycol].iloc[0])
        elif gg.shape[0] >= 1:
            last_year = int(gg[year_col].max())
            last_value = float(gg[gg[year_col] == last_year][ycol].iloc[0])
            baseline = last_value
            slope = 0.0
        else:
            continue
        rows.append({
            "country": c,
            "baseline_2030": baseline,
            "last_year": last_year,
            "last_value": last_value,
            "trend_slope_per_year": slope
        })
    out = pd.DataFrame(rows)
    return out


def apply_global_slowdown_scenario(
    baseline_df: pd.DataFrame,
    global_slowdown_pp: float,
    df_all: pd.DataFrame,
    cols: Dict[str, str],
    slope_gdp_to_unemp: float
) -> pd.DataFrame:

    gdp_col = cols["gdp_growth"]
    country_col = cols["country"]
    year_col = cols["year"]

    recent_avg = df_all.sort_values([country_col, year_col]).groupby(country_col)[gdp_col].apply(lambda s: s.dropna().iloc[-3:].mean() if s.dropna().shape[0] >= 1 else np.nan)
    recent_avg = recent_avg.to_dict()

    rows = []
    for idx, r in baseline_df.iterrows():
        c = r["country"]
        baseline = r["baseline_2030"]
        baseline_gdp = recent_avg.get(c, np.nan)
        if np.isnan(baseline_gdp):
            delta_gdp = -global_slowdown_pp
        else:
            delta_gdp = -global_slowdown_pp  
        delta_unemp_pp = slope_gdp_to_unemp * delta_gdp
        scenario_2030 = baseline + delta_unemp_pp
        rows.append({
            "country": c,
            "baseline_2030": baseline,
            "scenario_2030": scenario_2030,
            "delta_unemp_pp": delta_unemp_pp,
            "baseline_gdp_recent": baseline_gdp
        })
    out = pd.DataFrame(rows)
    out["flag_above_25pct"] = out["scenario_2030"] > 25.0
    return out

def _render_streamlit_ui(df: pd.DataFrame, cols: Dict[str, str]):
    st.title("Youth Unemployment >25% by 2030 under Global Slowdown Scenario")
    global_slowdown_pp = st.sidebar.number_input("Global slowdown (pp reduction in annual GDP growth)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    target_year = st.sidebar.number_input("Target year for outcome", min_value=2025, max_value=2035, value=2030, step=1)

    if st.button("Run Prediction", type="primary"):
        try:
            with st.spinner("Estimating model and projecting..."):
                slope, model_info = estimate_gdp_to_youth_unemp(df, cols)
                baseline_df = project_baseline_2030(df, cols, target_year=target_year)
                scenario_df = apply_global_slowdown_scenario(baseline_df, global_slowdown_pp, df, cols, slope)
                scenario_df = scenario_df.sort_values("scenario_2030", ascending=False)

            st.success("Projection complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk_count = len(scenario_df[scenario_df["flag_above_25pct"]])
                st.metric("Countries with >25% youth unemployment", high_risk_count)
            with col2:
                avg_increase = scenario_df["delta_unemp_pp"].mean()
                st.metric("Avg. youth unemployment increase (pp)", f"{avg_increase:.2f}")
            with col3:
                st.metric("GDP-Unemployment sensitivity", f"{slope:.3f}")

            st.markdown("#### Countries predicted to have youth unemployment > 25% in 2030")
            high_risk = scenario_df[scenario_df["flag_above_25pct"]]
            if len(high_risk) > 0:
                st.dataframe(
                    high_risk[["country", "baseline_2030", "scenario_2030", "baseline_gdp_recent"]].round(2),
                    use_container_width=True
                )
            else:
                st.info("No countries predicted to exceed 25% youth unemployment under this scenario.")

            st.markdown("#### Top 20 by scenario youth unemployment")
            display_cols = ["country", "baseline_2030", "scenario_2030", "baseline_gdp_recent"]
            st.dataframe(scenario_df[display_cols].head(20).round(2), use_container_width=True)

            st.markdown("#### Visualization: Top 20 Countries by Projected Youth Unemployment")
            chart_data = scenario_df.head(20).copy()
            chart_data = chart_data.sort_values("scenario_2030", ascending=True)  
            
            colors = ['red' if x else 'steelblue' for x in chart_data['flag_above_25pct']]
            
            fig = go.Figure(go.Bar(
                x=chart_data['scenario_2030'],
                y=chart_data['country'],
                orientation='h',
                marker_color=colors,
                text=[f"{val:.1f}%" for val in chart_data['scenario_2030']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Youth Unemployment Rate Projections for 2030 (Global Slowdown Scenario)",
                xaxis_title="Youth Unemployment Rate (%)",
                yaxis_title="Country",
                height=600,
                showlegend=False
            )
            
            fig.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="25% threshold")
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check your data format and try again.")

def main(data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
    try:
        if df is None:
            df = load_data(data_path)

        cols = detect_columns(df)
        missing = [k for k, v in cols.items() if v is None]
        
        try:
            _render_streamlit_ui(df, cols)
        except Exception as e:
            if missing:
                print(f"Could not detect required columns: {missing}")
                print(f"Detected mapping: {cols}")
                return

            slope, _ = estimate_gdp_to_youth_unemp(df, cols)
            baseline_df = project_baseline_2030(df, cols, target_year=2030)
            scenario_df = apply_global_slowdown_scenario(baseline_df, 1.5, df, cols, slope)
            print("Top 20 countries by projected youth unemployment in 2030:")
            print(scenario_df.sort_values("scenario_2030", ascending=False).head(20).to_string(index=False))
            
    except FileNotFoundError as e:
        if 'streamlit' in globals():
            st.error(f"Data file not found: {e}")
            st.info("Please upload a CSV file using the sidebar uploader.")
        else:
            print(f"Error: {e}")
    except Exception as e:
        if 'streamlit' in globals():
            st.error(f"An error occurred: {e}")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()