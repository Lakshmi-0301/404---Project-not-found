from typing import List, Tuple, Dict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_CSV = "datasets/processed/integrated_tren_dataset.csv"
PROJECT_YEAR = 2030
TOP_K = 5

INVERT_KEYWORDS = [
    "unemp", "poverty", "inflation", "mort", "death", "mortality",
    "vuln", "crime", "debt", "deficit", "loss", "violence", "risk",
    "unemploy", "corrupt", "instabil"
]

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DEFAULT_CSV)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found: {DEFAULT_CSV}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

def detect_country_year_cols(df: pd.DataFrame) -> Tuple[str, str]:
    country_candidates = ["country", "Country", "COUNTRY", "nation", "country_name", "Nation"]
    year_candidates = ["year", "Year", "YEAR", "yr", "time", "Time"]

    country_col = next((c for c in country_candidates if c in df.columns), None)
    if country_col is None:
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            country_col = obj_cols[0]
        else:
            raise ValueError("No suitable country column found in dataset.")

    year_col = next((y for y in year_candidates if y in df.columns), None)
    if year_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if df[c].dropna().between(1900, 2100).all() and len(df[c].dropna()) > 0:
                year_col = c
                break
        
        if year_col is None and len(numeric_cols) > 0:
            year_col = numeric_cols[0]

    if country_col is None or year_col is None:
        raise ValueError(f"Could not detect country ({country_col}) or year ({year_col}) columns.")

    return country_col, year_col

def select_indicator_columns(df: pd.DataFrame, country_col: str, year_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    indicators = [c for c in numeric_cols if c not in [country_col, year_col]]
    
    valid_indicators = []
    for col in indicators:
        non_null_ratio = df[col].notna().sum() / len(df)
        if non_null_ratio > 0.1:
            valid_indicators.append(col)
        else:
            st.info(f"Excluding '{col}' due to insufficient data ({non_null_ratio:.1%} coverage)")
    
    return valid_indicators

def needs_inversion(colname: str) -> bool:
    return any(kw in colname.lower() for kw in INVERT_KEYWORDS)

def compute_composite_scores(df: pd.DataFrame, indicators: List[str], 
                           country_col: str, year_col: str) -> pd.DataFrame:

    tmp = df[[country_col, year_col] + indicators].copy()
    
    for col in indicators:
        tmp[col] = tmp.groupby(country_col)[col].transform(
            lambda g: g.fillna(g.mean())
        )
    tmp[indicators] = tmp[indicators].fillna(tmp[indicators].mean())

    inverted_indicators = []
    for col in indicators:
        if needs_inversion(col):
            tmp[col] = tmp[col].max() - tmp[col]
            inverted_indicators.append(col)
    
    scaler = StandardScaler()
    try:
        scaled = scaler.fit_transform(tmp[indicators])
        scaled_df = pd.DataFrame(scaled, columns=indicators, index=tmp.index)
    except Exception as e:
        st.error(f"Error in standardization: {str(e)}")
        return tmp

    tmp["composite_score"] = scaled_df.mean(axis=1)
    
    for col in indicators:
        tmp[f"scaled_{col}"] = scaled_df[col]

    return tmp

def project_to_year(group: pd.DataFrame, year_col: str, target_year: int) -> float:
    years = group[year_col].astype(float).values.reshape(-1, 1)
    scores = group["composite_score"].astype(float).values
    
    if len(years) < 2:
        return float(scores[-1])
    
    try:
        model = LinearRegression().fit(years, scores)
        projection = float(model.predict([[target_year]])[0])
        return projection
    except Exception:
        return float(scores[-1])

def analyze_country_drivers(group: pd.DataFrame, indicators: List[str], 
                          year_col: str) -> List[Tuple[str, float, str]]:
    years = group[year_col].astype(float).values.reshape(-1, 1)
    drivers = []
    
    for ind in indicators:
        scaled_col = f"scaled_{ind}"
        if scaled_col not in group.columns:
            continue
            
        y = group[scaled_col].values
        
        if len(years) < 2:
            slope = float(group[scaled_col].mean())
        else:
            try:
                slope = float(LinearRegression().fit(years, y).coef_[0])
            except Exception:
                slope = float(group[scaled_col].mean())
        
        if slope > 0.05:
            interpretation = "Strong positive trend"
        elif slope > 0.01:
            interpretation = "Positive trend"
        elif slope < -0.05:
            interpretation = "Declining trend"
        elif slope < -0.01:
            interpretation = "Weak decline"
        else:
            interpretation = "Stable"
            
        drivers.append((ind, slope, interpretation))
    
    return sorted(drivers, key=lambda x: abs(x[1]), reverse=True)

def build_projection_and_rank(df: pd.DataFrame, indicators: List[str], 
                            country_col: str, year_col: str, 
                            target_year: int, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    results = []
    
    for country, group in df.groupby(country_col):
        group = group.sort_values(year_col)
        
        projected_score = project_to_year(group, year_col, target_year)
        
        drivers = analyze_country_drivers(group, indicators, year_col)
        
        current_score = float(group["composite_score"].iloc[-1])
        current_year = int(group[year_col].iloc[-1])
        
        results.append({
            "country": country,
            "projected_score": projected_score,
            "current_score": current_score,
            "current_year": current_year,
            "score_change": projected_score - current_score,
            "drivers": drivers
        })
    
    results_df = pd.DataFrame(results).sort_values("projected_score", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    
    return results_df, results_df.head(top_k)

def format_drivers(drivers: List[Tuple[str, float, str]], top_n: int = 3) -> str:
    if not drivers:
        return "No drivers available"
    
    formatted = []
    for i, (ind, slope, interp) in enumerate(drivers[:top_n]):
        clean_name = ind.replace('_', ' ').title()
        formatted.append(f"{clean_name} ({interp}, {slope:+.3f})")
    
    return "; ".join(formatted)

def create_projection_chart(top_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    countries = top_df["country"].tolist()
    current_scores = top_df["current_score"].tolist()
    projected_scores = top_df["projected_score"].tolist()
    
    fig.add_trace(go.Bar(
        name=f'Current ({top_df["current_year"].iloc[0]})',
        x=countries,
        y=current_scores,
        marker_color='lightblue',
        text=[f'{score:.3f}' for score in current_scores],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Projected ({PROJECT_YEAR})',
        x=countries,
        y=projected_scores,
        marker_color='darkblue',
        text=[f'{score:.3f}' for score in projected_scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Resilience Scores: Current vs {PROJECT_YEAR} Projection',
        xaxis_title='Countries',
        yaxis_title='Composite Resilience Score',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig

def display_country_analysis(country_data: Dict, indicators: List[str]) -> None:
    country = country_data["country"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Score", 
            f"{country_data['current_score']:.3f}",
            f"Year {country_data['current_year']}"
        )
    
    with col2:
        st.metric(
            f"{PROJECT_YEAR} Projection", 
            f"{country_data['projected_score']:.3f}",
            f"{country_data['score_change']:+.3f}"
        )
    
    with col3:
        st.metric(
            "Global Rank", 
            f"#{country_data['rank']}"
        )
    
    st.subheader(f"Key Success Factors for {country}")
    drivers = country_data["drivers"][:5]  
    
    for i, (indicator, slope, interpretation) in enumerate(drivers, 1):
        clean_name = indicator.replace('_', ' ').title()
        
        if "positive" in interpretation.lower():
            color = "🟢"
        elif "declining" in interpretation.lower():
            color = "🔴"
        else:
            color = "🟡"
        
        st.write(f"{color} **{clean_name}**: {interpretation} (trend: {slope:+.3f})")

def run_streamlit_app():  
    st.title("Resilience Projections to 2030")
    st.markdown("""
    This analysis identifies the top 5 countries most likely to rank in the highest resilience tier by 2030,
    based on current trends and key driving factors.
    """)

    with st.spinner("Loading dataset..."):
        df = load_data()

    try:
        country_col, year_col = detect_country_year_cols(df)
    except ValueError as e:
        st.error(f"{str(e)}")
        return

    indicators = select_indicator_columns(df, country_col, year_col)
    
    if not indicators:
        st.error("No suitable indicators found in the dataset.")
        return

    st.info(f"Found {len(indicators)} resilience indicators")

    with st.expander("Configure Indicators", expanded=False):
        selected_indicators = st.multiselect(
            "Select indicators for resilience analysis:",
            indicators,
            default=indicators[:min(10, len(indicators))],  
            help="Choose the indicators that best represent resilience factors"
        )
        
        if not selected_indicators:
            st.error("Please select at least one indicator.")
            return

    with st.spinner("Computing resilience projections..."):
        try:
            composite_df = compute_composite_scores(df, selected_indicators, country_col, year_col)
            all_results, top_countries = build_projection_and_rank(
                composite_df, selected_indicators, country_col, year_col, PROJECT_YEAR, TOP_K
            )
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            return

    st.header(f"Top {TOP_K} Most Resilient Countries by {PROJECT_YEAR}")
    
    chart = create_projection_chart(top_countries)
    st.plotly_chart(chart, use_container_width=True)
    
    st.subheader("Summary Rankings")
    display_df = top_countries.copy()
    display_df["key_drivers"] = display_df["drivers"].apply(lambda d: format_drivers(d, 2))
    
    summary_cols = ["rank", "country", "projected_score", "score_change", "key_drivers"]
    st.dataframe(
        display_df[summary_cols].rename(columns={
            "rank": "Rank",
            "country": "Country",
            "projected_score": "2030 Score",
            "score_change": "Score Change",
            "key_drivers": "Top Success Factors"
        }),
        use_container_width=True
    )

    st.header("Detailed Country Analysis")
    
    selected_country = st.selectbox(
        "Choose a country for detailed analysis:",
        top_countries["country"].tolist(),
        index=0
    )
    
    country_data = top_countries[top_countries["country"] == selected_country].iloc[0].to_dict()
    display_country_analysis(country_data, selected_indicators)
    
    st.header("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Patterns")
        all_drivers = []
        for drivers_list in top_countries["drivers"]:
            all_drivers.extend([d[0] for d in drivers_list[:3]])
        
        from collections import Counter
        common_drivers = Counter(all_drivers).most_common(5)
        
        st.write("Most common success factors among top countries:")
        for driver, count in common_drivers:
            clean_name = driver.replace('_', ' ').title()
            st.write(f"• **{clean_name}**: {count}/{TOP_K} countries")
    
    with col2:
        st.subheader("Growth Trajectories")
        improving = sum(1 for _, row in top_countries.iterrows() if row["score_change"] > 0)
        st.metric("Countries Improving", f"{improving}/{TOP_K}")
        
        avg_improvement = top_countries["score_change"].mean()
        st.metric("Average Score Change", f"{avg_improvement:+.3f}")

    with st.expander("Methodology", expanded=False):
        st.markdown("""
        ### How the Analysis Works:
        
        1. **Data Integration**: Combines multiple resilience indicators from the integrated dataset
        2. **Standardization**: All indicators are standardized to enable fair comparison
        3. **Inversion Handling**: Negative indicators (unemployment, poverty, etc.) are inverted
        4. **Composite Scoring**: Creates overall resilience scores from weighted indicators  
        5. **Trend Analysis**: Uses linear regression to project scores to 2030
        6. **Driver Analysis**: Identifies which factors contribute most to each country's success
        
        ### Key Features:
        - Automatic detection of country and year columns
        - Intelligent handling of missing data
        - Trend-based projections using historical data
        - Factor analysis to understand success drivers
        """)

def main():
    """Entry point for the application."""
    try:
        run_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your dataset format and try again.")

if __name__ == "__main__":
    main()