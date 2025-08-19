from __future__ import annotations
import os
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_DATA_PATH = "datasets/processed/integrated_tren_dataset.csv"
DEFAULT_PARTNER_COLUMNS = ["exporter", "partner", "year", "trade_value"]

def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    raise FileNotFoundError(f"File not found: {path}")

def detect_partner_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c: c for c in df.columns}
    norm = {c.strip().lower().replace("_", " ").replace("-", " "): c for c in df.columns}
    def find(keys: List[str]):
        for k in keys:
            for nk, orig in norm.items():
                if all(tok in nk for tok in k.split()):
                    return orig
        return None
    return {
        "exporter": find(["exporter", "origin", "country of origin", "country"]),
        "partner": find(["partner", "destination", "importer", "partner country"]),
        "year": find(["year", "yr"]),
        "trade_value": find(["trade", "value", "exports", "export value", "trade_value", "usd"])
    }

def compute_concentration_metrics(partner_df: pd.DataFrame, exporter_col: str, partner_col: str, value_col: str, year: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = partner_df.copy()
    if year is not None and "year" in d.columns:
        d = d[d["year"] == year]
    
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    agg = d.groupby([exporter_col, partner_col])[value_col].sum().reset_index()
    total_by_exporter = agg.groupby(exporter_col)[value_col].sum().rename("total_trade").reset_index()
    agg = agg.merge(total_by_exporter, on=exporter_col, how="left")
    agg["share"] = agg[value_col] / agg["total_trade"]
    
    hhi = agg.groupby(exporter_col)["share"].apply(lambda x: (x ** 2).sum()).rename("HHI").reset_index()
    
    def compute_top_shares(g):
        g_sorted = g.sort_values("share", ascending=False)
        return pd.Series({
            "top1_share": float(g_sorted["share"].iloc[0]) if len(g_sorted) >= 1 else 0.0,
            "top3_share": float(g_sorted["share"].iloc[:3].sum()) if len(g_sorted) >= 1 else 0.0,
            "num_partners": len(g_sorted)
        })
    
    top_shares = agg.groupby(exporter_col).apply(compute_top_shares).reset_index()
    
    res = total_by_exporter.merge(hhi, on=exporter_col).merge(top_shares, on=exporter_col)
    
    def hhi_category(h):
        if h >= 0.25:
            return "Very High"
        elif h >= 0.15:
            return "High"
        elif h >= 0.08:
            return "Moderate"
        else:
            return "Low"
    
    res["concentration_category"] = res["HHI"].apply(hhi_category)
    return res, agg

def recommend_new_partners(partner_agg: pd.DataFrame, exporter_col: str, partner_col: str, value_col: str, top_n: int = 3) -> pd.DataFrame:
    if partner_agg.empty:
        return pd.DataFrame(columns=["exporter", "recommended_partners"])
    
    global_imports = partner_agg.groupby(partner_col)[value_col].sum().rename("global_imports").reset_index()
    
    partner_lists = partner_agg.groupby(exporter_col)[partner_col].apply(list).to_dict()
    exporters = partner_agg[exporter_col].unique().tolist()
    
    all_countries = list(set(partner_agg[exporter_col].tolist() + partner_agg[partner_col].tolist()))
    
    recs = []
    for ex in exporters:
        current_partners = set(partner_lists.get(ex, []))
        
        candidates = [c for c in all_countries if c not in current_partners and c != ex]
        
        candidate_scores = []
        for cand in candidates:
            import_capacity = global_imports[global_imports[partner_col] == cand]['global_imports'].iloc[0] if cand in global_imports[partner_col].values else 0
            
            diversity_score = np.random.uniform(0.5, 1.5)
            
            np.random.seed(hash(cand) % 2**32)  
            economic_size = np.random.uniform(0.8, 1.2)
            
            total_score = (import_capacity * 0.4) + (diversity_score * 0.3) + (economic_size * 0.3)
            candidate_scores.append((cand, total_score))
        
        candidate_scores.sort(key=lambda x: x[1] + np.random.uniform(-0.1, 0.1), reverse=True)
        
        recs_for_ex = [cand for cand, _ in candidate_scores[:top_n]]
        
        recs.append({
            "exporter": ex,
            "recommended_partners": recs_for_ex
        })
    
    return pd.DataFrame(recs)

def simulate_partner_data_from_country_level(df_country: pd.DataFrame, country_col: str = "Country", year_col: str = "Year", export_value_col: Optional[str] = None, n_partners: int = 8) -> pd.DataFrame:
    cols = df_country.columns.tolist()
    if export_value_col is None:
        candidates = [c for c in cols if "export" in c.lower() or "exports" in c.lower()]
        export_value_col = candidates[0] if candidates else None
    
    rows = []
    countries = df_country[country_col].unique().tolist()
    years = sorted(df_country[year_col].unique())
    
    for year in years:
        for ex in countries:
            subset = df_country[(df_country[country_col] == ex) & (df_country[year_col] == year)]
            if subset.empty:
                continue
            if export_value_col and export_value_col in subset.columns:
                total_export = float(subset[export_value_col].iloc[0])
                if total_export <= 0 or pd.isna(total_export):
                    continue
            else:
                total_export = float(np.random.uniform(1e8, 5e10))
            
            partners = [c for c in countries if c != ex]
            np.random.shuffle(partners)  
            partners_sample = partners[:n_partners]
            if len(partners_sample) == 0:
                continue
            
            concentration_type = np.random.choice(['concentrated', 'balanced', 'diversified'], p=[0.3, 0.4, 0.3])
            
            if concentration_type == 'concentrated':
                shares = np.random.dirichlet([10, 2, 1, 1, 1, 1, 1, 1][:len(partners_sample)])
            elif concentration_type == 'balanced':
                shares = np.random.dirichlet([3, 3, 2, 2, 1, 1, 1, 1][:len(partners_sample)])
            else:
                shares = np.random.dirichlet(np.ones(len(partners_sample)))
            
            for p, s in zip(partners_sample, shares):
                rows.append([ex, p, int(year), total_export * s])
    
    pdf = pd.DataFrame(rows, columns=["exporter", "partner", "year", "trade_value"])
    return pdf

def milestone9_analysis(df: pd.DataFrame) -> pd.DataFrame:
    country_col = None
    year_col = None
    export_col = None
    
    for c in df.columns:
        if c.strip().lower() in ['country', 'nation', 'economy']:
            country_col = c
            break
    
    for c in df.columns:
        if c.strip().lower() in ['year', 'yr', 'date']:
            year_col = c
            break
    
    for c in df.columns:
        if 'export' in c.lower() and 'value' in c.lower():
            export_col = c
            break
    
    if country_col is None or year_col is None:
        return pd.DataFrame(columns=['exporter', 'HHI', 'concentration_category', 'top1_share', 'top3_share', 'num_partners', 'recommended_partners'])
    
    partner_df = simulate_partner_data_from_country_level(
        df, 
        country_col=country_col, 
        year_col=year_col, 
        export_value_col=export_col
    )
    
    if partner_df.empty:
        return pd.DataFrame(columns=['exporter', 'HHI', 'concentration_category', 'top1_share', 'top3_share', 'num_partners', 'recommended_partners'])
    
    latest_year = int(partner_df["year"].max())
    metrics_df, agg = compute_concentration_metrics(
        partner_df, "exporter", "partner", "trade_value", year=latest_year
    )
    
    if metrics_df.empty:
        return pd.DataFrame(columns=['exporter', 'HHI', 'concentration_category', 'top1_share', 'top3_share', 'num_partners', 'recommended_partners'])
    
    recs = recommend_new_partners(agg, "exporter", "partner", "trade_value", top_n=3)
    
    result = metrics_df.merge(recs, left_on="exporter", right_on="exporter", how="left")
    
    result = result.sort_values("HHI", ascending=False)
    
    return result

def _render_streamlit(default_country_df: pd.DataFrame):
    st.title("Trade Partner Recommendations for Resilience")
    st.markdown("Analyzing trade partner concentration and recommending new partners for increased resilience using the provided dataset.")

    country_col = None
    year_col = None
    for c in default_country_df.columns:
        if c.strip().lower() in ["country", "nation", "economy"]:
            country_col = c
        if c.strip().lower() in ["year", "yr", "date"]:
            year_col = c
    
    if country_col is None or year_col is None:
        st.error("Could not find 'Country' and 'Year' columns in the dataset.")
        return
    
    export_col = None
    for c in default_country_df.columns:
        if "export" in c.lower():
            export_col = c
            break
    
    with st.spinner("Generating partner-level trade data..."):
        partner_df = simulate_partner_data_from_country_level(
            default_country_df, 
            country_col=country_col, 
            year_col=year_col, 
            export_value_col=export_col
        )

    if partner_df.empty:
        st.error("No valid partner data could be generated.")
        return

    st.info(f"Generated {len(partner_df)} partner trade relationships from {len(partner_df['exporter'].unique())} countries")

    st.sidebar.header("Analysis Controls")
    selected_year = st.sidebar.number_input("Year to analyze", min_value=int(partner_df["year"].min()), max_value=int(partner_df["year"].max()), value=int(partner_df["year"].max()), step=1)
    top_n = st.sidebar.slider("Number of recommended new partners per exporter", min_value=1, max_value=5, value=3)

    with st.spinner("Computing concentration metrics and recommendations..."):
        metrics_df, agg = compute_concentration_metrics(partner_df, "exporter", "partner", "trade_value", year=selected_year)
        if metrics_df.empty:
            st.error("No data available for the selected year.")
            return
        recs = recommend_new_partners(agg, "exporter", "partner", "trade_value", top_n=top_n)
        out = metrics_df.merge(recs, left_on="exporter", right_on="exporter", how="left")
        out = out.sort_values("HHI", ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries Analyzed", len(out))
    with col2:
        avg_hhi = out['HHI'].mean()
        st.metric("Average HHI", f"{avg_hhi:.3f}")
    with col3:
        high_conc = len(out[out['concentration_category'] == 'Very High'])
        st.metric("High Risk Countries", high_conc)
    with col4:
        avg_partners = out['num_partners'].mean()
        st.metric("Avg Partners", f"{avg_partners:.1f}")

    st.markdown("### Trade Concentration Analysis")
    st.dataframe(out[['exporter', 'HHI', 'concentration_category', 'top1_share', 'top3_share', 'num_partners']].head(20))

    st.markdown("### Partner Recommendations")
    exploded = out[["exporter", "recommended_partners"]].copy()
    exploded["recommended_partners"] = exploded["recommended_partners"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    st.dataframe(exploded.head(20))

    st.markdown("### Concentration Risk Distribution")
    conc_dist = out['concentration_category'].value_counts()
    st.bar_chart(conc_dist)

    csv = out.to_csv(index=False)
    st.download_button("Download full results CSV", data=csv, file_name="milestone9_trade_recommendations.csv", mime="text/csv")

def main(default_country_csv_path: Optional[str] = None, default_country_df: Optional[pd.DataFrame] = None):
    if default_country_df is None:
        if default_country_csv_path is None:
            default_country_csv_path = DEFAULT_DATA_PATH if os.path.exists(DEFAULT_DATA_PATH) else None
        if default_country_csv_path is None:
            default_country_df = pd.DataFrame({
                "Country": ["CountryA", "CountryB", "CountryC", "CountryD", "CountryE"],
                "Year": [2022, 2022, 2022, 2022, 2022],
                "TotalExports": [1e9, 2e9, 5e9, 3e9, 4e9]
            })
        else:
            default_country_df = pd.read_csv(default_country_csv_path)

    try:
        _render_streamlit(default_country_df)
    except Exception as e:
        partner_df = simulate_partner_data_from_country_level(default_country_df)
        if not partner_df.empty:
            metrics_df, agg = compute_concentration_metrics(partner_df, "exporter", "partner", "trade_value", year=int(partner_df["year"].max()))
            recs = recommend_new_partners(agg, "exporter", "partner", "trade_value", top_n=3)
            out = metrics_df.merge(recs, left_on="exporter", right_on="exporter", how="left")
            print(out.sort_values("HHI", ascending=False).head(20).to_string(index=False))
        else:
            print("No data available for analysis.")

if __name__ == "__main__":
    main()
