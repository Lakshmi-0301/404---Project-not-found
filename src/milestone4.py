import os
import sys
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

def _safe_lower(s):
    try:
        return str(s).lower()
    except Exception:
        return s

def detect_columns(df: pd.DataFrame):
    cols = {c: _safe_lower(c) for c in df.columns}
    inv = {v: k for k, v in cols.items()}
    def pick(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in inv:
                return inv[cand]
        for k, v in cols.items():
            for cand in candidates:
                if cand in v:
                    return k
        return None

    country_col = pick(["country", "reporter", "importer", "economy", "country_name"])
    partner_col = pick(["partner", "exporter", "supplier", "source", "trade_partner"])
    product_col = pick(["product", "commodity", "item", "hs_desc", "category", "sector"])
    value_col   = pick(["value", "import_value", "imports", "val_usd", "trade_value", "usd", "amount"])
    year_col    = pick(["year", "time", "period"])
    hs_col      = pick(["hs", "hs_code", "hs6", "hs4", "hs_chapter", "hschapter", "chapter"])

    return dict(country=country_col, partner=partner_col, product=product_col,
                value=value_col, year=year_col, hs_code=hs_col)

def tag_agriculture(df: pd.DataFrame, product_col: Optional[str], hs_col: Optional[str]) -> pd.Series:
    mask = pd.Series([False] * len(df), index=df.index)

    if hs_col is not None and hs_col in df.columns:
        def chapter(v):
            try:
                s = str(v).strip()
                if len(s) >= 2 and s[:2].isdigit():
                    return int(s[:2])
                return int(float(s))
            except Exception:
                return None
        chapters = df[hs_col].map(chapter)
        mask_hs = chapters.apply(lambda x: (x is not None) and (1 <= x <= 24))
        mask = mask | mask_hs

    if product_col is not None and product_col in df.columns:
        keywords = [
            "agri", "agriculture", "farm", "food", "cereal", "grain", "wheat", "rice", "maize",
            "corn", "barley", "oat", "millet", "sorghum", "soy", "soybean", "oilseed",
            "palm", "sunflower", "rapeseed", "canola", "vegetable", "fruit", "banana",
            "apple", "citrus", "meat", "beef", "poultry", "chicken", "pork",
            "fish", "seafood", "dairy", "milk", "cheese", "butter", "egg",
            "sugar", "coffee", "cocoa", "tea", "spice", "pulse", "lentil", "pea",
            "bean", "chickpea", "cassava", "yam", "potato", "flour", "edible oil", "edible oils"
        ]
        prod_l = df[product_col].astype(str).str.lower()
        mask_kw = prod_l.apply(lambda s: any(k in s for k in keywords))
        mask = mask | mask_kw

    return mask

def compute_dependency(df: pd.DataFrame,
                       country_col: str,
                       partner_col: str,
                       value_col: str,
                       top_n_for_flag: int = 3,
                       threshold_share: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df.groupby([country_col, partner_col], dropna=False, as_index=False)[value_col].sum()
    tot = g.groupby(country_col, as_index=False)[value_col].sum().rename(columns={value_col: "total_agri_imports"})
    g = g.merge(tot, on=country_col, how="left")
    g["share"] = g[value_col] / g["total_agri_imports"].replace(0, np.nan)
    g["rank"] = g.groupby(country_col)["share"].rank(method="first", ascending=False).astype(int)
    g = g.sort_values([country_col, "share"], ascending=[True, False]).reset_index(drop=True)

    top_shares = g[g["rank"] <= 3].pivot(index=country_col, columns="rank", values="share").fillna(0.0)
    top_shares.columns = [f"top{int(c)}_share" for c in top_shares.columns]
    
    hhi = g.copy()
    hhi["share2"] = (hhi["share"].fillna(0.0) * 100) ** 2
    hhi = hhi.groupby(country_col, as_index=False)["share2"].sum().rename(columns={"share2": "hhi"})
    
    partner_count = g.groupby(country_col, as_index=False).size().rename(columns={"size": "partner_count"})
    
    cover = g[g["rank"] <= top_n_for_flag].groupby(country_col, as_index=False)["share"].sum().rename(columns={"share": "top_n_cover_share"})

    summary = tot.merge(top_shares, on=country_col, how="left").merge(hhi, on=country_col, how="left").merge(cover, on=country_col, how="left").merge(partner_count, on=country_col, how="left")
    for c in ["top1_share", "top2_share", "top3_share", "top_n_cover_share"]:
        if c not in summary.columns:
            summary[c] = 0.0
    
    summary["diversification_index"] = 1 - (summary["hhi"] / 10000)
    summary["concentrated_flag"] = summary["top_n_cover_share"].fillna(0.0) >= threshold_share
    summary["critical_dependency"] = (summary["top1_share"] >= 0.5) | (summary["top2_share"] >= 0.3)

    partner_shares = g[[country_col, partner_col, value_col, "share", "rank"]].copy()

    return partner_shares, summary

def simulate_export_ban(partner_shares: pd.DataFrame,
                        country_col: str,
                        partner_col: str,
                        value_col: str,
                        k_banned: int = 2) -> pd.DataFrame:
    ps = partner_shares.copy()
    ps = ps.sort_values([country_col, "share"], ascending=[True, False])
    
    topk = ps.groupby(country_col).head(k_banned)
    lost = topk.groupby(country_col, as_index=False)["share"].sum().rename(columns={"share": "lost_import_share"})
    
    hhi = ps.copy()
    hhi["share2"] = (hhi["share"].fillna(0.0) * 100) ** 2
    hhi = hhi.groupby(country_col, as_index=False)["share2"].sum().rename(columns={"share2": "hhi"})
    
    partner_count = ps.groupby(country_col).size().reset_index(name='total_partners')
    
    sim = lost.merge(hhi, on=country_col, how="left").merge(partner_count, on=country_col, how="left")
    sim["remaining_import_share"] = 1.0 - sim["lost_import_share"].fillna(0.0)
    
    sim["market_concentration"] = np.minimum(1.0, sim["hhi"].fillna(1000.0) / 5000.0)
    sim["supply_disruption_factor"] = 1 + sim["lost_import_share"].fillna(0.0) * 2
    sim["partner_scarcity"] = np.maximum(0.5, 1 - (sim["total_partners"] - k_banned) / 10)
    
    sim["vulnerability_score"] = (
        sim["lost_import_share"].fillna(0.0) * 40 +
        sim["market_concentration"] * 30 +
        sim["partner_scarcity"] * 30
    )
    sim["vulnerability_score"] = sim["vulnerability_score"].clip(0, 100)
    
    base_recovery = np.random.normal(18, 3, len(sim))
    dependency_factor = 1 + sim["lost_import_share"].fillna(0.0) * 1.5
    concentration_penalty = 1 + sim["market_concentration"] * 0.8
    diversification_bonus = np.maximum(0.5, 1 - (sim["total_partners"] - 5) / 20)
    
    sim["recovery_months"] = (
        base_recovery * 
        dependency_factor * 
        concentration_penalty * 
        diversification_bonus
    )
    sim["recovery_months"] = sim["recovery_months"].clip(6, 48).round(1)
    
    sim["food_security_risk"] = pd.cut(
        sim["vulnerability_score"], 
        bins=[0, 25, 50, 75, 100], 
        labels=["Low", "Moderate", "High", "Critical"]
    )
    
    return sim.sort_values("vulnerability_score", ascending=False)

def run_milestone4_analysis(df):
    st.title("Agricultural Import Dependency & Export Ban Risk")
    st.write("Identify countries most dependent on 2-3 partners for agricultural imports and model food security risk if those partners impose export bans.")
    
    if df.empty:
        st.error("No data available. Please load a dataset first.")
        return
    
    detected = detect_columns(df)
    
    if detected.get("country") is None or detected.get("partner") is None or detected.get("value") is None:
        st.error("Could not automatically detect required columns (country, partner, value). Please check your dataset structure.")
        return
    
    st.subheader("Analysis Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        threshold_share = st.slider("Top-N coverage threshold", 0.5, 0.95, 0.7, 0.05, 
                                   help="Countries where top N partners cover at least this share are flagged as concentrated.")
        top_n_for_flag = st.selectbox("N partners for concentration test", [2, 3], index=1)
    
    with col2:
        k_banned = st.selectbox("Ban scenario (top-k partners banned)", [2, 3], index=0)
    
    country_col = detected["country"]
    partner_col = detected["partner"]  
    value_col = detected["value"]
    product_col = detected.get("product")
    hs_col = detected.get("hs_code")
    
    st.write(f"**Detected columns:** Country: {country_col}, Partner: {partner_col}, Value: {value_col}")
    if product_col:
        st.write(f"Product: {product_col}")
    if hs_col:
        st.write(f"HS Code: {hs_col}")
    
    keep = [c for c in [country_col, partner_col, value_col, product_col, hs_col] if c is not None]
    slim = df[keep].copy()
    
    mask = tag_agriculture(slim, product_col, hs_col)
    agri = slim[mask].copy()
    
    if agri.empty:
        st.error("No agricultural records detected. Try adjusting the dataset or ensure it contains agricultural trade data.")
        return
    
    agri[value_col] = pd.to_numeric(agri[value_col], errors="coerce").fillna(0.0)
    agri = agri[agri[value_col] > 0]
    
    st.write(f"Found {len(agri):,} agricultural trade records from {agri[country_col].nunique()} countries")
    
    partner_shares, summary = compute_dependency(
        agri, country_col, partner_col, value_col,
        top_n_for_flag=top_n_for_flag, threshold_share=threshold_share
    )
    
    st.subheader("Import Dependency Analysis")
    st.write(f"Countries where top **{top_n_for_flag}** partners cover ≥ **{int(threshold_share*100)}%** of agricultural imports:")
    
    conc = summary.sort_values("top_n_cover_share", ascending=False)
    
    display_summary = conc.copy()
    display_summary["diversification_index"] = display_summary["diversification_index"].round(3)
    display_summary["hhi"] = display_summary["hhi"].round(0)
    for col in ["top1_share", "top2_share", "top3_share", "top_n_cover_share"]:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].round(3)
    
    st.dataframe(display_summary)
    
    st.write(f"**Key Insights:**")
    critical_countries = len(conc[conc["critical_dependency"]])
    concentrated_countries = len(conc[conc["concentrated_flag"]])
    avg_diversification = conc["diversification_index"].mean()
    st.write(f"- {critical_countries} countries have critical dependency (>50% from top partner or >30% from top 2)")
    st.write(f"- {concentrated_countries} countries are flagged as concentrated based on top-{top_n_for_flag} threshold")
    st.write(f"- Average market diversification index: {avg_diversification:.2f} (1.0 = perfectly diversified)")
    
    st.subheader("Most Dependent Countries (Top 15)")
    top_dep = conc[conc["concentrated_flag"]].head(15)
    if len(top_dep) == 0 and len(conc) > 0:
        top_dep = conc.head(15)
    
    if not top_dep.empty:
        chart_data = top_dep.set_index(country_col)[["top1_share", "top2_share", "top3_share"]]
        st.bar_chart(chart_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Dependent Country", 
                     top_dep.iloc[0][country_col], 
                     f"{top_dep.iloc[0]['top_n_cover_share']:.1%} from top {top_n_for_flag}")
        with col2:
            st.metric("Highest HHI", 
                     conc.loc[conc["hhi"].idxmax(), country_col],
                     f"{conc['hhi'].max():.0f}")
    
    st.subheader(f"Food Security Risk Assessment: Top-{k_banned} Partner Export Ban")
    sim = simulate_export_ban(partner_shares, country_col, partner_col, value_col, k_banned=k_banned)
    
    if not sim.empty:
        st.write("**Vulnerability Analysis Results:**")
        
        display_sim = sim.copy()
        for col in ["lost_import_share", "remaining_import_share", "market_concentration", "vulnerability_score"]:
            if col in display_sim.columns:
                display_sim[col] = display_sim[col].round(3)
        
        st.dataframe(display_sim)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Vulnerability Scores:**")
            vuln_chart = sim.set_index(country_col)["vulnerability_score"].head(12)
            st.bar_chart(vuln_chart)
            
        with col2:
            st.write("**Recovery Time (Months):**")
            recovery_chart = sim.set_index(country_col)["recovery_months"].head(12)
            st.bar_chart(recovery_chart)
            
        with col3:
            st.write("**Risk Distribution:**")
            risk_dist = sim["food_security_risk"].value_counts()
            st.bar_chart(risk_dist)
        
        high_risk = len(sim[sim["vulnerability_score"] > 70])
        avg_recovery = sim["recovery_months"].mean()
        worst_case = sim.iloc[0]
        
        st.write("**Risk Assessment Summary:**")
        st.write(f"- **{high_risk}** countries face high vulnerability (score >70)")
        st.write(f"- **Average recovery time:** {avg_recovery:.1f} months across all countries")
        st.write(f"- **Highest risk country:** {worst_case[country_col]} (score: {worst_case['vulnerability_score']:.1f}, recovery: {worst_case['recovery_months']:.1f} months)")
        
        critical_risk_countries = sim[sim["food_security_risk"] == "Critical"]
        if len(critical_risk_countries) > 0:
            st.error(f"🚨 **{len(critical_risk_countries)} countries** face CRITICAL food security risk from export bans")
            st.write("Critical risk countries:", ", ".join(critical_risk_countries[country_col].head(10).tolist()))
    else:
        st.error("No vulnerability analysis results generated")
    
    st.subheader("Country Details")
    countries = list(partner_shares[country_col].dropna().unique())
    if len(countries) > 0:
        sel_cty = st.selectbox("Select a country for detailed analysis", options=sorted(countries))
        ps = partner_shares[partner_shares[country_col] == sel_cty].sort_values("share", ascending=False).copy()
        ps["share_pct"] = (ps["share"] * 100).round(2)
        st.write(f"**Partner shares for {sel_cty}:**")
        display_cols = [partner_col, value_col, "share_pct", "rank"]
        st.dataframe(ps[display_cols])
        
        if not ps.empty:
            partner_chart = ps.set_index(partner_col)["share"].head(10)
            st.bar_chart(partner_chart)
    
    with st.expander("Download Results"):
        st.download_button("Download Partner Shares CSV", 
                          data=partner_shares.to_csv(index=False), 
                          file_name="agricultural_partner_shares.csv")
        st.download_button("Download Country Dependency Summary CSV", 
                          data=summary.to_csv(index=False), 
                          file_name="agricultural_dependency_summary.csv")
        st.download_button("Download Vulnerability Assessment CSV", 
                          data=sim.to_csv(index=False), 
                          file_name="food_security_vulnerability_assessment.csv")

def main(df):
    run_milestone4_analysis(df)
