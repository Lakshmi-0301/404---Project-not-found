import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

REQUIRED_COLUMNS = [
    'Country', 'Year', 'GDP growth (annual %)', 'GDP per capita (current US$)',
    'Resilience_Score', 'Trade_Dependency_Index', 'Shock_Impact_Score',
    'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)',
    'Unemployment with advanced education (% of total labor force with advanced education)',
    'agr_yield_avg', 'agr_production_avg', 'disaster_deaths', 'disaster_affected',
    'GDP (current US$)', 'Unemployment, total (% of total labor force) (modeled ILO estimate)_y'
]

class PolicyScenarioAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_columns()
        self._prep()

    def _ensure_columns(self):
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            for col in missing:
                name = col.lower()
                if "unemployment" in name:
                    self.df[col] = 10.0
                elif "gdp per capita" in name:
                    self.df[col] = 3000.0
                elif "gdp" in name:
                    self.df[col] = 1e9
                elif "score" in name:
                    self.df[col] = 0.5
                else:
                    self.df[col] = 0.0

    def _prep(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
        self.df['Baseline_Resilience'] = self.df['Resilience_Score']
        self.df['Baseline_GDP_Growth'] = self.df['GDP growth (annual %)']

    @staticmethod
    def _nz_div(a, b, default=0.0):
        return np.where(b == 0, default, a / b)


    def _aggregate_by_country(self, df, agg_method='mean'):
        """Aggregate data by country using specified method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_dict = {col: agg_method for col in numeric_cols}
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col != 'Country':
                agg_dict[col] = 'first'
        
        return df.groupby('Country').agg(agg_dict).reset_index()

    def scenario_1_trade_diversification(self, trade_reduction=0.25):
        d = self.df.copy()
        red_factor = np.minimum(0.4, trade_reduction * (1 + d['Trade_Dependency_Index']))
        d['Trade_Dependency_Index_New'] = d['Trade_Dependency_Index'] * (1 - red_factor)

        resilience_improv = red_factor * 0.8 * (1 + d['Trade_Dependency_Index'])
        gdp_impact = -red_factor * 0.3 * d['Trade_Dependency_Index']
        mult = 2.0 - d['Resilience_Score']

        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + resilience_improv * mult)
        d['GDP_Growth_New'] = d['GDP growth (annual %)'] + gdp_impact

        u_base = d['Unemployment, total (% of total labor force) (modeled ILO estimate)_y']
        d['Unemployment_New'] = np.maximum(0, u_base - resilience_improv * u_base * 0.1)
        return d

    def scenario_2_youth_employment(self, unemployment_reduction=0.5, aggregate=True):
        d = self.df.copy()
        yu = d['Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)']
        red = np.minimum(0.8, unemployment_reduction * (1 + yu / 20.0))
        d['Youth_Unemployment_New'] = yu * (1 - red)
        
        gdp_pc = d['GDP per capita (current US$)']
        gdp_pc_norm = self._nz_div(gdp_pc, gdp_pc.max(), 0)
        growth_boost = red * 0.6 * (2.0 - gdp_pc_norm)
        res_boost = red * 0.4 * (1.5 - d['Resilience_Score'])
        
        d['GDP_Growth_New'] = d['GDP growth (annual %)'] + growth_boost
        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + res_boost)
        
        total_u = d['Unemployment, total (% of total labor force) (modeled ILO estimate)_y']
        spill = red * 0.3 * self._nz_div(yu, total_u, 0)
        d['Total_Unemployment_New'] = np.maximum(0, total_u - spill)
        
        if aggregate:
            d = self._aggregate_by_country(d)
        
        return d

    def scenario_3_agricultural_productivity(self, yield_increase=0.25):
        d = self.df.copy()
        y0 = d['agr_yield_avg']; p0 = d['agr_production_avg']
        pot = yield_increase * (1 + 1/(1 + y0/1000.0))
        d['Agr_Yield_New'] = y0 * (1 + pot)
        d['Agr_Production_New'] = p0 * (1 + pot)

        agr_importance = self._nz_div(p0, p0.max(), 0)
        gdp_boost = pot * 0.8 * agr_importance
        res_boost = pot * 0.6 * agr_importance

        if 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)' in d.columns:
            pov = d['Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)']
            d['Poverty_Reduction'] = pov * (1 - pot * 0.4 * agr_importance)
        else:
            d['Poverty_Reduction'] = 20.0 * (1 - pot * 0.4 * agr_importance)

        d['Rural_Employment_New'] = 50 + pot * 0.5 * agr_importance * 10
        d['GDP_Growth_New'] = d['GDP growth (annual %)'] + gdp_boost
        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + res_boost)
        return d

    def scenario_4_disaster_preparedness(self, preparedness_investment=0.02):
        d = self.df.copy()
        vuln = d['disaster_deaths'] + d['disaster_affected'] / 1000.0
        vuln = vuln.replace(0, vuln.median())
        vfac = 1 + self._nz_div(vuln, vuln.max(), 0)
        invest_rate = preparedness_investment * vfac
        d['Disaster_Investment'] = d['GDP (current US$)'] * invest_rate

        shock = d['Shock_Impact_Score']
        red_pot = invest_rate * 10 * self._nz_div(shock, shock.max(), 0)
        red = np.minimum(0.6, red_pot)

        d['Disaster_Deaths_New'] = np.maximum(0, d['disaster_deaths'] * (1 - red))
        d['Disaster_Affected_New'] = np.maximum(0, d['disaster_affected'] * (1 - red))

        mult = 2.0 - d['Resilience_Score']
        res_improv = invest_rate * 5 * mult
        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + res_improv)
        d['Shock_Impact_Score_New'] = np.maximum(0, shock * (1 - red))
        return d

    def scenario_5_education_investment(self, education_boost=0.5,aggregate=True):
        d = self.df.copy()
        edu_u = d['Unemployment with advanced education (% of total labor force with advanced education)']
        improve = np.minimum(0.8, education_boost * (1 + edu_u / 15.0))
        d['Educated_Unemployment_New'] = edu_u * (1 - improve)
        
        gdp_pc = d['GDP per capita (current US$)']
        gdp_pc_norm = self._nz_div(gdp_pc, gdp_pc.max(), 0)
        edu_eff = 1.0 / (1 + edu_u / 10.0)
        prod_boost = improve * 0.8 * edu_eff
        innov = improve * 0.6 * edu_eff * gdp_pc_norm
        
        d['GDP_Growth_New'] = d['GDP growth (annual %)'] + prod_boost
        d['GDP_Per_Capita_New'] = gdp_pc * (1 + innov * 0.1 * (2.0 - gdp_pc_norm))
        
        mult = 2.0 - d['Resilience_Score']
        res_improv = improve * 0.4 * mult
        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + res_improv)
        
        if aggregate:
            d = self._aggregate_by_country(d)
        
        return d

    def scenario_6_integrated_policy(self):
        d = self.df.copy()
        trade_reduction = 0.15
        youth = 0.30
        agr = 0.15
        dis = 0.01
        edu = 0.30

        res_boost = (trade_reduction*0.8 + youth*0.4 + agr*0.6 + dis*5 + edu*0.4) * 0.9
        gdp_boost = (youth*0.6 + agr*0.8 + edu*0.8 - trade_reduction*0.3) * 0.85

        d['Resilience_Score_New'] = np.minimum(1.0, d['Resilience_Score'] + res_boost)
        d['GDP_Growth_New'] = d['GDP growth (annual %)'] + gdp_boost
        d['Shock_Impact_Score_New'] = np.maximum(0, d['Shock_Impact_Score'] * (1 - res_boost * 0.5))
        return d

    def summary_table(self, countries=None):
        if countries is None:
            countries = self.df['Country'].head(5).tolist()
        rows = []

        scenarios = [
            ('Trade_Div', self.scenario_1_trade_diversification),
            ('Youth_Emp', self.scenario_2_youth_employment),
            ('Agr_Prod', self.scenario_3_agricultural_productivity),
            ('Disaster_Prep', self.scenario_4_disaster_preparedness),
            ('Education', self.scenario_5_education_investment),
            ('Integrated', self.scenario_6_integrated_policy),
        ]

        for c in countries:
            base = self.df[self.df['Country'] == c]
            if base.empty:
                continue
            base = base.iloc[0]
            row = {
                'Country': c,
                'Baseline_Resilience': f"{base['Resilience_Score']:.3f}",
                'Baseline_GDP_Growth': f"{base['GDP growth (annual %)']:.2f}%",
                'Baseline_Trade_Dep': f"{base['Trade_Dependency_Index']:.3f}"
            }
            for name, fn in scenarios:
                r = fn()
                cs = r[r['Country'] == c]
                if not cs.empty:
                    new_res = cs['Resilience_Score_New'].iloc[0]
                    row[name] = f"{(new_res - base['Resilience_Score']):.3f}"
            rows.append(row)
        out = pd.DataFrame(rows)
        return out

    def plot_all_scenarios(self, country_sample=None):
        d = self.df
        if country_sample is None:
            country_sample = d['Country'].dropna().unique().tolist()[:10]

        base = d[d['Country'].isin(country_sample)].copy()
        if base.empty:
            st.write("No matching countries to plot.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Policy Scenarios Impact Analysis - What-If Simulations for 2030',
                     fontsize=16, fontweight='bold')
        width = 0.35

        s1 = self.scenario_1_trade_diversification()
        s1 = s1[s1['Country'].isin(country_sample)]
        axes[0, 0].scatter(base['Trade_Dependency_Index'], base['Resilience_Score'], alpha=0.6, label='Baseline', s=60)
        axes[0, 0].scatter(s1['Trade_Dependency_Index_New'], s1['Resilience_Score_New'], alpha=0.8, label='After Trade Diversification', s=60)
        axes[0, 0].set_xlabel('Trade Dependency Index'); axes[0, 0].set_ylabel('Resilience Score'); axes[0, 0].set_title('Scenario 1'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        s2 = self.scenario_2_youth_employment()
        s2 = s2[s2['Country'].isin(country_sample)]
        s2_agg = s2.groupby('Country').first().reset_index()
        base_agg = base.groupby('Country').first().reset_index()
        
        country_order = s2_agg['Country'].tolist()
        x = np.arange(len(country_order))
        
        byu = base_agg['Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)']
        syu = s2_agg['Youth_Unemployment_New']
        
        axes[0, 1].bar(x - width/2, byu, width, label='Baseline', alpha=0.7)
        axes[0, 1].bar(x + width/2, syu, width, label='After Program', alpha=0.7)
        axes[0, 1].set_title('Scenario 2'); axes[0, 1].set_ylabel('Youth Unemployment (%)')
        axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(country_order, rotation=45, ha='right'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        s3 = self.scenario_3_agricultural_productivity(); s3 = s3[s3['Country'].isin(country_sample)]
        axes[1, 0].scatter(base['agr_yield_avg'].values, base['GDP growth (annual %)'].values, alpha=0.6, label='Baseline', s=60)
        axes[1, 0].scatter(s3['Agr_Yield_New'].values[:len(base)], s3['GDP_Growth_New'].values[:len(base)], alpha=0.8, label='After Improvement', s=60)
        axes[1, 0].set_xlabel('Agricultural Yield'); axes[1, 0].set_ylabel('GDP Growth (%)'); axes[1, 0].set_title('Scenario 3'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        s4 = self.scenario_4_disaster_preparedness(); s4 = s4[s4['Country'].isin(country_sample)]
        b_aligned = base.set_index('Country').reindex(s4['Country']).reset_index()
        res_delta = (s4['Resilience_Score_New'] - b_aligned['Resilience_Score']).values
        shock_delta = (b_aligned['Shock_Impact_Score'] - s4['Shock_Impact_Score_New']).values
        ok = ~(np.isnan(res_delta) | np.isnan(shock_delta))
        axes[1, 1].scatter(res_delta[ok], shock_delta[ok], s=80, alpha=0.7)
        axes[1, 1].set_xlabel('Resilience ↑'); axes[1, 1].set_ylabel('Shock Impact ↓'); axes[1, 1].set_title('Scenario 4'); axes[1, 1].grid(True, alpha=0.3)

        s5 = self.scenario_5_education_investment()
        s5 = s5[s5['Country'].isin(country_sample)]
        s5_agg = s5.groupby('Country').first().reset_index()
        base_agg_s5 = base.groupby('Country').first().reset_index()

        country_order_s5 = s5_agg['Country'].tolist()
        x2 = np.arange(len(country_order_s5))
        
        edu_base = base_agg_s5['Unemployment with advanced education (% of total labor force with advanced education)']
        edu_new = s5_agg['Educated_Unemployment_New']
        
        axes[2, 0].bar(x2 - width/2, edu_base, width, label='Baseline', alpha=0.7)
        axes[2, 0].bar(x2 + width/2, edu_new, width, label='After Investment', alpha=0.7)
        axes[2, 0].set_title('Scenario 5'); axes[2, 0].set_ylabel('Educated Unemployment (%)')
        axes[2, 0].set_xticks(x2); axes[2, 0].set_xticklabels(country_order_s5, rotation=45, ha='right'); axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

        s6 = self.scenario_6_integrated_policy(); s6 = s6[s6['Country'].isin(country_sample)]
        metrics = ['Resilience_Score', 'GDP growth (annual %)', 'Shock_Impact_Score']
        base_vals = [base[m].mean() for m in metrics]
        scen_vals = [s6['Resilience_Score_New'].mean(), s6['GDP_Growth_New'].mean(), s6['Shock_Impact_Score_New'].mean()]
        xm = np.arange(3)
        axes[2, 1].bar(xm - width/2, base_vals, width, label='Baseline', alpha=0.7)
        axes[2, 1].bar(xm + width/2, scen_vals, width, label='Integrated', alpha=0.7)
        axes[2, 1].set_xticks(xm); axes[2, 1].set_xticklabels(['Resilience', 'GDP Growth', 'Shock Impact'])
        axes[2, 1].set_title('Scenario 6'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)


    def country_recommendations(self, country: str, top_k=3):
        base_row = self.df[self.df['Country'] == country]
        if base_row.empty:
            return f"No data for {country}."
        base_row = base_row.iloc[0]
        
        scenarios = {
            'Trade Diversification (25%)': self.scenario_1_trade_diversification(),
            'Youth Employment Programs (50%)': self.scenario_2_youth_employment(),
            'Agricultural Productivity (+25%)': self.scenario_3_agricultural_productivity(),
            'Disaster Preparedness (2% GDP)': self.scenario_4_disaster_preparedness(),
            'Education Investment (50%)': self.scenario_5_education_investment(),
            'Integrated Policy Package': self.scenario_6_integrated_policy()
        }
        
        results = []
        for name, df_sc in scenarios.items():
            row = df_sc[df_sc['Country'] == country]
            if row.empty:
                continue
            row = row.iloc[0]
            res_gain = row['Resilience_Score_New'] - base_row['Resilience_Score']
            gdp_gain = row.get('GDP_Growth_New', base_row['GDP growth (annual %)']) - base_row['GDP growth (annual %)']
            shock_red = base_row['Shock_Impact_Score'] - row.get('Shock_Impact_Score_New', base_row['Shock_Impact_Score'])
            score = (res_gain * 0.6) + (self._nz_div(shock_red, max(1e-9, base_row['Shock_Impact_Score']), 0) * 0.25) + (gdp_gain/10.0 * 0.15)
            results.append((name, res_gain, gdp_gain, shock_red, score))
        
        results.sort(key=lambda x: x[-1], reverse=True)
        top = results[:top_k]
        
        recommendations = []
        
        recommendations.append(f"## Strategic Recommendations for {country}")
        recommendations.append(f"*Policy horizon: 2030 | Based on current resilience: {base_row['Resilience_Score']:.3f}*")
        recommendations.append("")
        
        recommendations.append("### **Top Policy Priorities**")
        for i, (name, r, g, s, score) in enumerate(top, 1):
            impact_level = "High Impact" if score > 0.1 else "Medium Impact" if score > 0.05 else "Low Impact"
            recommendations.append(f"**{i}. {name}** ({impact_level})")
            recommendations.append(f"   • Resilience boost: **+{r:.3f}** points")
            recommendations.append(f"   • GDP growth change: **{g:+.2f}** percentage points")
            recommendations.append(f"   • Shock impact reduction: **-{s:.3f}** points")
            recommendations.append("")
        
        recommendations.append("### **Tailored Insights**")
        
        if base_row['Trade_Dependency_Index'] > 0.7:
            recommendations.append("• **Critical Trade Risk**")
            recommendations.append("  - Extremely high trade dependency detected")
            recommendations.append("  - Urgent need for export market diversification")
            recommendations.append("  - Consider regional trade partnerships and domestic market development")
        elif base_row['Trade_Dependency_Index'] > 0.5:
            recommendations.append("• **Moderate Trade Risk**")
            recommendations.append("  - Significant trade dependency requires attention")
            recommendations.append("  - Gradual diversification strategy recommended")
        
        if base_row['Resilience_Score'] < 0.3:
            recommendations.append("• **Low Resilience Alert**")
            recommendations.append("  - Critical vulnerability to external shocks")
            recommendations.append("  - Immediate focus on disaster preparedness essential")
            recommendations.append("  - Build emergency reserves and early warning systems")
        elif base_row['Resilience_Score'] < 0.5:
            recommendations.append("• **Resilience Gap**")
            recommendations.append("  - Below-average shock absorption capacity")
            recommendations.append("  - Strengthen institutional frameworks and social safety nets")
        
        if base_row['agr_production_avg'] > self.df['agr_production_avg'].median():
            recommendations.append("• **Agricultural Advantage**")
            recommendations.append("  - Strong agricultural base identified")
            recommendations.append("  - High potential for productivity improvements")
            recommendations.append("  - Focus on technology adoption, storage, and rural logistics")
            recommendations.append("  - Link smallholder farmers to value chains")
        
        youth_unemployment = base_row['Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)']
        if youth_unemployment > 20:
            recommendations.append("• **Youth Employment Crisis**")
            recommendations.append("  - High youth unemployment demands immediate action")
            recommendations.append("  - Skills training programs aligned with market needs")
            recommendations.append("  - Support for youth entrepreneurship and startups")
        elif youth_unemployment > 15:
            recommendations.append("• **Youth Employment Priority**")
            recommendations.append("  - Moderate youth unemployment needs attention")
            recommendations.append("  - Focus on vocational training and digital skills")
        
        edu_unemployment = base_row['Unemployment with advanced education (% of total labor force with advanced education)']
        if edu_unemployment > 10:
            recommendations.append("• **Skills Mismatch Issue**")
            recommendations.append("  - High educated unemployment suggests skills-jobs mismatch")
            recommendations.append("  - Reform education curricula to match industry needs")
            recommendations.append("  - Strengthen university-industry partnerships")
        
        gdp_pc = base_row['GDP per capita (current US$)']
        if gdp_pc < 5000:
            recommendations.append("• **Development Priority**")
            recommendations.append("  - Low-income context requires inclusive growth strategies")
            recommendations.append("  - Focus on basic infrastructure and human capital")
            recommendations.append("  - Leverage international development partnerships")
        elif gdp_pc > 15000:
            recommendations.append("• **Innovation Focus**")
            recommendations.append("  - Higher-income status enables innovation-led growth")
            recommendations.append("  - Invest in R&D, digitalization, and knowledge economy")
        
        recommendations.append("")
        recommendations.append("### **Implementation Timeline**")
        recommendations.append("")
        recommendations.append("• **Phase 1 (0-12 months)**: Policy framework development and stakeholder alignment")
        recommendations.append("")
        recommendations.append("• **Phase 2 (1-3 years)**: Pilot programs and initial rollout of priority interventions")
        recommendations.append("")
        recommendations.append("• **Phase 3 (3-7 years)**: Full-scale implementation and impact measurement")
        recommendations.append("")
        recommendations.append("• **Phase 4 (7-10 years)**: Evaluation, adjustment, and sustainability planning")
        recommendations.append("")
        
        recommendations.append("")
        recommendations.append("### **Risk Mitigation**")
        recommendations.append("")
        if base_row['Shock_Impact_Score'] > 0.7:
            recommendations.append("• **High shock vulnerability**: Build contingency funds and flexible policy mechanisms")
            recommendations.append("")
        recommendations.append("• **Political economy**: Ensure broad coalition support for sustained implementation")
        recommendations.append("")
        recommendations.append("• **Capacity constraints**: Invest in institutional capacity building alongside policy reforms")
        recommendations.append("")
        recommendations.append("• **External dependencies**: Monitor global economic conditions and maintain policy flexibility")
        recommendations.append("")
        
        return "\n".join(recommendations)

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df

def main():
    st.title("Policy Scenario Explorer — What-If Simulations for 2030")

    data_file = "datasets/integrated_tren_dataset_with_indexes.csv"
    df = load_data(data_file)
    analyzer = PolicyScenarioAnalyzer(df)

    countries = df['Country'].dropna().unique().tolist()
    sorted_countries = sorted(countries)

    st.markdown("### Select Countries for Analysis")
    selected_countries = st.multiselect(
        "Choose countries to analyze:",
        options=sorted_countries,
        default=sorted_countries[:5],
        key="country_selector"
    )

    if not selected_countries:
        st.warning("Please select at least one country to begin analysis.")
        st.stop()

    with st.sidebar:
        st.header("Policy Configuration")

        use_trade = st.checkbox("Trade Diversification (25%)", True)
        use_youth = st.checkbox("Youth Employment Programs (50%)", True)
        use_agri = st.checkbox("Agricultural Productivity (+25%)", True)
        use_dis = st.checkbox("Disaster Preparedness (2% GDP)", True)
        use_edu = st.checkbox("Education Investment (50%)", True)
        use_int = st.checkbox("Integrated Policy Package", True)

        st.markdown("---")
        with st.expander("Advanced Settings"):
            trade_intensity = st.slider("Trade Diversification Intensity", 0.1, 0.5, 0.25, 0.05)
            youth_target = st.slider("Youth Unemployment Reduction Target", 0.2, 0.8, 0.5, 0.1)
            agri_boost = st.slider("Agricultural Productivity Increase", 0.1, 0.5, 0.25, 0.05)
            disaster_investment = st.slider("Disaster Preparedness Investment (% of GDP)", 0.01, 0.05, 0.02, 0.01)
            edu_investment = st.slider("Education Investment Intensity", 0.2, 0.8, 0.5, 0.1)

        st.markdown("---")
        st.subheader("Analysis Options")
        show_summary = st.checkbox("Show Summary Table", True)
        show_plots = st.checkbox("Show Scenario Plots", True)
        show_recommendations = st.checkbox("Show Country Recommendations", True)
        st.markdown("---")
        st.caption(" **Tip:** Toggle policies to focus on specific interventions. Use advanced settings to fine-tune scenario parameters.")
    
    if not selected_countries:
        st.warning(" Please select at least one country from the navbar above to begin analysis.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries Selected", len(selected_countries))
    with col2:
        avg_resilience = df[df['Country'].isin(selected_countries)]['Resilience_Score'].mean()
        st.metric("Avg. Resilience Score", f"{avg_resilience:.3f}")
    with col3:
        avg_gdp_growth = df[df['Country'].isin(selected_countries)]['GDP growth (annual %)'].mean()
        st.metric("Avg. GDP Growth", f"{avg_gdp_growth:.2f}%")
    with col4:
        avg_trade_dep = df[df['Country'].isin(selected_countries)]['Trade_Dependency_Index'].mean()
        st.metric("Avg. Trade Dependency", f"{avg_trade_dep:.3f}")
    
    st.markdown("---")
    
    if show_summary:
        st.subheader("Scenario Impact Summary")
        summary = analyzer.summary_table(selected_countries)
        st.dataframe(summary, use_container_width=True)
        
        csv = summary.to_csv(index=False)
        st.download_button(
            label=" Download Summary as CSV",
            data=csv,
            file_name=f"policy_scenario_summary_{len(selected_countries)}_countries.csv",
            mime="text/csv"
        )
    
    if show_plots:
        st.subheader("Multi-Scenario Comparison")
        
        num_countries = len(selected_countries)
        fig_width = max(16, num_countries * 1.5)
        fig, axes = plt.subplots(3, 2, figsize=(fig_width, 18))
        fig.suptitle('Policy Scenarios Impact Analysis - What-If Simulations for 2030',
                     fontsize=16, fontweight='bold')
        width = 0.35
        
        base = df[df['Country'].isin(selected_countries)].copy()
        
        if use_trade:
            s1 = analyzer.scenario_1_trade_diversification(trade_intensity)
            s1 = s1[s1['Country'].isin(selected_countries)]
            axes[0, 0].scatter(s1['Trade_Dependency_Index'], s1['Resilience_Score'], alpha=0.6, label='Baseline', s=60)
            axes[0, 0].scatter(s1['Trade_Dependency_Index_New'], s1['Resilience_Score_New'], alpha=0.8, label='After Trade Diversification', s=60)
            axes[0, 0].set_xlabel('Trade Dependency Index')
            axes[0, 0].set_ylabel('Resilience Score')
            axes[0, 0].set_title('Scenario 1: Trade Diversification')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Trade Diversification\nDisabled', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Scenario 1: Trade Diversification (Disabled)')
        
        if use_youth:
            s2 = analyzer.scenario_2_youth_employment(youth_target)
            s2 = s2[s2['Country'].isin(selected_countries)]
            x = np.arange(len(s2))
            byu = s2['Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)'].values
            syu = s2['Youth_Unemployment_New'].values
            axes[0, 1].bar(x - width/2, byu, width, label='Baseline', alpha=0.7)
            axes[0, 1].bar(x + width/2, syu, width, label='After Program', alpha=0.7)
            axes[0, 1].set_title('Scenario 2: Youth Employment Programs')
            axes[0, 1].set_ylabel('Youth Unemployment (%)')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(s2['Country'].tolist(), rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Youth Employment\nPrograms Disabled', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Scenario 2: Youth Employment (Disabled)')
        
        if use_agri:
            s3 = analyzer.scenario_3_agricultural_productivity(agri_boost)
            s3 = s3[s3['Country'].isin(selected_countries)]
            axes[1, 0].scatter(s3['agr_yield_avg'].values, s3['GDP growth (annual %)'].values, alpha=0.6, label='Baseline', s=60)
            axes[1, 0].scatter(s3['Agr_Yield_New'].values, s3['GDP_Growth_New'].values, alpha=0.8, label='After Improvement', s=60)
            axes[1, 0].set_xlabel('Agricultural Yield')
            axes[1, 0].set_ylabel('GDP Growth (%)')
            axes[1, 0].set_title('Scenario 3: Agricultural Productivity')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Agricultural Productivity\nDisabled', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Scenario 3: Agricultural Productivity (Disabled)')
        
        if use_dis:
            s4 = analyzer.scenario_4_disaster_preparedness(disaster_investment)
            s4 = s4[s4['Country'].isin(selected_countries)]
            b_aligned = base.set_index('Country').reindex(s4['Country']).reset_index()
            res_delta = (s4['Resilience_Score_New'] - b_aligned['Resilience_Score']).values
            shock_delta = (b_aligned['Shock_Impact_Score'] - s4['Shock_Impact_Score_New']).values
            ok = ~(np.isnan(res_delta) | np.isnan(shock_delta))
            axes[1, 1].scatter(res_delta[ok], shock_delta[ok], s=80, alpha=0.7)
            axes[1, 1].set_xlabel('Resilience Improvement')
            axes[1, 1].set_ylabel('Shock Impact Reduction')
            axes[1, 1].set_title('Scenario 4: Disaster Preparedness')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Disaster Preparedness\nDisabled', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Scenario 4: Disaster Preparedness (Disabled)')
        
        if use_edu:
            s5 = analyzer.scenario_5_education_investment(edu_investment)
            s5 = s5[s5['Country'].isin(selected_countries)]
            x2 = np.arange(len(s5))
            axes[2, 0].bar(x2 - width/2,
                          s5['Unemployment with advanced education (% of total labor force with advanced education)'].values,
                          width, label='Baseline', alpha=0.7)
            axes[2, 0].bar(x2 + width/2, s5['Educated_Unemployment_New'].values, width, label='After Investment', alpha=0.7)
            axes[2, 0].set_title('Scenario 5: Education Investment')
            axes[2, 0].set_ylabel('Educated Unemployment (%)')
            axes[2, 0].set_xticks(x2)
            axes[2, 0].set_xticklabels(s5['Country'].tolist(), rotation=45, ha='right')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Education Investment\nDisabled', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Scenario 5: Education Investment (Disabled)')
        
        if use_int:
            s6 = analyzer.scenario_6_integrated_policy()
            s6 = s6[s6['Country'].isin(selected_countries)]
            metrics = ['Resilience_Score', 'GDP growth (annual %)', 'Shock_Impact_Score']
            base_vals = [base[m].mean() for m in metrics]
            scen_vals = [s6['Resilience_Score_New'].mean(), s6['GDP_Growth_New'].mean(), s6['Shock_Impact_Score_New'].mean()]
            xm = np.arange(3)
            axes[2, 1].bar(xm - width/2, base_vals, width, label='Baseline', alpha=0.7)
            axes[2, 1].bar(xm + width/2, scen_vals, width, label='Integrated', alpha=0.7)
            axes[2, 1].set_xticks(xm)
            axes[2, 1].set_xticklabels(['Resilience', 'GDP Growth', 'Shock Impact'])
            axes[2, 1].set_title('Scenario 6: Integrated Policy Package')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'Integrated Policy\nPackage Disabled', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Scenario 6: Integrated Policy (Disabled)')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    if show_recommendations and selected_countries:
        rec_text = analyzer.country_recommendations(selected_countries[0])
        st.markdown(rec_text)
        
        if len(selected_countries) > 1:
            with st.expander(f"View recommendations for other countries ({len(selected_countries)-1} more)"):
                for country in selected_countries[1:4]:  
                    rec_text_other = analyzer.country_recommendations(country, top_k=2) 
                    st.markdown(rec_text_other)
                    st.markdown("---")
    
    st.markdown("---")
    st.subheader("Detailed Policy Impact Analysis")
    
    details = []
    base = df[df['Country'] == selected_countries[0]].iloc[0]
    
    if use_trade: details.append(("Trade Diversification", analyzer.scenario_1_trade_diversification(trade_intensity)))
    if use_youth: details.append(("Youth Employment", analyzer.scenario_2_youth_employment(youth_target)))
    if use_agri:  details.append(("Agricultural Productivity", analyzer.scenario_3_agricultural_productivity(agri_boost)))
    if use_dis:   details.append(("Disaster Preparedness", analyzer.scenario_4_disaster_preparedness(disaster_investment)))
    if use_edu:   details.append(("Education Investment", analyzer.scenario_5_education_investment(edu_investment)))
    if use_int:   details.append(("Integrated Package", analyzer.scenario_6_integrated_policy()))
    
    rows = []
    for name, sdf in details:
        row = sdf[sdf['Country'] == selected_countries[0]].iloc[0]
        rows.append({
            "Scenario": name,
            "Resilience Δ": f"{row['Resilience_Score_New'] - base['Resilience_Score']:+.3f}",
            "GDP Growth Δ (pp)": f"{row.get('GDP_Growth_New', base['GDP growth (annual %)']) - base['GDP growth (annual %)']:+.2f}",
            "Shock Impact Δ": f"{base['Shock_Impact_Score'] - row.get('Shock_Impact_Score_New', base['Shock_Impact_Score']):+.3f}"
        })
    
    if rows:
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        csv_detailed = results_df.to_csv(index=False)
        st.download_button(
            label=" Download Detailed Analysis",
            data=csv_detailed,
            file_name=f"detailed_policy_analysis_{selected_countries[0]}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()