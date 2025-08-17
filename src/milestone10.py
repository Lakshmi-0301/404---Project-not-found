import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"datasets/processed/integrated_tren_dataset.csv")
        return df
    except:
        st.error("Could not load the dataset. Please check the file path.")
        return None

def prepare_features(df):
    feature_cols = [
        'Exports of goods and services (% of GDP)',
        'GDP (current US$)',
        'GDP growth (annual %)',
        'GDP per capita (current US$)',
        'Imports of goods and services (% of GDP)',
        'Inflation, consumer prices (annual %)',
        'Trade (% of GDP)',
        'Employment to population ratio, 15+, female (%) (modeled ILO estimate)',
        'Employment to population ratio, 15+, male (%) (modeled ILO estimate)',
        'Current account balance (% of GDP)',
        'Foreign direct investment, net inflows (% of GDP)',
        'Life expectancy at birth, total (years)',
        'Population growth (annual %)',
        'Urban population (% of total population)',
        'Trade_Dependency_Index',
        'Resilience_Score',
        'Spending_Efficiency',
        'Shock_Impact_Score'
    ]
    
    target_cols = {
        'gdp': 'GDP (current US$)',
        'poverty': 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)',
        'unemployment': 'Unemployment, total (% of total labor force) (modeled ILO estimate)_y'
    }
    
    return feature_cols, target_cols

def train_models(df, feature_cols, target_cols):
    models = {}
    scalers = {}
    performance = {}
    
    for target_name, target_col in target_cols.items():
        df_clean = df.dropna(subset=feature_cols + [target_col])
        
        if len(df_clean) < 50:
            continue
            
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if target_name == 'gdp':
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=150, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if target_name == 'gdp':
            mae_percentage = (mae / y_test.mean()) * 100
            performance[target_name] = {'mae': mae, 'mae_percentage': mae_percentage, 'r2': r2}
        else:
            performance[target_name] = {'mae': mae, 'r2': r2}
        
        models[target_name] = model
        scalers[target_name] = scaler
    
    return models, scalers, performance

def apply_disaster_shock(base_features, severity=8.5):
    features = base_features.copy()
    
    shock_multiplier = min(severity / 10, 1.0)
    
    features['GDP growth (annual %)'] *= (1 - 0.15 * shock_multiplier)
    features['Trade (% of GDP)'] *= (1 - 0.12 * shock_multiplier)
    features['Foreign direct investment, net inflows (% of GDP)'] *= (1 - 0.25 * shock_multiplier)
    features['Resilience_Score'] *= (1 - 0.20 * shock_multiplier)
    features['Shock_Impact_Score'] += 0.3 * shock_multiplier
    
    return features

def apply_trade_war_shock(base_features, trade_reduction=0.20):
    features = base_features.copy()
    
    features['Trade (% of GDP)'] *= (1 - trade_reduction)
    features['Exports of goods and services (% of GDP)'] *= (1 - trade_reduction * 0.8)
    features['Imports of goods and services (% of GDP)'] *= (1 - trade_reduction * 0.8)
    features['GDP growth (annual %)'] *= (1 - trade_reduction * 0.5)
    features['Trade_Dependency_Index'] *= (1 - trade_reduction * 0.6)
    
    return features

def get_country_latest_data(df, country):
    country_data = df[df['Country'] == country].sort_values('Year').iloc[-1]
    return country_data

def predict_scenarios(df, models, scalers, feature_cols):
    countries = df['Country'].unique()
    predictions = {}
    
    for country in countries:
        try:
            latest_data = get_country_latest_data(df, country)
            base_features = latest_data[feature_cols].to_dict()
            
            baseline_pred = {}
            disaster_pred = {}
            trade_war_pred = {}
            combined_pred = {}
            
            for target_name, model in models.items():
                if target_name in scalers:
                    scaler = scalers[target_name]
                    
                    baseline_features = pd.DataFrame([base_features])
                    baseline_scaled = scaler.transform(baseline_features.fillna(baseline_features.mean()))
                    baseline_pred[target_name] = model.predict(baseline_scaled)[0]
                    
                    disaster_features = apply_disaster_shock(base_features)
                    disaster_df = pd.DataFrame([disaster_features])
                    disaster_scaled = scaler.transform(disaster_df.fillna(disaster_df.mean()))
                    disaster_pred[target_name] = model.predict(disaster_scaled)[0]
                    
                    trade_war_features = apply_trade_war_shock(base_features)
                    trade_war_df = pd.DataFrame([trade_war_features])
                    trade_war_scaled = scaler.transform(trade_war_df.fillna(trade_war_df.mean()))
                    trade_war_pred[target_name] = model.predict(trade_war_scaled)[0]
                    
                    combined_features = apply_trade_war_shock(apply_disaster_shock(base_features))
                    combined_df = pd.DataFrame([combined_features])
                    combined_scaled = scaler.transform(combined_df.fillna(combined_df.mean()))
                    combined_pred[target_name] = model.predict(combined_scaled)[0]
            
            predictions[country] = {
                'baseline': baseline_pred,
                'disaster': disaster_pred,
                'trade_war': trade_war_pred,
                'combined': combined_pred
            }
        except:
            continue
    
    return predictions

def create_comparison_chart(predictions, metric, top_n=15):
    data = []
    
    for country, preds in predictions.items():
        if metric in preds['baseline']:
            baseline = preds['baseline'][metric]
            disaster = preds['disaster'][metric]
            trade_war = preds['trade_war'][metric]
            combined = preds['combined'][metric]
            
            data.append({
                'Country': country,
                'Baseline': baseline,
                'Disaster (2026)': disaster,
                'Trade War (2027)': trade_war,
                'Combined Shocks': combined,
                'Disaster Impact': ((disaster - baseline) / baseline * 100) if baseline != 0 else 0,
                'Trade War Impact': ((trade_war - baseline) / baseline * 100) if baseline != 0 else 0,
                'Combined Impact': ((combined - baseline) / baseline * 100) if baseline != 0 else 0
            })
    
    df_viz = pd.DataFrame(data)
    
    if metric == 'gdp':
        df_viz = df_viz.nlargest(top_n, 'Baseline')
        impact_col = 'Combined Impact'
        df_viz = df_viz.sort_values(impact_col)
    else:
        df_viz = df_viz.nlargest(top_n, 'Combined Impact')
    
    return df_viz

def main():
    st.title("Economic Prediction Dashboard 2030")
    st.subheader("Impact Analysis of Disasters and Trade Wars")
    
    df = load_data()
    if df is None:
        return
    
    feature_cols, target_cols = prepare_features(df)
    
    with st.spinner("Training predictive models..."):
        models, scalers, performance = train_models(df, feature_cols, target_cols)
    
    st.subheader("Model Performance")
    perf_row1_col1, perf_row1_col2, perf_row1_col3 = st.columns(3)
    perf_row2_col1, perf_row2_col2, perf_row2_col3 = st.columns(3)
    
    with perf_row1_col1:
        if 'gdp' in performance:
            st.metric("GDP Model R²", f"{performance['gdp']['r2']:.3f}")
    
    with perf_row1_col2:
        if 'poverty' in performance:
            st.metric("Poverty Model R²", f"{performance['poverty']['r2']:.3f}")
    
    with perf_row1_col3:
        if 'unemployment' in performance:
            st.metric("Unemployment Model R²", f"{performance['unemployment']['r2']:.3f}")
    
    with perf_row2_col1:
        if 'gdp' in performance:
            st.metric("GDP Model MAE", f"{performance['gdp']['mae_percentage']:.2f}%")
    
    with perf_row2_col2:
        if 'poverty' in performance:
            st.metric("Poverty Model MAE", f"{performance['poverty']['mae']:.3f}%")
    
    with perf_row2_col3:
        if 'unemployment' in performance:
            st.metric("Unemployment Model MAE", f"{performance['unemployment']['mae']:.3f}%")
    
    with st.spinner("Generating predictions for all countries..."):
        predictions = predict_scenarios(df, models, scalers, feature_cols)
    
    st.subheader("Scenario Predictions")
    
    tab1, tab2, tab3, tab4 = st.tabs(["GDP Analysis", "Poverty Analysis", "Unemployment Analysis", "Country Comparison"])
    
    with tab1:
        if 'gdp' in models:
            st.subheader("GDP Predictions (Current US$)")
            
            gdp_data = create_comparison_chart(predictions, 'gdp', 20)
            
            fig1 = px.bar(gdp_data.head(15), 
                        x='Country', 
                        y=['Baseline', 'Disaster (2026)', 'Trade War (2027)', 'Combined Shocks'],
                        title="GDP Predictions when a disaster hits in 2026",
                        barmode='group')
            fig1.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True, key="gdp_predictions_main")
            
            st.subheader("GDP Impact Percentage")
            impact_fig1 = px.bar(gdp_data.head(15),
                              x='Country',
                              y=['Disaster Impact', 'Trade War Impact', 'Combined Impact'],
                              title="GDP Impact when a Trade War hits in 2027",
                              barmode='group')
            impact_fig1.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(impact_fig1, use_container_width=True, key="gdp_impact_main")
    
    with tab2:
        if 'poverty' in models:
            st.subheader("Poverty Rate Predictions (% of population)")
            
            poverty_data = create_comparison_chart(predictions, 'poverty', 15)
            
            fig2 = px.bar(poverty_data, 
                        x='Country', 
                        y=['Baseline', 'Disaster (2026)', 'Trade War (2027)', 'Combined Shocks'],
                        title="Poverty Rate Predictions when a disaster hits in 2026",
                        barmode='group')
            fig2.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True, key="poverty_predictions_main")
            
            st.subheader("Most Vulnerable Countries (Highest Poverty Increase)")
            vulnerable = poverty_data.nlargest(10, 'Combined Impact')[['Country', 'Baseline', 'Combined Shocks', 'Combined Impact']]
            st.dataframe(vulnerable, use_container_width=True)
    
    with tab3:
        if 'unemployment' in models:
            st.subheader("Unemployment Rate Predictions (% of labor force)")
            
            unemployment_data = create_comparison_chart(predictions, 'unemployment', 15)
            
            fig3 = px.bar(unemployment_data, 
                        x='Country', 
                        y=['Baseline', 'Disaster (2026)', 'Trade War (2027)', 'Combined Shocks'],
                        title="Unemployment Rate Predictions when a trade war hits in 2027",
                        barmode='group')
            fig3.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True, key="unemployment_predictions_main")
            
            st.subheader("Countries with Highest Unemployment Risk")
            risk_countries = unemployment_data.nlargest(10, 'Combined Impact')[['Country', 'Baseline', 'Combined Shocks', 'Combined Impact']]
            st.dataframe(risk_countries, use_container_width=True)
    
    with tab4:
        st.subheader("Individual Country Analysis")
        
        available_countries = list(predictions.keys())
        selected_country = st.selectbox("Select Country", available_countries)
        
        if selected_country and selected_country in predictions:
            country_pred = predictions[selected_country]
            
            metrics_data = []
            
            for metric in ['gdp', 'poverty', 'unemployment']:
                if metric in country_pred['baseline']:
                    baseline = country_pred['baseline'][metric]
                    combined = country_pred['combined'][metric]
                    
                    if metric == 'gdp':
                        change = ((combined - baseline) / baseline * 100) if baseline != 0 else 0
                        metrics_data.append(['GDP (Trillion USD)', f"${baseline/1e12:.2f}T", f"${combined/1e12:.2f}T", f"{change:.1f}%"])
                    elif metric == 'poverty':
                        change = ((combined - baseline) / baseline * 100) if baseline != 0 else 0
                        metrics_data.append(['Poverty Rate (%)', f"{baseline:.1f}%", f"{combined:.1f}%", f"{change:+.1f}%"])
                    else:  
                        change = ((combined - baseline) / baseline * 100) if baseline != 0 else 0
                        metrics_data.append(['Unemployment Rate (%)', f"{baseline:.1f}%", f"{combined:.1f}%", f"{change:+.1f}%"])
            
            metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Baseline 2030', 'Combined Shocks', 'Change'])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            scenarios = ['Baseline', 'Disaster Only', 'Trade War Only', 'Combined Shocks']
            
            for i, metric in enumerate(['gdp', 'poverty', 'unemployment']):
                if metric in country_pred['baseline']:
                    values = [
                        country_pred['baseline'][metric],
                        country_pred['disaster'][metric],
                        country_pred['trade_war'][metric],
                        country_pred['combined'][metric]
                    ]
                    
                    country_fig = go.Figure()
                    country_fig.add_trace(go.Bar(x=scenarios, y=values, name=metric.upper()))
                    
                    if metric == 'gdp':
                        title = f"{selected_country} - GDP Predictions (USD)"
                        country_fig.update_layout(yaxis=dict(tickformat='$.2s'))
                    elif metric == 'poverty':
                        title = f"{selected_country} - Poverty Rate Predictions (%)"
                    else:
                        title = f"{selected_country} - Unemployment Rate Predictions (%)"
                    
                    country_fig.update_layout(title=title, height=400)
                    st.plotly_chart(country_fig, use_container_width=True, key=f"country_{metric}_{selected_country.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}")
    
    st.subheader("Key Findings")
    
    if predictions:
        gdp_impacts = []
        poverty_impacts = []
        unemployment_impacts = []
        
        for country, preds in predictions.items():
            if 'gdp' in preds['baseline'] and 'gdp' in preds['combined']:
                baseline_gdp = preds['baseline']['gdp']
                combined_gdp = preds['combined']['gdp']
                if baseline_gdp != 0:
                    impact = (combined_gdp - baseline_gdp) / baseline_gdp * 100
                    gdp_impacts.append(impact)
            
            if 'poverty' in preds['baseline'] and 'poverty' in preds['combined']:
                baseline_pov = preds['baseline']['poverty']
                combined_pov = preds['combined']['poverty']
                if baseline_pov != 0:
                    impact = (combined_pov - baseline_pov) / baseline_pov * 100
                    poverty_impacts.append(impact)
            
            if 'unemployment' in preds['baseline'] and 'unemployment' in preds['combined']:
                baseline_unemp = preds['baseline']['unemployment']
                combined_unemp = preds['combined']['unemployment']
                if baseline_unemp != 0:
                    impact = (combined_unemp - baseline_unemp) / baseline_unemp * 100
                    unemployment_impacts.append(impact)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if gdp_impacts:
                avg_gdp_impact = np.mean(gdp_impacts)
                st.metric("Average GDP Impact", f"{avg_gdp_impact:.1f}%")
        
        with col2:
            if poverty_impacts:
                avg_poverty_impact = np.mean(poverty_impacts)
                st.metric("Average Poverty Change", f"{avg_poverty_impact:+.1f}%")
        
        with col3:
            if unemployment_impacts:
                avg_unemployment_impact = np.mean(unemployment_impacts)
                st.metric("Average Unemployment Change", f"{avg_unemployment_impact:+.1f}%")
    
    with st.expander("Methodology"):
        st.write("**Disaster Shock (2026):** Reduces GDP growth by 15%, trade by 12%, FDI by 25%, and resilience score by 20%")
        st.write("**Trade War Shock (2027):** Reduces global trade by 20%, exports/imports by 16%, and GDP growth by 10%")
        st.write("**Models:** Gradient Boosting for GDP, Random Forest for poverty and unemployment")
        st.write("**Features:** 18 economic indicators including trade dependency, resilience scores, and shock impact measures")
        st.write("**Changes are shown as percentage changes rather than percentage points**")

if __name__ == "__main__":
    main()