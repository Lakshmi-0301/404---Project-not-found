import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "datasets/processed/integrated_tren_dataset.csv"
df = pd.read_csv(file_path)
scaler = MinMaxScaler()

cols_for_trade_dependency = [
    "Imports of goods and services (% of GDP)",
    "Exports of goods and services (% of GDP)"
]

cols_for_resilience = [
    "GDP growth (annual %)",
    "Employment to population ratio, 15+, female (%) (modeled ILO estimate)",
    "Employment to population ratio, 15+, male (%) (modeled ILO estimate)",
    "Life expectancy at birth, total (years)",
    "disaster_deaths",
    "disaster_affected"
]

cols_for_spending_efficiency = [
    "GDP per capita (current US$)",
    "External debt stocks (% of GNI)",
    "Inflation, consumer prices (annual %)"
]

cols_for_shock_impact = [
    "disaster_deaths",
    "disaster_injured",
    "disaster_affected",
    "disaster_damage_usd_thousands",
    "pop_total_population___both_sexes"
]

df_scaled = df.copy()
for cols in [cols_for_trade_dependency, cols_for_resilience,
             cols_for_spending_efficiency, cols_for_shock_impact]:
    for c in cols:
        if c in df.columns:  
            df_scaled[c] = scaler.fit_transform(df[[c]])

df_scaled["Trade_Dependency_Index"] = (
    df_scaled["Imports of goods and services (% of GDP)"] + 
    df_scaled["Exports of goods and services (% of GDP)"])/ 2

df_scaled["Resilience_Score"] = (
    df_scaled["GDP growth (annual %)"] +
    (df_scaled["Employment to population ratio, 15+, female (%) (modeled ILO estimate)"] +
     df_scaled["Employment to population ratio, 15+, male (%) (modeled ILO estimate)"]) / 2 +
    df_scaled["Life expectancy at birth, total (years)"]) / 3 - ((df_scaled["disaster_deaths"] + df_scaled["disaster_affected"]) / 2)

df_scaled["Spending_Efficiency"] = df_scaled["GDP per capita (current US$)"] / (
    df_scaled["External debt stocks (% of GNI)"] + df_scaled["Inflation, consumer prices (annual %)"] + 1e-6)

df_scaled["Shock_Impact_Score"] = (
    df_scaled["disaster_deaths"] 
    + df_scaled["disaster_injured"] 
    + df_scaled["disaster_affected"] 
    + df_scaled["disaster_damage_usd_thousands"]) / (df_scaled["pop_total_population___both_sexes"] + 1e-6)

output_path = "datasets/processed/integrated_tren_dataset.csv"
df_scaled.to_csv(output_path, index=False)

print("Feature engineering completed. File saved at:", output_path)
