# Trade Resilience & Economic Networks (TREN)

## Overview
The TREN Dataset Project is an analytics initiative for a coalition of 25 nations, ranging from economic superpowers to developing economies. The project analyzes historical data from 2000–2024 on economic, trade, demographic, agricultural and welfare metrics to predict economic outcomes in 2030 under various crisis and policy scenarios. It aims to strengthen resilience against global recessions, trade disputes, climate disasters, pandemics and political instability by providing data-driven recommendations.

Key objectives include data integration, feature engineering, modeling & forecasting, visualization & insights and policy recommendations.

## Dataset Overview
The dataset covers multiple themes from 2000–2024:
1. **Trade Data (Exports & Imports)**: Value (USD), volume (tonnes), and trade partners; split into 2000–2012 and 2013–2024 segments.
2. **Core Economic Indicators**: GDP, GDP per capita, growth, inflation, trade % of GDP, government spending % of GDP.
3. **Crop & Livestock**: Food production index, yield, livestock counts, agricultural exports/imports.
4. **Disasters**: Year, type, severity index, damages (USD), lives lost, recovery years.
5. **Employment & Unemployment**: Employment-to-population ratios, unemployment %, labour force participation.
6. **Population & Demographics**: Total population, urbanization %, growth %, median age.
7. **Resilience Metrics**: Vulnerability score, trade diversification index, disaster recovery score.
8. **Social & Welfare**: Poverty rates ($2.15/day, $3.65/day, national line), life expectancy, Gini index, HDI.

## Challenge Tasks
1. **Data Integration & Cleaning**: Merge datasets into a Country–Year framework; handle missing data, conflicts, and unit normalization.
2. **Feature Engineering**: Create composite indexes like Trade Dependency Index, Resilience Score, Spending Efficiency, Shock Impact Score.
3. **Modeling & Forecasting**: Predict GDP growth, poverty rates, and trade resilience for 2030 under baseline, increased social spending, trade diversification, and global crisis scenarios.
4. **Visualization & Insights**: Generate heatmaps, trade network graphs, shock maps; identify top 3 vulnerabilities per nation.
5. **Policy Recommendations**: Provide country-specific strategies; simulate "what-if" scenarios.

## Problems to Solve
The project addresses 13 specific problems, including:
- Identifying countries at risk from single-partner trade collapse and simulating GDP impacts.
- Modeling cascading effects from China's export drop.
- Assessing drought impacts on agricultural exports.
- Evaluating food security risks from import dependencies.
- Predicting youth unemployment under slowdowns.
- Analyzing export risks from ageing demographics.
- Building and disrupting global trade networks.
- Identifying mutually beneficial trade pairs and simulating collapses.
- Recommending new trade partners.
- Predicting outcomes under disaster or trade war scenarios.
- Modeling best/worst-case 2030 GDP and poverty.
- Identifying top resilience countries.
- Optimizing new trade links to minimize GDP losses under constraints.

## File Structure
- `dashboard/`: Contains summary_dashboard.png for overview visualizations.
- `data/`
  - `processed/`: integrated_tren_dataset.csv (preprocessed data).
  - `raw/`: Original datasets.
- `integrated_tren_dataset.csv`: Dataset with engineered features.
- `heatmaps/`: Heatmap visualizations.
- `shockmap/`: Shock impact maps.
- `src/`: Source code for data preprocessing, feature engineering, visualization, and milestones.
- `trade_networks/`: Trade network graphs.
- `vulnerabilities/`: Vulnerability analyses.
- `app.py`: Main application script.
- `requirements.txt`: Project dependencies.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Lakshmi-0301/404---Project-not-found.git
   ```
2. Navigate to the project directory:
   ```
   cd your-project-directory
   ```
3. Install dependencies (Python 3.8+ recommended):
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing**:
   - Run to generate `data/processed/integrated_tren_dataset.csv`:
     ```
     python src/data_processor.py
     ```
2. **Feature Engineering**:
   - Run to generate feature engineered `integrated_tren_dataset.csv`:
     ```
     python src/feature_engineering.py
     ```
3. **Run the Application**:
   - Launch the dashboard and analyses:
     ```
     python app.py
     ```
4. **Visualizations & Models**:
   - Scripts in `src/` handle specific tasks like visualizations, modeling, and simulations.


## Contributors

**Anandika M**  
Applied Mathematics and Computational Sciences  
PSG College of Technology  

**R Lakshmi**  
Applied Mathematics and Computational Sciences  
PSG College of Technology  

**Thirumagal Meena A**  
Applied Mathematics and Computational Sciences  
PSG College of Technology
