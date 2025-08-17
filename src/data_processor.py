import pandas as pd
import numpy as np
import os
import warnings

BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, 'datasets')
PROCESSED_DIR = os.path.join(BASE_DIR, 'datasets/processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

TARGET_COUNTRIES = [
    'Afghanistan', 'Argentina', 'Australia', 'Bangladesh', 'Belgium', 
    'Canada', 'China', 'Croatia', 'France', 'Germany', 'India', 'Iran', 
    'Iraq', 'Israel', 'Italy', 'Japan', 'Pakistan', 'Portugal', 'Russia', 
    'Saudi Arabia', 'Spain', 'Sri Lanka', 'Sweden', 'United Kingdom', 'United States'
]

COUNTRY_MAPPING = {
    'Iran (Islamic Republic of)': 'Iran',
    'Russian Federation': 'Russia',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United States of America': 'United States'
}

ANALYSIS_YEARS = list(range(2000, 2025))


class TRENDataCleaner:

    def __init__(self):
        self.master_df = None
        self.country_year_framework = self.create_master_framework()

    def create_master_framework(self) -> pd.DataFrame:
        countries, years = [], []
        for country in TARGET_COUNTRIES:
            for year in ANALYSIS_YEARS:
                countries.append(country)
                years.append(year)

        return pd.DataFrame({'Country': countries, 'Year': years})

    def standardize_country_names(self, df: pd.DataFrame, country_col: str) -> pd.DataFrame:
        df[country_col] = df[country_col].replace(COUNTRY_MAPPING)
        return df

    def clean_trade_data(self, file_path: str, trade_type: str) -> pd.DataFrame:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                return pd.DataFrame()

            df = self.standardize_country_names(df, 'reporterDesc')
            df = df[df['reporterDesc'].isin(TARGET_COUNTRIES)].copy()

            trade_df = df[['reporterDesc', 'refYear', 'primaryValue']].copy()
            trade_df.columns = ['Country', 'Year', f'{trade_type.lower()}_value_usd']

            trade_df[f'{trade_type.lower()}_value_usd'] = pd.to_numeric(
                trade_df[f'{trade_type.lower()}_value_usd'], errors='coerce'
            )
            trade_df = trade_df.dropna(subset=[f'{trade_type.lower()}_value_usd'])
            trade_df = trade_df.groupby(['Country', 'Year'], as_index=False).sum()

            return trade_df
        except Exception:
            return pd.DataFrame()

    def process_all_trade_data(self) -> pd.DataFrame:
        trade_files = [
            ('2000-2012_Export.csv', 'Export'),
            ('2013-2024_Export.csv', 'Export'),
            ('2000-2012_Import.csv', 'Import'),
            ('2013-2024_Import.csv', 'Import')
        ]
        all_trade_data = []
        for filename, trade_type in trade_files:
            file_path = os.path.join(RAW_DIR, filename)
            if os.path.exists(file_path):
                trade_df = self.clean_trade_data(file_path, trade_type)
                if not trade_df.empty:
                    all_trade_data.append(trade_df)

        if all_trade_data:
            exports = [df for df in all_trade_data if 'export_value_usd' in df.columns]
            imports = [df for df in all_trade_data if 'import_value_usd' in df.columns]

            combined_exports = exports[0]
            for exp_df in exports[1:]:
                combined_exports = pd.merge(combined_exports, exp_df, on=['Country', 'Year'], how='outer', suffixes=('', '_dup'))
                combined_exports['export_value_usd'] = combined_exports['export_value_usd'].fillna(combined_exports['export_value_usd_dup'])
                combined_exports = combined_exports.drop(columns=['export_value_usd_dup'])

            combined_imports = imports[0]
            for imp_df in imports[1:]:
                combined_imports = pd.merge(combined_imports, imp_df, on=['Country', 'Year'], how='outer', suffixes=('', '_dup'))
                combined_imports['import_value_usd'] = combined_imports['import_value_usd'].fillna(combined_imports['import_value_usd_dup'])
                combined_imports = combined_imports.drop(columns=['import_value_usd_dup'])

            return pd.merge(combined_exports, combined_imports, on=['Country', 'Year'], how='outer')

        return pd.DataFrame()

    def clean_world_bank_format(self, file_path: str, dataset_name: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df = self.standardize_country_names(df, 'Country Name')
            df = df[df['Country Name'].isin(TARGET_COUNTRIES)].copy()

            year_columns = [col for col in df.columns if '[YR' in col]
            id_vars = ['Country Name', 'Series Name', 'Series Code']

            df_melted = pd.melt(
                df, id_vars=id_vars,
                value_vars=year_columns,
                var_name='Year_Raw',
                value_name='Value'
            )
            df_melted['Year'] = df_melted['Year_Raw'].str.extract(r'(\d{4})').astype(int)
            df_melted['Value'] = pd.to_numeric(df_melted['Value'].replace('..', np.nan), errors='coerce')
            df_melted = df_melted.dropna(subset=['Value'])

            df_pivot = df_melted.pivot_table(
                index=['Country Name', 'Year'],
                columns='Series Name',
                values='Value',
                aggfunc='first'
            ).reset_index()
            df_pivot.rename(columns={'Country Name': 'Country'}, inplace=True)

            return df_pivot
        except Exception:
            return pd.DataFrame()

    def clean_crop_livestock_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(os.path.join(RAW_DIR, 'crop_and_livestock.csv'))
            df = self.standardize_country_names(df, 'Area')
            df = df[df['Area'].isin(TARGET_COUNTRIES)].copy()
            df = df[df['Element'].isin(['Production', 'Yield', 'Area harvested'])].copy()
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df = df.dropna(subset=['Value'])

            agg_df = df.groupby(['Area', 'Year', 'Element'], as_index=False).agg({'Value': ['sum', 'mean', 'count']})
            agg_df.columns = ['Country', 'Year', 'Element', 'Total_Value', 'Avg_Value', 'Item_Count']

            pivot_df = agg_df.pivot_table(
                index=['Country', 'Year'],
                columns='Element',
                values=['Total_Value', 'Item_Count', 'Avg_Value'],
                fill_value=0
            ).reset_index()

            new_columns = []
            for col in pivot_df.columns:
                if isinstance(col, tuple):
                    if col[1] == '':
                        new_columns.append(col[0])
                    else:
                        element = col[1].lower().replace(' ', '_').replace('-', '_')
                        metric = col[0].lower().replace('_value', '')
                        new_columns.append(f"agr_{element}_{metric}")
                else:
                    new_columns.append(str(col))
            pivot_df.columns = new_columns
            return pivot_df
        except Exception:
            return pd.DataFrame()

    def clean_population_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(os.path.join(RAW_DIR, 'population_and_demographics.csv'))
            df = self.standardize_country_names(df, 'Area')
            df = df[df['Area'].isin(TARGET_COUNTRIES)].copy()
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce') * 1000
            df = df.dropna(subset=['Value'])

            df_pivot = df.pivot_table(
                index=['Area', 'Year'],
                columns='Element',
                values='Value',
                aggfunc='first'
            ).reset_index()
            df_pivot.rename(columns={'Area': 'Country'}, inplace=True)

            df_pivot.columns = [
                'Country' if col == 'Country' else
                'Year' if col == 'Year' else
                f"pop_{col.lower().replace(' ', '_').replace('-', '_')}"
                for col in df_pivot.columns
            ]
            return df_pivot
        except Exception:
            return pd.DataFrame()

    def clean_disaster_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(os.path.join(RAW_DIR, 'disasters.csv'))
            df = self.standardize_country_names(df, 'Country')
            df = df[df['Country'].isin(TARGET_COUNTRIES)].copy()

            numeric_cols = ['Total Deaths', 'No. Injured', 'No. Affected', 'Total Affected', 
                            "Total Damage ('000 US$)", "AID Contribution ('000 US$)"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=numeric_cols, how='all')

            disaster_agg = df.groupby(['Country', 'Start Year'], as_index=False).agg({
                'DisNo.': 'count',
                'Total Deaths': 'sum',
                'No. Injured': 'sum',
                'No. Affected': 'sum',
                'Total Affected': 'sum',
                "Total Damage ('000 US$)": 'sum',
                "AID Contribution ('000 US$)": 'sum'
            })
            disaster_agg.columns = ['Country', 'Year', 'disaster_count', 'disaster_deaths', 
                                    'disaster_injured', 'disaster_affected', 'disaster_total_affected',
                                    'disaster_damage_usd_thousands', 'disaster_aid_usd_thousands']
            return disaster_agg
        except Exception:
            return pd.DataFrame()

    def integrate_all_data(self) -> pd.DataFrame:

        master_df = self.country_year_framework.copy()
        datasets_to_process = [
            (self.process_all_trade_data(), "Trade"),
            (self.clean_world_bank_format(os.path.join(RAW_DIR, 'Core_economic_indicators.csv'), "Core Economic"), "Core Economic"),
            (self.clean_world_bank_format(os.path.join(RAW_DIR, 'Employment_Unemployment.csv'), "Employment"), "Employment"),
            (self.clean_world_bank_format(os.path.join(RAW_DIR, 'Resiliance.csv'), "Resilience"), "Resilience"),
            (self.clean_world_bank_format(os.path.join(RAW_DIR, 'Social_and_welfare.csv'), "Social Welfare"), "Social Welfare"),
            (self.clean_crop_livestock_data(), "Agriculture"),
            (self.clean_population_data(), "Population"),
            (self.clean_disaster_data(), "Disasters")
        ]
        for dataset_df, _ in datasets_to_process:
            if not dataset_df.empty and 'Country' in dataset_df.columns and 'Year' in dataset_df.columns:
                master_df = pd.merge(master_df, dataset_df, on=['Country', 'Year'], how='left')

        master_df = self.handle_missing_values(master_df)
        self.generate_data_quality_report(master_df)
        return master_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
        zero_fill_cols = [col for col in df.columns if 'disaster_' in col or ('agr_' in col and ('count' in col or 'total' in col))]
        df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        interpolate_cols = [col for col in numeric_cols if col != 'Year' and col not in zero_fill_cols]

        def interpolate_group(group):
            group[interpolate_cols] = group[interpolate_cols].interpolate(method='linear', limit_direction='both')
            return group
        df = df.groupby('Country').apply(interpolate_group).reset_index(drop=True)

        for col in interpolate_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        return df

    def generate_data_quality_report(self, df: pd.DataFrame) -> None:
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'countries_covered': df['Country'].nunique(),
            'years_covered': df['Year'].nunique(),
            'missing_data_by_column': df.isnull().sum().to_dict()
        }
        pd.DataFrame([report]).to_csv(os.path.join(PROCESSED_DIR, 'data_quality_report.csv'), index=False)

        missing_summary = df.isnull().sum().reset_index()
        missing_summary.columns = ['Column', 'Missing_Count']
        missing_summary['Missing_Percentage'] = (missing_summary['Missing_Count'] / len(df)) * 100
        missing_summary.sort_values('Missing_Percentage', ascending=False).to_csv(
            os.path.join(PROCESSED_DIR, 'missing_data_summary.csv'), index=False
        )

    def save_processed_data(self, df: pd.DataFrame) -> None:
        df.to_csv(os.path.join(PROCESSED_DIR, 'integrated_tren_dataset.csv'), index=False)

    def run_full_pipeline(self) -> pd.DataFrame:
        integrated_df = self.integrate_all_data()
        self.save_processed_data(integrated_df)
        return integrated_df


def main():
    cleaner = TRENDataCleaner()
    integrated_data = cleaner.run_full_pipeline()

    print(f"\nPipeline Summary:")
    print(f"- Total records: {len(integrated_data):,}")
    print(f"- Total columns: {len(integrated_data.columns):,}")
    print(f"- Countries: {integrated_data['Country'].nunique()}")
    print(f"- Years: {integrated_data['Year'].min()} - {integrated_data['Year'].max()}")
    print(f"\nFiles saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()