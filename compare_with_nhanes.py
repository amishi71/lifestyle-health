import pandas as pd
import numpy as np

def load_data():
    # Load your survey CSV file
    survey_df = pd.read_csv("Lifestyle and Health Survey (Responses) - Form Responses 1.csv")
    
    # Load processed NHANES dataset
    nhanes_df = pd.read_csv("data/processed/nhanes_final.csv")  # Adjust path if different
    
    return survey_df, nhanes_df

def print_comparisons(survey_df, nhanes_df):
    print("=== SUMMARY COMPARISON ===")
    print(f"Survey Total Responses: {len(survey_df)}")
    print(f"NHANES Total Sample Size: {len(nhanes_df)}")
    print("-" * 40)

    # Example: Column mappings & metric comparisons
    # Map your survey column names to the NHANES dataset column names below:
    mapping = {
        # 'Survey Column Name': 'NHANES_Column_Name'
        # e.g., 'How many hours of sleep do you get?': 'sleep_hours',
        # 'Age': 'age'
    }

    for survey_col, nhanes_col in mapping.items():
        if survey_col in survey_df.columns and nhanes_col in nhanes_df.columns:
            print(f"\nMetric: {survey_col} vs NHANES ({nhanes_col})")
            print(f"  Survey Mean: {survey_df[survey_col].mean():.2f} (Std: {survey_df[survey_col].std():.2f})")
            print(f"  NHANES Mean: {nhanes_df[nhanes_col].mean():.2f} (Std: {nhanes_df[nhanes_col].std():.2f})")

if __name__ == "__main__":
    survey_df, nhanes_df = load_data()
    print("Survey Columns:", survey_df.columns.tolist())
    print("NHANES Columns:", nhanes_df.columns.tolist())
    print_comparisons(survey_df, nhanes_df)
