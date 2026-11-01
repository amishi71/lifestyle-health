import pandas as pd
import numpy as np

# Google Form URL for reference:
# https://docs.google.com/forms/d/e/1FAIpQLSd8eotYu9Zw3FNI-1JronRVy7cmfMl8SYo_OGJWS33JKXSbvw/viewform

def load_data():
    # Load your survey CSV file
    survey_df = pd.read_csv("Lifestyle and Health Survey (Responses) - Form Responses 1.csv")
    
    # Load processed NHANES dataset
    nhanes_df = pd.read_csv("data/processed/nhanes_clean.csv")
    
    return survey_df, nhanes_df

def print_comparisons(survey_df, nhanes_df):
    print("=== SUMMARY COMPARISON ===")
    print(f"Survey Total Responses: {len(survey_df)}")
    print(f"NHANES Total Sample Size: {len(nhanes_df)}")
    print("-" * 50)

    # 1. Feature Engineering: Calculate BMI in survey data if Height and Weight exist
    if 'Weight (kg)' in survey_df.columns and 'Height (cm)' in survey_df.columns:
        height_m = survey_df['Height (cm)'] / 100
        survey_df['bmi'] = survey_df['Weight (kg)'] / (height_m ** 2)

    # 2. Feature Engineering: Convert sitting time (hours) to sedentary minutes to align with NHANES
    if 'Sitting time per day (hours)' in survey_df.columns:
        survey_df['sedentary_min'] = survey_df['Sitting time per day (hours)'] * 60

    # Direct mapping between your survey headers and NHANES variable names
    mapping = {
        'Age (years)': 'age',
        'Average sleep per night (hours)': 'sleep_hrs',
        'sedentary_min': 'sedentary_min',
        'bmi': 'bmi'
    }

    # Print mean and std comparison for shared variables
    for survey_col, nhanes_col in mapping.items():
        if survey_col in survey_df.columns and nhanes_col in nhanes_df.columns:
            s_mean = survey_df[survey_col].mean()
            s_std = survey_df[survey_col].std()
            n_mean = nhanes_df[nhanes_col].mean()
            n_std = nhanes_df[nhanes_col].std()

            print(f"\nMetric: {survey_col} vs NHANES ({nhanes_col})")
            print(f"  Survey Mean : {s_mean:.2f} (Std: {s_std:.2f})")
            print(f"  NHANES Mean : {n_mean:.2f} (Std: {n_std:.2f})")

if __name__ == "__main__":
    survey_df, nhanes_df = load_data()
    print_comparisons(survey_df, nhanes_df)
