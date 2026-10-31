import pandas as pd

df = pd.read_csv("data/processed/nhanes_clean.csv")

print("ROWS, COLUMNS:", df.shape)
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES PER COLUMN:")
print(df.isna().sum())

print("\nSUMMARY (min, average, max):")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print(df.describe().T[["count", "mean", "min", "max"]].round(2))