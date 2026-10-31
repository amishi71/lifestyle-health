import pandas as pd
from pathlib import Path

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

out = Path("outputs")
print("FILES IN outputs/:")
for f in sorted(out.rglob("*")):
    if f.is_file():
        print("  ", f)

for f in sorted(out.rglob("*.csv")):
    print("\n" + "=" * 70)
    print(f.name)
    print("=" * 70)
    print(pd.read_csv(f).round(3).to_string())