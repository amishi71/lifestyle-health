"""Download NHANES 2017-2018 (CDC) and merge into one analytic table on SEQN."""
import requests, pandas as pd, numpy as np
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/processed/nhanes_clean.csv")
FILES = ["DEMO_J", "BPX_J", "BMX_J", "SLQ_J", "PAQ_J", "DR1TOT_J", "GHB_J", "HDL_J", "TCHOL_J", "DPQ_J"]
URLS = ["https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{f}.XPT",
        "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{f}.XPT"]

def fetch(name):
    p = RAW / f"{name}.XPT"
    if p.exists(): return p
    for u in URLS:
        r = requests.get(u.format(f=name), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.ok and r.content[:6] == b"HEADER":
            p.write_bytes(r.content); return p
    raise RuntimeError(f"Could not download {name}; save it manually into data/raw/")

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    t = {n: pd.read_sas(fetch(n), format="xport") for n in FILES}
    df = t["DEMO_J"][["SEQN", "RIAGENDR", "RIDAGEYR"]]
    for n in FILES[1:]:
        df = df.merge(t[n], on="SEQN", how="left")

    d = pd.DataFrame({"seqn": df.SEQN, "age": df.RIDAGEYR, "sex": df.RIAGENDR.map({1: 0, 2: 1})})  # 1 = female
    # ---- lifestyle ----
    d["sleep_hrs"] = (5 * df.SLD012 + 2 * df.SLD013) / 7           # weighted weekday/weekend
    paq = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]        # 1 = yes, 2 = no
    d["activity_domains"] = (df[paq] == 1).sum(axis=1).where(df[paq].notna().any(axis=1))
    d["sedentary_min"] = df.PAD680.where(df.PAD680 < 1440)
    ok = df.DR1DRSTZ == 1                                           # reliable dietary recall only
    for new, old in {"kcal": "DR1TKCAL", "sugar_g": "DR1TSUGR", "sodium_mg": "DR1TSODI",
                     "fiber_g": "DR1TFIBE", "fat_g": "DR1TTFAT"}.items():
        d[new] = df[old].where(ok)
    # ---- health indicators ----
    d["sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan).mean(axis=1)
    d["dbp"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1)
    d["bmi"], d["waist_cm"] = df.BMXBMI, df.BMXWAIST
    d["hba1c"], d["hdl"], d["tchol"] = df.LBXGH, df.LBDHDD, df.LBXTC
    ph = [f"DPQ0{i}0" for i in range(1, 10)]
    d["phq9"] = df[ph].where(df[ph] <= 3).sum(axis=1, min_count=9)   # 7/9 refused/unknown -> NaN
    d = d[d.age >= 18]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(d)} adults")

if __name__ == "__main__":
    main()
