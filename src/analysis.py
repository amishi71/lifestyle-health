"""Correlation -> significance -> adjusted regression -> VIF -> explainable ML -> personas."""
import joblib, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, shap

pd.set_option("display.width", 200)          # show full tables in the terminal (no "...")

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
LIFE = ["sleep_hrs", "activity_domains", "sedentary_min", "kcal", "sugar_g", "sodium_mg", "fiber_g", "fat_g"]
# FIX: kcal had VIF > 10 (it overlaps with fat, sugar and sodium), which flipped signs in the
# adjusted regression. It is kept for Spearman, ML and personas, but left out of the regression/VIF.
ADJ = [c for c in LIFE if c != "kcal"]
HEALTH = ["sbp", "dbp", "bmi", "waist_cm", "hba1c", "hdl", "tchol", "phq9"]
MODEL_TARGETS = ["sbp", "bmi", "hba1c", "phq9"]
df = pd.read_csv("data/processed/nhanes_clean.csv").dropna(subset=LIFE + ["age", "sex"])
for c in ["kcal", "sugar_g", "sodium_mg", "fiber_g", "fat_g"]:      # trim extreme dietary outliers (top 0.5%)
    df = df[df[c] <= df[c].quantile(0.995)]
print("Analytic sample:", len(df))

# 1) Spearman + Benjamini-Hochberg FDR
rows = []
for l in LIFE:
    for h in HEALTH:
        s = df[[l, h]].dropna(); r, p = spearmanr(s[l], s[h]); rows.append((l, h, r, p, len(s)))
cor = pd.DataFrame(rows, columns=["lifestyle", "health", "rho", "p", "n"])
cor["p_fdr"] = multipletests(cor.p, method="fdr_bh")[1]
cor.to_csv(OUT / "spearman_results.csv", index=False)
plt.figure(figsize=(9, 6))
sns.heatmap(cor.pivot(index="lifestyle", columns="health", values="rho")[HEALTH], annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Spearman correlation: lifestyle vs health indicators"); plt.tight_layout()
plt.savefig(OUT / "heatmap_spearman.png", dpi=150); plt.close()

# 2) Adjusted OLS (standardised lifestyle, controlling for age and sex) -- uses ADJ (no kcal)
res = []
for h in HEALTH:
    s = df.dropna(subset=[h]); X = s[ADJ].apply(lambda c: (c - c.mean()) / c.std())
    X[["age", "sex"]] = s[["age", "sex"]]; m = sm.OLS(s[h], sm.add_constant(X)).fit(); ci = m.conf_int()
    for l in ADJ:
        res.append((h, l, m.params[l], ci.loc[l, 0], ci.loc[l, 1], m.pvalues[l], m.rsquared))
ols = pd.DataFrame(res, columns=["health", "lifestyle", "beta_per_SD", "ci_low", "ci_high", "p", "model_R2"])
ols["p_fdr"] = multipletests(ols.p, method="fdr_bh")[1]
ols.to_csv(OUT / "adjusted_regression.csv", index=False)

# 3) Multicollinearity -- check the same variables used in the regression
Xv = sm.add_constant(df[ADJ + ["age", "sex"]])
pd.DataFrame({"feature": Xv.columns[1:], "VIF": [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])]}
             ).to_csv(OUT / "vif.csv", index=False)

# 4) Explainable ML (gradient boosting + SHAP)
FEATS = LIFE + ["age", "sex"]; models, perf = {}, []
for t in MODEL_TARGETS:
    s = df.dropna(subset=[t]); Xtr, Xte, ytr, yte = train_test_split(s[FEATS], s[t], test_size=0.2, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42).fit(Xtr, ytr)
    perf.append((t, gb.score(Xte, yte), cross_val_score(gb, s[FEATS], s[t], cv=5, scoring="r2").mean(), len(s)))
    models[t] = gb
    sv = shap.TreeExplainer(gb).shap_values(Xte.iloc[:500])
    shap.summary_plot(sv, Xte.iloc[:500], show=False); plt.title(f"SHAP: {t}"); plt.tight_layout()
    plt.savefig(OUT / f"shap_{t}.png", dpi=150, bbox_inches="tight"); plt.close()
pd.DataFrame(perf, columns=["target", "test_R2", "cv_R2", "n"]).to_csv(OUT / "model_performance.csv", index=False)
joblib.dump({"models": models, "features": FEATS, "data": df[FEATS + MODEL_TARGETS]}, OUT / "models.joblib")

# 5) Lifestyle personas (k chosen by silhouette)
Z = StandardScaler().fit_transform(df[LIFE])
sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z)) for k in range(3, 7)}
pd.Series(sil, name="silhouette").rename_axis("k").to_csv(OUT / "silhouette_scores.csv")
k = max(sil, key=sil.get); df["persona"] = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
# FIX: also show average age and % female per persona, to help name and explain them
prof = df.groupby("persona")[["age", "sex"] + LIFE + HEALTH].mean().round(2)
prof = prof.rename(columns={"sex": "pct_female"}); prof["pct_female"] = (prof["pct_female"] * 100).round(1)
prof["size"] = df.persona.value_counts().sort_index()
prof.to_csv(OUT / "persona_profiles.csv")
zp = (df.groupby("persona")[LIFE].mean() - df[LIFE].mean()) / df[LIFE].std()
plt.figure(figsize=(9, 4)); sns.heatmap(zp, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title(f"Lifestyle personas (k={k}, z-scores vs population)"); plt.tight_layout(); plt.savefig(OUT / "personas.png", dpi=150)
plt.close()
print("Silhouette scores:", {kk: round(v, 3) for kk, v in sil.items()})
print("Done. Best k =", k); print(pd.read_csv(OUT / "model_performance.csv"))
