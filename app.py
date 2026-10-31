import joblib, pandas as pd, streamlit as st
from pathlib import Path
st.set_page_config(page_title="Lifestyle & Health", layout="wide")
art = joblib.load("outputs/models.joblib"); data = art["data"]; F = art["features"]
LABEL = {"sbp": "Systolic BP (mmHg)", "bmi": "BMI", "hba1c": "HbA1c (%)", "phq9": "PHQ-9 depression score"}
st.title("Lifestyle & Health: what-if explorer")
st.caption("Built on CDC NHANES 2017-2018. Associations, not causation; not medical advice.")
t1, t2 = st.tabs(["What-if simulator", "Findings"])
with t1:
    c1, c2 = st.columns(2); v = {}
    with c1:
        v["sleep_hrs"] = st.slider("Sleep (hrs/night)", 3.0, 12.0, 7.0, 0.5)
        v["activity_domains"] = st.slider("Active domains (work/travel/recreation, 0-5)", 0, 5, 1)
        v["sedentary_min"] = st.slider("Sedentary minutes/day", 0, 1000, 360, 30)
        v["age"] = st.slider("Age", 18, 80, 30); v["sex"] = 1 if st.radio("Sex", ["Male", "Female"]) == "Female" else 0
    with c2:
        v["kcal"] = st.slider("Calories/day", 800, 4500, 2000, 100)
        v["sugar_g"] = st.slider("Sugar (g)", 0, 300, 90, 5)
        v["sodium_mg"] = st.slider("Sodium (mg)", 500, 7000, 3400, 100)
        v["fiber_g"] = st.slider("Fibre (g)", 0, 50, 16); v["fat_g"] = st.slider("Fat (g)", 10, 200, 80, 5)
    x = pd.DataFrame([v])[F]; base = data[F].median().to_frame().T.assign(age=v["age"], sex=v["sex"])[F]
    for col, (t, m) in zip(st.columns(4), art["models"].items()):
        p = float(m.predict(x)[0]); b = float(m.predict(base)[0]); pct = (data[t].dropna() < p).mean() * 100
        col.metric(LABEL[t], f"{p:.1f}", f"{p - b:+.2f} vs median lifestyle", delta_color="inverse")
        col.caption(f"~{pct:.0f}th percentile of NHANES adults")
with t2:
    out = Path("outputs")
    for img in ["heatmap_spearman.png", "personas.png"] + [f"shap_{t}.png" for t in art["models"]]:
        if (out / img).exists(): st.image(str(out / img))
    st.dataframe(pd.read_csv(out / "model_performance.csv"))
