"""Lifestyle & Health dashboard (NHANES 2017-18): what-if simulator, findings, personas, SHAP.

Run:  streamlit run app.py      (after: python src/analysis.py)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="Habitly", page_icon="💜", layout="wide")

# ----------------------------------------------------------------------------- style
PALETTE = {"purple": "#6D3FD6", "lav": "#EFE8FC", "pink": "#E98AAE", "orange": "#F2895B", "teal": "#4FBFAF",
           "blue": "#7E93E6", "yellow": "#F4B740", "good": "#2BA88F", "warn": "#F29A3E", "bad": "#E2577F",
           "muted": "#9A93AE", "ink": "#2A2140"}
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@500;700;800;900&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp div,
.stApp span:not([data-testid="stIconMaterial"]) { font-family: 'Nunito', sans-serif !important; }
.stApp { background: #F7F3FE; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1220px; }
h1, h2, h3, h4 { color: #2A2140; font-weight: 900 !important; }
.hero { background: linear-gradient(120deg, #7C4DDB 0%, #9B6CF0 60%, #B793F5 100%); color: #fff; border-radius: 28px;
        padding: 26px 30px; margin-bottom: 18px; display: flex; align-items: center; gap: 22px;
        box-shadow: 0 14px 34px rgba(109,63,214,.28); }
.hero h1 { color: #fff !important; font-size: 2.1rem; margin: 0 0 4px 0; line-height: 1.1; }
.hero p  { color: rgba(255,255,255,.9); margin: 0; font-size: 1.05rem; font-weight: 600; }
.chip { display: inline-block; background: rgba(255,255,255,.2); color: #fff; padding: 5px 13px; border-radius: 999px;
        font-size: .82rem; font-weight: 700; margin: 10px 6px 0 0; }
.card { background: #fff; border-radius: 24px; padding: 18px 20px; box-shadow: 0 6px 20px rgba(109,63,214,.08); height: 100%; }
.tile { border-radius: 24px; padding: 18px 20px; color: #fff; position: relative; overflow: hidden; margin-bottom: 14px;
        box-shadow: 0 8px 22px rgba(42,33,64,.12); }
.tile .lbl { font-size: .85rem; font-weight: 800; opacity: .92; }
.tile .val { font-size: 2.3rem; font-weight: 900; line-height: 1.05; margin: 4px 0 6px 0; }
.tile .unit { font-size: .95rem; font-weight: 700; opacity: .85; margin-left: 4px; }
.tile .face { position: absolute; right: 12px; top: 12px; }
.status { display: inline-block; background: #fff; padding: 4px 12px; border-radius: 999px; font-size: .8rem; font-weight: 800; }
.tile .sub { font-size: .85rem; font-weight: 700; margin-top: 10px; opacity: .95; }
.kpi-label { font-size: .82rem; color: #9A93AE; font-weight: 800; }
.kpi-value { font-size: 2.1rem; font-weight: 900; color: #2A2140; line-height: 1.1; margin: 4px 0 2px 0; }
.small { font-size: .85rem; color: #8A82A0; font-weight: 600; }
.section-title { font-size: 1.3rem; font-weight: 900; color: #2A2140; margin: 8px 0 2px 0; }
.persona-name { font-size: 1.2rem; font-weight: 900; margin: 6px 0 2px 0; }
.trait { font-size: .92rem; margin: 4px 0; font-weight: 600; }
.stTabs [role="tablist"] { gap: 6px; background: #fff; padding: 6px; border-radius: 999px; border: none;
        box-shadow: 0 4px 14px rgba(109,63,214,.08); width: fit-content; max-width: 100%; }
.stTabs [role="tab"] { border-radius: 999px; padding: 8px 18px; font-weight: 800; color: #6B6385;
        background: transparent; border: none; padding: 8px 18px !important; }
.stTabs [role="tab"] p { font-weight: 800; font-size: .95rem; }
.stTabs [role="tab"][aria-selected="true"] { background: #6D3FD6; color: #fff; }
.stTabs [role="tab"][aria-selected="true"] p { color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stButton > button { border-radius: 999px; font-weight: 800; border: 2px solid #E3D8FA; background: #fff; color: #6D3FD6; }
.stButton > button:hover { border-color: #6D3FD6; color: #fff; background: #6D3FD6; }
[data-testid="stExpander"] { background: #fff; border-radius: 20px !important; border: none !important;
        box-shadow: 0 4px 14px rgba(109,63,214,.07); }
[data-testid="stExpander"] details { border: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


def face(shape="blob", color="#F2895B", size=56, mood="happy", limbs=False):
    """Simple flat shape with a line-drawn face (SVG). Shapes: blob=circle, square, pentagon=hexagon,
    cloud=dome, drop=leaf. Moods: happy (dot eyes + smile), calm (closed eyes), sad (dot eyes + frown)."""
    body = {
        "blob": '<circle cx="50" cy="52" r="34"/>',
        "square": '<rect x="18" y="20" width="64" height="62" rx="12"/>',
        "pentagon": '<path d="M50 14 L84 33 L84 71 L50 90 L16 71 L16 33 Z" stroke-linejoin="round" stroke-width="8"/>',
        "cloud": '<path d="M14 78 A36 36 0 0 1 86 78 Z"/>',
        "drop": '<path d="M28 84 C14 60 30 22 72 16 C82 46 74 80 28 84 Z"/>',
    }[shape]
    ink = "#2A2140"
    eye_y = 64 if shape == "cloud" else 52
    if mood == "calm":
        eyes = (f'<path d="M36 {eye_y} q5 5 10 0 M54 {eye_y} q5 5 10 0" stroke="{ink}" stroke-width="3.2" '
                f'fill="none" stroke-linecap="round"/>')
    else:
        eyes = f'<circle cx="40" cy="{eye_y}" r="3.6" fill="{ink}"/><circle cx="60" cy="{eye_y}" r="3.6" fill="{ink}"/>'
    my = eye_y + 10
    mouth = {"happy": f'<path d="M44 {my} q6 6 12 0"', "calm": f'<path d="M46 {my} q4 3 8 0"',
             "sad": f'<path d="M44 {my + 4} q6 -6 12 0"'}[mood]
    mouth += f' stroke="{ink}" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
    extra = ""
    if limbs:  # little stick arms and legs, like a doodle
        extra = (f'<path d="M18 56 Q8 60 6 70 M82 56 Q92 60 94 70 M40 86 L38 97 M60 86 L62 97" stroke="{ink}" '
                 f'stroke-width="2.6" fill="none" stroke-linecap="round"/>'
                 f'<circle cx="6" cy="71" r="3" fill="{ink}"/><circle cx="94" cy="71" r="3" fill="{ink}"/>')
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
            f'{extra}<g fill="{color}" stroke="{color}">{body}</g>{eyes}{mouth}</svg>')


# ----------------------------------------------------------------------------- data
LIFE = ["sleep_hrs", "activity_domains", "sedentary_min", "kcal", "sugar_g", "sodium_mg", "fiber_g", "fat_g"]
NICE = {"sleep_hrs": "Sleep", "activity_domains": "Activity", "sedentary_min": "Sitting time", "kcal": "Calories",
        "sugar_g": "Sugar", "sodium_mg": "Sodium", "fiber_g": "Fibre", "fat_g": "Fat", "age": "Age", "sex": "Sex",
        "sbp": "Systolic BP", "dbp": "Diastolic BP", "bmi": "BMI", "waist_cm": "Waist", "hba1c": "HbA1c",
        "hdl": "HDL chol.", "tchol": "Total chol.", "phq9": "PHQ-9"}

# target -> (label, unit, decimals, health bands [(upper limit, label, colour)], axis max)
TARGETS = {
    "sbp":   ("Systolic BP", "mmHg", 0, [(120, "Normal", "good"), (130, "Elevated", "warn"), (999, "High", "bad")], 180),
    "bmi":   ("BMI", "kg/m²", 1, [(18.5, "Underweight", "warn"), (25, "Healthy", "good"), (30, "Overweight", "warn"),
                                   (999, "Obese", "bad")], 45),
    "hba1c": ("HbA1c", "%", 2, [(5.7, "Normal", "good"), (6.5, "Prediabetes", "warn"), (99, "Diabetes range", "bad")], 9),
    "phq9":  ("Depression (PHQ-9)", "/ 27", 1, [(5, "Minimal", "good"), (10, "Mild", "warn"), (99, "Moderate+", "bad")], 27),
}


@st.cache_resource
def load_bundle():
    return joblib.load(OUT / "models.joblib")


@st.cache_data
def load_csv(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


if not (OUT / "models.joblib").exists():
    st.error("No trained models found. Run `python src/analysis.py` first, then refresh this page.")
    st.stop()

bundle = load_bundle()
models, FEATS, data = bundle["models"], bundle["features"], bundle["data"]
med = data[LIFE].median()

# ----------------------------------------------------------------------------- hero
hero_faces = (face("square", "#F4B740", 70, "happy", True) + face("cloud", "#E98AAE", 70, "calm", True)
              + face("pentagon", "#4FBFAF", 70, "calm", True) + face("blob", "#7E93E6", 70, "happy", True))
st.markdown(
    f"""
<div class="hero">
  <div style="flex:1">
    <h1>Habitly</h1>
    <p>Correlation analysis of lifestyle parameters and their impact on general health indicators.</p>
    <span class="chip">CDC NHANES 2017–18</span><span class="chip">{len(data):,} adults</span>
    <span class="chip">Correlation · Regression · ML · Clustering</span>
  </div>
  <div style="display:flex; gap:6px; flex-wrap:nowrap; justify-content:flex-end">{hero_faces}</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_sim, tab_find, tab_pers, tab_shap, tab_about = st.tabs(
    ["Simulator", "Findings", "Personas", "Model insights", "About the project"])


def band_of(target, value):
    for limit, label, colour in TARGETS[target][3]:
        if value < limit:
            return label, PALETTE[colour]
    return TARGETS[target][3][-1][1], PALETTE[TARGETS[target][3][-1][2]]


def predict(row):
    X = pd.DataFrame([row])[FEATS]
    return {t: float(models[t].predict(X)[0]) for t in TARGETS if t in models}


# ----------------------------------------------------------------------------- simulator
PRESETS = {
    "Typical adult": {k: float(med[k]) for k in LIFE},
    "Healthy habits": {"sleep_hrs": 8.0, "activity_domains": 3.0, "sedentary_min": 180.0, "kcal": 2000.0,
                       "sugar_g": 50.0, "sodium_mg": 2300.0, "fiber_g": 30.0, "fat_g": 65.0},
    "Couch & fast food": {"sleep_hrs": 5.5, "activity_domains": 0.0, "sedentary_min": 600.0, "kcal": 2800.0,
                          "sugar_g": 180.0, "sodium_mg": 5000.0, "fiber_g": 8.0, "fat_g": 120.0},
}
SLIDERS = {  # key: (label, min, max, step, format)
    "sleep_hrs": ("Sleep (hours per night)", 3.0, 12.0, 0.5, "%.1f"),
    "activity_domains": ("Active areas of life (work, travel, sport: 0–5)", 0.0, 5.0, 1.0, "%.0f"),
    "sedentary_min": ("Sitting time (minutes per day)", 0.0, 960.0, 30.0, "%.0f"),
    "kcal": ("Calories (kcal per day)", 800.0, 4500.0, 50.0, "%.0f"),
    "sugar_g": ("Sugar (g per day)", 10.0, 300.0, 5.0, "%.0f"),
    "sodium_mg": ("Sodium (mg per day)", 800.0, 8000.0, 100.0, "%.0f"),
    "fiber_g": ("Fibre (g per day)", 2.0, 50.0, 1.0, "%.0f"),
    "fat_g": ("Fat (g per day)", 20.0, 200.0, 5.0, "%.0f"),
}


def clamp(k, v):
    lo, hi = SLIDERS[k][1], SLIDERS[k][2]
    return float(min(max(v, lo), hi))


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = clamp(k, v)


for k in LIFE:  # initialise once
    if k not in st.session_state:
        st.session_state[k] = clamp(k, med[k])

with tab_sim:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown('<div class="section-title">Build a profile</div>', unsafe_allow_html=True)
        st.caption("Pick a starting point, then fine-tune the sliders.")
        b1, b2, b3 = st.columns(3)
        b1.button("Typical", on_click=apply_preset, args=("Typical adult",), use_container_width=True)
        b2.button("Healthy", on_click=apply_preset, args=("Healthy habits",), use_container_width=True)
        b3.button("Less healthy", on_click=apply_preset, args=("Couch & fast food",), use_container_width=True)

        a1, a2 = st.columns(2)
        age = a1.slider("Age", 18, 80, 35)
        sex_label = a2.radio("Sex", ["Male", "Female"], horizontal=True)
        sex = 1 if sex_label == "Female" else 0

        with st.expander("Sleep & activity", expanded=True):
            for k in ["sleep_hrs", "activity_domains", "sedentary_min"]:
                lab, lo, hi, step, fmt = SLIDERS[k]
                st.slider(lab, lo, hi, step=step, format=fmt, key=k)
        with st.expander("Diet (one typical day)", expanded=True):
            for k in ["kcal", "sugar_g", "sodium_mg", "fiber_g", "fat_g"]:
                lab, lo, hi, step, fmt = SLIDERS[k]
                st.slider(lab, lo, hi, step=step, format=fmt, key=k)

    row = {k: st.session_state[k] for k in LIFE}
    row.update({"age": age, "sex": sex})
    base = {k: float(med[k]) for k in LIFE}
    base.update({"age": age, "sex": sex})
    pred, ref = predict(row), predict(base)

    with right:
        st.markdown('<div class="section-title">Predicted health indicators</div>', unsafe_allow_html=True)
        st.caption(f"Compared with a {age}-year-old {sex_label.lower()} with a typical (median) lifestyle.")
        cols = st.columns(2)
        for i, (t, (label, unit, dec, _, axis_max)) in enumerate(TARGETS.items()):
            if t not in pred:
                continue
            v, d = pred[t], pred[t] - ref[t]
            band, colour = band_of(t, v)
            pct = int(round((data[t].dropna() < v).mean() * 100))
            fill = max(3, min(100, v / axis_max * 100))
            arrow = "▲" if d > 0 else "▼"
            dcol = None  # direction shown by arrow
            if abs(d) < 10 ** (-dec) / 2:
                arrow, dcol = "●", PALETTE["muted"]
            look = {"sbp": ("#E98AAE", "blob"), "bmi": ("#F2895B", "square"),
                    "hba1c": ("#4FBFAF", "drop"), "phq9": ("#7E93E6", "cloud")}[t]
            mood = {"good": "happy", "warn": "calm", "bad": "sad"}[[c for _, l, c in TARGETS[t][3] if l == band][0]]
            icon = "●"
            cols[i % 2].markdown(
                f"""
<div class="tile" style="background:{look[0]}">
  <div class="face">{face(look[1], "#ffffff", 52, mood).replace('fill="#ffffff" stroke="#ffffff"', 'fill="rgba(255,255,255,.35)" stroke="rgba(255,255,255,.0)"')}</div>
  <div class="lbl">{label}</div>
  <div class="val">{v:.{dec}f}<span class="unit">{unit}</span></div>
  <span class="status" style="color:{look[0]}">{icon} {band}</span>
  <div class="sub">{arrow} {d:+.{dec}f} vs typical lifestyle · ~{pct}th percentile</div>
</div>""",
                unsafe_allow_html=True,
            )

        # Which habit matters most for this person?
        st.markdown('<div class="section-title">Which habit moves the needle?</div>', unsafe_allow_html=True)
        tsel = st.selectbox("Indicator", list(TARGETS), format_func=lambda t: TARGETS[t][0], label_visibility="collapsed")
        healthy = PRESETS["Healthy habits"]
        effects = []
        for k in LIFE:
            r2 = dict(row); r2[k] = healthy[k]
            effects.append((NICE[k], predict(r2)[tsel] - pred[tsel]))
        eff = pd.DataFrame(effects, columns=["habit", "change"]).sort_values("change")
        fig = go.Figure(go.Bar(
            x=eff["change"], y=eff["habit"], orientation="h",
            marker_color=[PALETTE["teal"] if c < 0 else PALETTE["pink"] for c in eff["change"]], marker_line_width=0,
            text=[f"{c:+.2f}" for c in eff["change"]], textposition="outside", cliponaxis=False))
        fig.update_layout(height=300, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", xaxis_title=f"Change in {TARGETS[tsel][0]} ({TARGETS[tsel][1]})",
                          xaxis=dict(zeroline=True, zerolinecolor="#94A3B8", gridcolor="rgba(148,163,184,.2)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Each bar: what happens if only that one habit is switched to the 'Healthy' preset value. "
                   "Green = improves the indicator. Changes are small because lifestyle explains only part of health.")

# ----------------------------------------------------------------------------- findings
with tab_find:
    cor, ols, perf, vif = (load_csv(n) for n in
                           ["spearman_results.csv", "adjusted_regression.csv", "model_performance.csv", "vif.csv"])

    if cor is not None:
        sig = cor[cor.p_fdr < 0.05].copy()
        top = sig.reindex(sig.rho.abs().sort_values(ascending=False).index).head(3)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="card"><div class="kpi-label">Pairs tested</div>'
                    f'<div class="kpi-value">{len(cor)}</div><div class="small">lifestyle × health</div></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="card"><div class="kpi-label">Significant</div>'
                    f'<div class="kpi-value" style="color:{PALETTE["purple"]}">{len(sig)}</div>'
                    f'<div class="small">after FDR correction</div></div>', unsafe_allow_html=True)
        if len(top):
            t0 = top.iloc[0]
            k3.markdown(f'<div class="card"><div class="kpi-label">Strongest link</div>'
                        f'<div class="kpi-value" style="font-size:1.35rem">{NICE[t0.lifestyle]} → {NICE[t0.health]}</div>'
                        f'<div class="small">ρ = {t0.rho:.3f}</div></div>', unsafe_allow_html=True)
        if perf is not None:
            best = perf.sort_values("cv_R2").iloc[-1]
            k4.markdown(f'<div class="card"><div class="kpi-label">Best model</div>'
                        f'<div class="kpi-value" style="font-size:1.35rem">{NICE.get(best.target, best.target)}</div>'
                        f'<div class="small">cross-validated R² = {best.cv_R2:.2f}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-title">Correlation map</div>', unsafe_allow_html=True)
        st.caption("Spearman ρ between each habit and each health indicator. ✱ = significant after "
                   "Benjamini-Hochberg FDR correction. Teal = negative link, pink = positive link.")
        piv = cor.pivot(index="lifestyle", columns="health", values="rho")
        pfd = cor.pivot(index="lifestyle", columns="health", values="p_fdr")
        rows_ = [r for r in LIFE if r in piv.index]
        cols_ = [c for c in ["sbp", "dbp", "bmi", "waist_cm", "hba1c", "hdl", "tchol", "phq9"] if c in piv.columns]
        piv, pfd = piv.loc[rows_, cols_], pfd.loc[rows_, cols_]
        text = [[f"{piv.iat[i, j]:.2f}{'✱' if pfd.iat[i, j] < .05 else ''}" for j in range(piv.shape[1])]
                for i in range(piv.shape[0])]
        lim = max(0.2, float(piv.abs().max().max()))
        hm = go.Figure(go.Heatmap(
            z=piv.values, x=[NICE[c] for c in cols_], y=[NICE[r] for r in rows_], text=text,
            texttemplate="%{text}", colorscale=[[0, "#4FBFAF"], [0.5, "#FFFFFF"], [1, "#E98AAE"]], zmin=-lim, zmax=lim,
            colorbar=dict(title="ρ", thickness=12), hovertemplate="%{y} → %{x}<br>ρ = %{z:.3f}<extra></extra>"))
        hm.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False})

    if ols is not None:
        st.markdown('<div class="section-title">Effect sizes (adjusted for age and sex)</div>', unsafe_allow_html=True)
        st.caption("Change in the health indicator for a 1-standard-deviation increase in each habit, with 95% "
                   "confidence interval. Bars crossing zero are not significant.")
        hsel = st.selectbox("Health indicator", list(dict.fromkeys(ols.health)),
                            format_func=lambda h: NICE.get(h, h), key="forest")
        s = ols[ols.health == hsel].sort_values("beta_per_SD")
        colours = [PALETTE["muted"] if p >= .05 else (PALETTE["teal"] if b < 0 else PALETTE["pink"])
                   for b, p in zip(s.beta_per_SD, s.p_fdr)]
        fp = go.Figure(go.Scatter(
            x=s.beta_per_SD, y=[NICE[l] for l in s.lifestyle], mode="markers",
            marker=dict(size=13, color=colours, line=dict(width=1, color="white")),
            error_x=dict(type="data", symmetric=False, array=s.ci_high - s.beta_per_SD,
                         arrayminus=s.beta_per_SD - s.ci_low, color="#94A3B8", thickness=2),
            hovertemplate="%{y}: %{x:.3f} per SD<extra></extra>"))
        fp.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
        fp.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                         paper_bgcolor="rgba(0,0,0,0)", xaxis_title=f"Change in {NICE.get(hsel, hsel)} per 1 SD",
                         xaxis=dict(gridcolor="rgba(148,163,184,.2)"))
        st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">Headline takeaways</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.markdown('<div class="card">' + face("blob", "#4FBFAF", 50) + '<div style="height:4px"></div><div class="persona-name">Fibre is protective</div>'
                '<div class="trait">Linked with lower BMI, waist, blood pressure and depression score.</div></div>',
                unsafe_allow_html=True)
    t2.markdown('<div class="card">' + face("square", "#F2895B", 50, "calm") + '<div style="height:4px"></div><div class="persona-name">Sitting adds up</div>'
                '<div class="trait">More sitting time goes with higher BMI, bigger waist and higher PHQ-9.</div></div>',
                unsafe_allow_html=True)
    t3.markdown('<div class="card">' + face("cloud", "#7E93E6", 50) + '<div style="height:4px"></div><div class="persona-name">Sleep is U-shaped</div>'
                '<div class="trait">Both very short and very long sleep go with more depressive symptoms.</div></div>',
                unsafe_allow_html=True)

    with st.expander("Model performance and multicollinearity check"):
        c1, c2 = st.columns(2)
        if perf is not None:
            c1.markdown("**Gradient boosting performance**")
            c1.dataframe(perf.rename(columns={"target": "Target", "test_R2": "Test R²", "cv_R2": "CV R²", "n": "People"})
                         .round(3), hide_index=True, use_container_width=True)
        if vif is not None:
            c2.markdown("**VIF (all below 5 = no problematic overlap)**")
            c2.dataframe(vif.round(2), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- personas
with tab_pers:
    prof = load_csv("persona_profiles.csv")
    if prof is None:
        st.info("Run `python src/analysis.py` to create the personas.")
    else:
        if "persona" in prof.columns:
            prof = prof.set_index("persona")
        # name personas by their characteristics (robust to cluster numbering)
        order = prof["kcal"].sort_values().index.tolist()
        names = {}
        if len(order) == 3:
            names = {order[0]: (("cloud", "sad"), "Light Eaters, Less Active", "Eat the least and move the least"),
                     order[1]: (("square", "calm"), "Sedentary Average Eaters", "Typical diet, but the most sitting"),
                     order[2]: (("blob", "happy"), "Active Big Eaters", "Eat the most and stay the most active")}
        st.markdown('<div class="section-title">Three lifestyle personas</div>', unsafe_allow_html=True)
        st.caption("Found with K-Means clustering on the 8 lifestyle variables (number of groups chosen by silhouette score).")
        pcols = st.columns(len(prof))
        z_mean, z_sd = data[LIFE].mean(), data[LIFE].std()
        radar = go.Figure()
        radar_cols = ["#F2895B", "#7E93E6", "#4FBFAF", "#E98AAE", "#F4B740", "#6D3FD6"]
        for i, (pid, r) in enumerate(prof.iterrows()):
            look, name, tag = names.get(pid, (("drop", "calm"), f"Persona {pid}", ""))
            extra = ""
            if "age" in r and "pct_female" in r:
                extra = f'<div class="small">Avg age {r["age"]:.0f} · {r["pct_female"]:.0f}% female</div>'
            pcols[i].markdown(
                f"""
<div class="card" style="padding:0; overflow:hidden">
<div style="background:{radar_cols[i % 6]}; padding:18px 20px; color:#fff">
{face(look[0], "#ffffff", 58, look[1]).replace('fill="#ffffff" stroke="#ffffff"', 'fill="rgba(255,255,255,.4)" stroke="rgba(255,255,255,0)"')}
<div class="persona-name" style="color:#fff">{name}</div>
<div style="font-weight:700; opacity:.92; font-size:.88rem">{int(r['size']):,} people · {tag}</div>
</div>
<div style="padding:14px 20px 18px 20px">
{extra}
<div class="kpi-label" style="margin-top:8px">LIFESTYLE</div>
<div class="trait">{r['kcal']:,.0f} kcal · {r['fiber_g']:.0f} g fibre</div>
<div class="trait">{r['activity_domains']:.2f} active areas · {r['sedentary_min']:.0f} min sitting</div>
<div class="trait">{r['sleep_hrs']:.1f} h sleep</div>
<div class="kpi-label" style="margin-top:10px">HEALTH</div>
<div class="trait">HbA1c {r['hba1c']:.2f}% · SBP {r['sbp']:.0f} mmHg</div>
<div class="trait">BMI {r['bmi']:.1f} · PHQ-9 {r['phq9']:.2f}</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
            z = ((r[LIFE] - z_mean) / z_sd).tolist()
            radar.add_trace(go.Scatterpolar(r=z + z[:1], theta=[NICE[k] for k in LIFE] + [NICE[LIFE[0]]],
                                            name=name, line=dict(color=radar_cols[i % 6], width=2.5), fill="toself",
                                            opacity=.55))
        st.write("")
        st.markdown('<div class="section-title">Lifestyle fingerprint</div>', unsafe_allow_html=True)
        st.caption("How far each persona sits above (+) or below (−) the population average, in standard deviations.")
        radar.update_layout(height=460, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)",
                            polar=dict(radialaxis=dict(range=[-1.5, 1.5], gridcolor="rgba(148,163,184,.3)"),
                                       angularaxis=dict(gridcolor="rgba(148,163,184,.3)"), bgcolor="rgba(0,0,0,0)"),
                            legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})
        sil = load_csv("silhouette_scores.csv")
        if sil is not None:
            st.caption(f"Silhouette score for the chosen k: {sil.silhouette.max():.3f}. Low values mean the personas "
                       "overlap, so they are soft groupings rather than sharply separate types.")

# ----------------------------------------------------------------------------- SHAP
with tab_shap:
    st.markdown('<div class="section-title">What drives each prediction?</div>', unsafe_allow_html=True)
    st.caption("SHAP summary plots: each dot is one person. Right = pushes the prediction up, left = down. "
               "Red = high value of that feature, blue = low. For sex, blue = male and red = female.")
    notes = {
        "sbp": "Age dominates. High sodium pushes BP up by up to ~10 mmHg; low fibre pushes it up; women run lower.",
        "bmi": "Low fibre, more sitting and more fat push BMI up. High calories look 'protective': likely under-reporting.",
        "hba1c": "Age dominates. Short sleep and low activity raise HbA1c. Low sugar/kcal raising it is reverse causation.",
        "phq9": "Women score higher. Sleep is U-shaped: both very short and very long sleep raise the score.",
    }
    choice = st.radio("Indicator", list(TARGETS), format_func=lambda t: TARGETS[t][0], horizontal=True,
                      label_visibility="collapsed")
    c1, c2 = st.columns([1.6, 1], gap="large")
    img = OUT / f"shap_{choice}.png"
    if img.exists():
        c1.image(str(img), use_container_width=True)
    c2.markdown(f'<div class="card">{face("drop", "#F4B740", 50)}<div class="persona-name">How to read it</div>'
                f'<div class="trait">{notes[choice]}</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- about
with tab_about:
    a1, a2 = st.columns([1.3, 1], gap="large")
    a1.markdown("""
#### Why we picked this

Everyone hears "sleep more, sit less, eat more fibre", but we wanted to see how much
those habits actually show up in real health numbers, and which ones matter most.

#### The data

We used NHANES 2017–18, a health survey run by the US Centers for Disease Control (CDC).
It is one of the few free datasets where the *same* people answered lifestyle questions,
had a physical exam and gave a blood sample. We merged 10 of its files and kept adults
aged 18 and over with a usable one-day food diary, which left **4,658 people**.

Because the data is American, the results describe US adults. Diets and activity
patterns in India are different, so we would not apply these numbers here directly.

#### What went wrong (and how we fixed it)

The first version of the dataset *looked* fine, but a few numbers were off:

- Some diastolic blood pressures were 0. We traced this to how Python reads SAS files:
  zeros come in as tiny numbers like 0.000…05, so the code that was meant to remove them
  never matched. We converted them back to real zeros first.
- A few food diaries reported 0 or over 12,000 calories in a day, so we dropped diet
  values outside 500–5,000 kcal.
- When we ran the regression, total calories overlapped heavily with fat, sugar and salt
  (VIF above 10), which flipped some results the wrong way. We removed calories from that
  model and the results became sensible.
""")
    a2.markdown("""
#### How we analysed it

1. Spearman correlation for every habit–health pair, corrected for testing 64 pairs at once
2. Regression adjusted for age and sex, so older people's higher BP does not get blamed on habits
3. Gradient boosting models, explained with SHAP plots
4. K-Means clustering to group people into lifestyle personas

#### What to keep in mind

- This is a snapshot survey, so it shows **links, not cause and effect**.
- One day of food diary is a rough picture of someone's normal diet.
- Sleep and activity are self-reported.
- We did not apply NHANES survey weights; that would be the next step.
- Habits explain only a small part of health. Age and sex matter far more.
""")
    st.caption("Habitly · Data Analysis mini project · For learning only, not medical advice.")