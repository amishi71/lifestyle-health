# Lifestyle & Health: Correlation Analysis (NHANES 2017-18)
Correlation analysis of lifestyle parameters (sleep, activity, diet) and their impact on general health indicators
(blood pressure, BMI, waist, HbA1c, cholesterol, depression score), using real CDC NHANES data.

## Run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/build_dataset.py     # downloads 10 NHANES .XPT files, merges on SEQN
    python src/analysis.py          # stats, models, SHAP, personas -> outputs/
    streamlit run app.py            # what-if dashboard

## Pipeline
1. Merge NHANES files (demographics, BP, body measures, sleep, physical activity, 24h diet recall, HbA1c, lipids, PHQ-9)
2. Spearman correlations with Benjamini-Hochberg FDR correction
3. Age/sex-adjusted OLS (effect per 1 SD of each lifestyle factor, 95% CI)
4. VIF multicollinearity check
5. Gradient boosting + SHAP for sbp, bmi, hba1c, phq9
6. K-Means lifestyle personas (k by silhouette)
7. Streamlit what-if simulator

## Limitations (write these in your report)
Cross-sectional so no causation; one 24h diet recall is noisy; sleep and activity are self-reported;
NHANES uses survey weights (not applied here, mention as future scope); expect small R2 for lifestyle-only models.
