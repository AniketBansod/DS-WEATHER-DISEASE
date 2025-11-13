# Weather–Disease Prediction: Presenter Notes

These notes explain what each major block of code does and how to interpret the outputs, so you can present confidently from basics to advanced.

## 1) Data ingestion and setup

- Files: `notebooks/01_eda.ipynb` (analysis), `data/Weather-related disease prediction.csv` (raw data).
- Typical code: import libraries (pandas, numpy, matplotlib/seaborn, sklearn), set plotting style and random seed, and load the CSV into a DataFrame `df`.
- What to say: “We load the dataset, inspect schema with `df.info()`/`df.describe()`, and review missing values and unique counts to understand data quality.”

## 2) Target and feature groups

- Target: `prognosis` (multi-class).
- Features:
  - Weather numerics: Temperature (C), Humidity, Wind Speed (km/h) (names vary in raw data).
  - Demographic: Age, Gender.
  - Symptoms: many binary (0/1) indicators.
- Code pattern: detect weather columns robustly
  ```python
  possible_weather = ['Temperature (C)','Temperature_C','Temp','Humidity','Wind Speed (km/h)','WindSpeed']
  weather_cols = [c for c in possible_weather if c in df.columns]
  if not weather_cols:
      weather_cols = [c for c in df.columns if any(k in c.lower() for k in ['temp','humid','wind'])]
  weather_cols = list(dict.fromkeys(weather_cols))
  ```
- Why: column names can differ; substring fallback recovers likely weather features.

## 3) Class distribution and basic EDA

- Code: `df['prognosis'].value_counts()` → barplot; save to `outputs/figures/target_distribution.png`.
- Interpretation: “We have class imbalance; therefore we later use stratified splits and macro F1.”

## 4) Symptom frequency and correlation (clustermap)

- Code pattern:
  ```python
  symptom_cols = [...]  # all 0/1 symptom columns
  symptom_counts = df[symptom_cols].sum().sort_values(ascending=False)
  top_sym = symptom_counts.index[:40]
  sns.clustermap(df[top_sym].corr(), cmap='vlag', linewidths=.5, figsize=(10,10))
  plt.savefig('outputs/figures/symptom_clustermap.png', dpi=150)
  ```
- What it computes: Pearson correlation on binary equals the phi coefficient. Dendrogram groups symptoms that co-occur.
- How to read: red blocks near the diagonal = symptom bundles (co-occurrence); blue = negative association; white = near-independent.
- Caveats: rare symptoms yield unstable correlations; clustering is unsupervised.

## 5) Weather vs disease differences

- Boxplot code:
  ```python
  top6 = df['prognosis'].value_counts().head(6).index
  sns.boxplot(x='prognosis', y=weather_cols[0], data=df[df['prognosis'].isin(top6)])
  ```
- Statistical tests:
  - ANOVA compares means of a numeric feature across classes (assumes normality/variance homogeneity).
  - Kruskal–Wallis is non‑parametric, rank‑based; robust when assumptions fail.
- Interpretation of very small p-values: temperature (or selected feature) distributions differ significantly between diseases.

## 6) Symptom–disease association (Chi‑square)

- Code pattern:
  ```python
  from scipy.stats import chi2_contingency
  top5 = df['prognosis'].value_counts().head(5).index
  ct = pd.crosstab(df[df['prognosis'].isin(top5)]['prognosis'], df['high_fever'])
  chi2, p, dof, expected = chi2_contingency(ct)
  ```
- What it tests: H0 = independence between symptom (0/1) and prognosis (Top‑5). Small p ⇒ association.
- Report effect size: Cramer’s V (0–1) to quantify strength.

## 7) Train/test split and preprocessing

- Protocol: Stratified 80/20 split on `prognosis`.
- Preprocessing pipelines (recommended and used during tuning):
  - Numeric: `SimpleImputer(median) → StandardScaler()`.
  - Categorical: `SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')`.
  - Symptoms: passthrough (already 0/1).
- Feature engineering examples: `symptom_sum` (row-wise sum of symptoms), `temp_x_fever` (interaction).
- Leakage note: always fit imputers/scalers on training folds only (done inside `Pipeline`/CV).

## 8) Baseline models

- LogisticRegression (OvR): interpretable linear baseline (needs standardized numerics).
- RandomForestClassifier: non-linear, handles mixed types, feature*importances*.
- Metrics: macro F1 (primary), accuracy and ROC‑AUC OvR (secondary). Save confusion matrices and reports under `outputs/`.

## 9) Hyperparameter tuning and selection

- Script/section: `src/model_tuning_interpret.py` (also in notebook cells).
- Method: `RandomizedSearchCV` (n_iter≈16, StratifiedKFold(3), scoring='f1_macro').
- Spaces:
  - RF: `n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`.
  - XGB: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.
  - LGBM: `n_estimators`, `num_leaves`, `learning_rate`, `min_child_samples`.
- Artifacts: results CSVs in `outputs/tables/` and final plots under `outputs/figures/`.

## 10) Interpretation with SHAP

- For tree models use `shap.TreeExplainer`; compute global mean |SHAP| by feature and summary plots.
- Outputs: per-class and overall importance plots; tables saved under `outputs/tables/`.
- Talking point: SHAP attributes prediction differences to features via Shapley values (game theory). For binaries, it highlights the most influential symptoms.

## 11) Persist and deploy

- Training for app: `train_and_save.py` saves `models/weather_disease_model.joblib`, `models/feature_names.joblib`, and `models/label_encoder.joblib`.
- App: `app.py` (Streamlit) collects inputs, reconstructs feature vector aligned to training `feature_names`, recomputes engineered features, predicts, and shows top‑5 probabilities.
- Risk & mitigation: ensure the same preprocessing in training and inference (ideally export a single `Pipeline`).

## 12) How to interpret common outputs quickly

- Confusion matrix: rows = true, cols = predicted; look for diagonal strength, class‑specific errors.
- Classification report: per‑class precision/recall/F1; macro F1 is the average across classes.
- ROC‑AUC OvR: computes per-class one‑vs‑rest AUC and averages; complements F1 by threshold‑free ranking quality.
- Feature importance vs SHAP: tree impurity importances can be biased; SHAP is more reliable for global/local explanations.

## 13) Viva Q&A anchors

- Why macro F1? Class imbalance.
- Why RandomizedSearch? Efficient exploration of large spaces.
- How do you avoid leakage? Fit preprocessing only on training folds and package with the model.
- What is Cramer’s V? Effect size for chi‑square; 0–1 scale.
- Why OneHotEncoder(handle_unknown='ignore')? Safety on unseen categories.

## 14) Reproducibility

- Set `random_state` everywhere (splits/models).
- Save encoder and feature order; write results to `outputs/` for your report.

Use this as your talk track; when presenting, show a plot/table and immediately map it to the code above (what/why/how/interpretation).
