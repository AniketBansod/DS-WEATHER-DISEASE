# src/model_tuning_interpret.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ensure outputs exist
os.makedirs("../outputs/figures", exist_ok=True)
os.makedirs("../outputs/tables", exist_ok=True)
os.makedirs("../outputs/models", exist_ok=True)

# Try optional boosters
HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass

# SHAP import (may be heavy)
HAS_SHAP = True
try:
    import shap
except Exception:
    HAS_SHAP = False

# -------------------------
# Load preprocessed dataset
# -------------------------
df = pd.read_csv("../outputs/processed_dataset.csv")
target_col = "prognosis"
if target_col not in df.columns:
    raise ValueError(f"{target_col} not found in processed dataset")

X = df.drop(columns=[target_col])
y = df[target_col].astype(str)  # ensure string labels

# Encode labels numerically for search/training
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# -------------------------
# Candidate estimators + param spaces (small)
# -------------------------
estimators = {}
param_dists = {}

# RandomForest (baseline)
estimators["RandomForest"] = RandomForestClassifier(random_state=42)
param_dists["RandomForest"] = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 8),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", None]
}

# Optional: XGBoost
if HAS_XGB:
    estimators["XGBoost"] = XGBClassifier(objective="multi:softprob", random_state=42, use_label_encoder=False, eval_metric="mlogloss")
    param_dists["XGBoost"] = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    }

# Optional: LightGBM
if HAS_LGBM:
    estimators["LightGBM"] = LGBMClassifier(objective="multiclass", random_state=42)
    param_dists["LightGBM"] = {
        "n_estimators": randint(100, 400),
        "num_leaves": randint(15, 127),
        "learning_rate": uniform(0.01, 0.3),
        "min_child_samples": randint(5, 50)
    }

if not estimators:
    raise RuntimeError("No estimators found. Install scikit-learn and optionally xgboost/lightgbm.")

# -------------------------
# Randomized search per estimator (small n_iter)
# -------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
n_iter = 16  # small for laptop

tuning_results = []
best_models = {}

for name, est in estimators.items():
    print(f"\nRunning RandomizedSearchCV for: {name}")
    rs = RandomizedSearchCV(estimator=est,
                            param_distributions=param_dists[name],
                            n_iter=n_iter,
                            scoring="f1_macro",
                            cv=cv,
                            verbose=1,
                            random_state=42,
                            n_jobs=2)  # limit parallelism
    rs.fit(X_train, y_train)
    print(f"Best {name} score (cv): {rs.best_score_:.4f}")
    print("Best params:", rs.best_params_)

    # Save results
    res_df = pd.DataFrame(rs.cv_results_)
    res_path = f"../outputs/tables/tuning_{name}_cv_results.csv"
    res_df.to_csv(res_path, index=False)
    tuning_results.append({
        "estimator": name,
        "best_score": rs.best_score_,
        "best_params": rs.best_params_
    })

    best_models[name] = rs.best_estimator_

# Consolidate tuning results
tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv("../outputs/tables/tuning_results.csv", index=False)
print("\nSaved tuning summary to ../outputs/tables/tuning_results.csv")

# -------------------------
# Pick best model by cv score
# -------------------------
best_name = tuning_df.sort_values("best_score", ascending=False).iloc[0]["estimator"]
final_model = best_models[best_name]
print(f"\nSelected model: {best_name}")

# Refit selected model on full training set (already fitted by RandomizedSearchCV, but be explicit)
final_model.fit(X_train, y_train)

# Save final model
model_path = "../outputs/models/model_final.joblib"
joblib.dump({"model": final_model, "label_encoder": le, "feature_columns": X.columns.tolist()}, model_path)
print(f"Saved final model to {model_path}")

# -------------------------
# Evaluate on test set
# -------------------------
y_pred = final_model.predict(X_test)
print("\nClassification report on test set (labels = original):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save test predictions table
pred_df = pd.DataFrame({
    "y_true": le.inverse_transform(y_test),
    "y_pred": le.inverse_transform(y_pred)
})
pred_df.to_csv("../outputs/tables/test_predictions.csv", index=False)
print("Saved test_predictions.csv")

# -------------------------
# Feature importance (for tree models)
# -------------------------
feat_names = X.columns.tolist()
if hasattr(final_model, "feature_importances_"):
    importances = final_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv("../outputs/tables/feature_importances.csv", index=False)

    # Plot top 30
    top_n = min(30, len(fi_df))
    plt.figure(figsize=(8, max(4, 0.25 * top_n)))
    plt.barh(fi_df["feature"].head(top_n)[::-1], fi_df["importance"].head(top_n)[::-1])
    plt.xlabel("Importance")
    plt.title(f"Feature importances ({best_name})")
    plt.tight_layout()
    plt.savefig("../outputs/figures/feature_importances.png", bbox_inches="tight")
    plt.close()
    print("Saved feature_importances.png and ../outputs/tables/feature_importances.csv")
else:
    print("Selected model has no feature_importances_. Skipping global importance plot.")

# -------------------------
# SHAP explanations
# -------------------------
if not HAS_SHAP:
    print("shap package not available. Install shap to generate SHAP plots: pip install shap")
else:
    try:
        # Use a sample for SHAP if dataset large
        max_shap_sample = 2000
        if X_test.shape[0] > max_shap_sample:
            shap_idx = np.random.choice(range(X_test.shape[0]), size=max_shap_sample, replace=False)
            X_shap = X_test.iloc[shap_idx]
            y_shap = y_test[shap_idx]
        else:
            X_shap = X_test
            y_shap = y_test

        # TreeExplainer for tree models, otherwise fallback to KernelExplainer (slow)
        if hasattr(final_model, "predict_proba") and final_model.__class__.__name__.lower().find("forest") >= 0 or \
           final_model.__class__.__name__.lower().find("xgb") >= 0 or \
           final_model.__class__.__name__.lower().find("lgbm") >= 0:
            explainer = shap.TreeExplainer(final_model)
        else:
            # fallback (may be slow)
            explainer = shap.Explainer(final_model.predict_proba, X_shap)

        print("Computing SHAP values (may take time)...")
        shap_values = explainer.shap_values(X_shap)

        # For multiclass tree explainer shap_values is list-like, one array per class
        # Save global summary (for multiclass we combine absolute mean across classes)
        if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3):
            # Convert to array: classes x samples x features
            if isinstance(shap_values, list):
                # list of arrays [n_samples, n_features] per class
                mean_abs_per_feat_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
                mean_abs_per_feat = np.mean(np.vstack(mean_abs_per_feat_per_class), axis=0)
            else:
                # numpy array (C x S x F)
                mean_abs_per_feat = np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
        else:
            # shap_values is (n_samples, n_features) for binary or regression
            mean_abs_per_feat = np.mean(np.abs(shap_values), axis=0)

        shap_mean_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_per_feat})
        shap_mean_df = shap_mean_df.sort_values("mean_abs_shap", ascending=False)
        shap_mean_df.to_csv("../outputs/tables/shap_mean_abs.csv", index=False)

        # SHAP summary plot using shap.summary_plot — for multiclass use the first class's shap_values for plotting
        try:
            # Create a figure and save
            plt.figure(figsize=(8, 6))
            # For summary_plot, use shap.summary_plot (writes to current figure)
            if isinstance(shap_values, list):
                # Use aggregated shap values (sum across classes) for a global summary
                agg_shap = np.vstack(shap_values).mean(axis=0)  # samples x features? adapt..
                # fallback to using class 0 for summary_plot to avoid shape mismatch
                shap.summary_plot(shap_values[0], X_shap, show=False, max_display=30)
            else:
                shap.summary_plot(shap_values, X_shap, show=False, max_display=30)
            plt.tight_layout()
            plt.savefig("../outputs/figures/shap_summary.png", bbox_inches="tight")
            plt.close()
            print("Saved SHAP summary: ../outputs/figures/shap_summary.png")
        except Exception as e:
            print("Could not create shap.summary_plot:", e)

        # Per-class importance: mean abs shap per class
        if isinstance(shap_values, list):
            for class_idx, class_name in enumerate(le.classes_):
                mean_abs = np.mean(np.abs(shap_values[class_idx]), axis=0)
                df_sh = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
                df_sh = df_sh.sort_values("mean_abs_shap", ascending=False).head(30)

                plt.figure(figsize=(8, max(4, 0.25 * df_sh.shape[0])))
                plt.barh(df_sh["feature"][::-1], df_sh["mean_abs_shap"][::-1])
                plt.xlabel("mean |SHAP value|")
                plt.title(f"Top SHAP features for class: {class_name}")
                plt.tight_layout()
                fname = f"../outputs/figures/shap_per_class_{class_name.replace(' ', '_')}.png"
                plt.savefig(fname, bbox_inches="tight")
                plt.close()
            print("Saved per-class SHAP importance plots.")
        else:
            print("SHAP values not in expected multiclass format. Saved global summary only.")

    except Exception as e:
        print("Error computing SHAP:", e)

print("\nStep 7 completed. Final model and interpretation outputs saved under ../outputs/")
