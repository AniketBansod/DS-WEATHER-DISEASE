import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

# -------------------
# Ensure output dirs
# -------------------
os.makedirs("../outputs/figures", exist_ok=True)
os.makedirs("../outputs/tables", exist_ok=True)
os.makedirs("../outputs/plots", exist_ok=True)

# -------------------
# Optional gradient boosting imports
# -------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# -------------------
# Load preprocessed dataset
# -------------------
df = pd.read_csv("../outputs/processed_dataset.csv")
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode target labels numerically (needed for XGB/LGBM)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# -------------------
# Candidate models
# -------------------
models = {
    "LogReg": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )

if HAS_LGBM:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, learning_rate=0.1, random_state=42
    )

# -------------------
# Train + Evaluate
# -------------------
results = []
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Map predictions back to original labels for readability
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]

    results.append({"Model": name, "Accuracy": acc, "Macro F1": macro_f1})

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title(f"Confusion Matrix: {name}")
    plt.savefig(f"../outputs/plots/confusion_{name}.png", bbox_inches="tight")
    plt.close()

    # ROC Curve (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(le.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves (macro-averaged)
        plt.figure(figsize=(7, 6))
        for i in range(len(le.classes_)):
            plt.plot(fpr[i], tpr[i], lw=1,
                     label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (One-vs-Rest): {name}")
        plt.legend(loc="best", fontsize="small")
        plt.savefig(f"../outputs/plots/roc_{name}.png", bbox_inches="tight")
        plt.close()

        auc_score = roc_auc_score(y_test_bin, y_score, multi_class="ovr")
        print(f"{name} ROC-AUC (OvR): {auc_score:.3f}")

# -------------------
# Save results table
# -------------------
results_df = pd.DataFrame(results)
results_df.to_csv("../outputs/model_baselines.csv", index=False)
print("\nSaved results to ../outputs/model_baselines.csv")
print(results_df)

# -------------------
# Summary Bar Plot
# -------------------
plt.figure(figsize=(8, 6))
x = np.arange(len(results_df))
bar_width = 0.35

plt.bar(x - bar_width/2, results_df["Accuracy"], width=bar_width, label="Accuracy")
plt.bar(x + bar_width/2, results_df["Macro F1"], width=bar_width, label="Macro F1")

plt.xticks(x, results_df["Model"])
plt.ylabel("Score")
plt.title("Model Comparison: Accuracy vs Macro F1")
plt.legend()
plt.ylim(0, 1.05)

plt.savefig("../outputs/figures/model_comparison.png", bbox_inches="tight")
plt.close()
print("Saved summary plot to ../outputs/figures/model_comparison.png")
