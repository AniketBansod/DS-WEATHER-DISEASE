# src/eda_utils.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    return pd.read_csv(data_path)

def plot_target_distribution(df, outdir):
    plt.figure(figsize=(8,5))
    df['prognosis'].value_counts().plot(kind='bar')
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figures", "target_distribution.png"))
    plt.close()

def plot_symptom_counts(df, outdir):
    symptom_cols = [c for c in df.columns if c not in ["prognosis", "Age", "Gender"]]
    freq = df[symptom_cols].sum().sort_values(ascending=False).head(30)
    plt.figure(figsize=(10,6))
    freq.plot(kind='bar')
    plt.title("Top Symptom Frequencies")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figures", "symptom_freq_top30.png"))
    plt.close()

def run_all(data_path, outdir):
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)

    # 1. Load data
    df = load_data(data_path)

    # 2. Save missing values
    missing = df.isna().sum()
    missing.to_csv(os.path.join(outdir, "tables", "missing.csv"))

    # 3. Target distribution
    plot_target_distribution(df, outdir)

    # 4. Symptom counts
    plot_symptom_counts(df, outdir)

    print("✅ All EDA outputs saved in:", outdir)
