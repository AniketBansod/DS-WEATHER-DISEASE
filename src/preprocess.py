import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -------------------------
# Load raw dataset
# -------------------------
df = pd.read_csv("../data/disease_weather.csv")

# Separate features
target = "prognosis"
num_features = ['Age', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
cat_features = ['Gender']

# Symptom features (binary columns: 0/1)
symptom_features = [c for c in df.columns if c not in num_features + cat_features + [target]]

# -------------------------
# Preprocessing pipelines
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

symptom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))  # symptoms already 0/1
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
        ("symptom", symptom_transformer, symptom_features)
    ]
)

# -------------------------
# Fit + transform
# -------------------------
X = df.drop(columns=[target])
y = df[target]

X_processed = preprocessor.fit_transform(X)

# Convert back to DataFrame
num_cols = num_features
cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_features)
sym_cols = symptom_features

all_cols = list(num_cols) + list(cat_cols) + list(sym_cols)
df_processed = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
                            columns=all_cols)
df_processed[target] = y.values

# -------------------------
# Save processed dataset
# -------------------------
df_processed.to_csv("../outputs/processed_dataset.csv", index=False)
print("✅ Saved processed dataset to ../outputs/processed_dataset.csv")
