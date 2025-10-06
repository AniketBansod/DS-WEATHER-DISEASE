Weather & Disease Prediction using Machine Learning
📌 Project Overview

This project predicts disease outcomes based on weather conditions and symptoms using machine learning.
We built a complete data science pipeline, starting from raw dataset exploration to model deployment in a Streamlit web app.

The project demonstrates:

Data preprocessing & feature engineering

Exploratory Data Analysis (EDA)

Model training, hyperparameter tuning, and interpretability (with SHAP)

Deployment via a user-friendly Streamlit app

### 📂 Project Structure
DS-WEATHER-DISEASE/
│
├── data/                          
│   └── Weather-related disease prediction.csv
│
├── notebooks/                     
│   └── 01_eda.ipynb
│
├── outputs/                       
│   ├── figures/                   # EDA plots
│   ├── models/                    # Trained models
│   │   ├── weather_disease_model.joblib
│   │   ├── feature_names.joblib
│   │   └── label_encoder.joblib
│   ├── plots/                     # Model evaluation plots
│   ├── tables/                    # Processed + results CSVs
│   │   └── processed_dataset.csv
│   ├── model_baselines.csv        
│   └── EDA_presentation.pptx      
│
├── src/                           
│   ├── preprocess.py
│   ├── model_baselines.py
│   ├── model_tuning_interpret.py
│   ├── eda_utils.py
│   └── run_eda.py
│
├── app.py                         # Streamlit demo app
├── train_and_save.py              # Train & save final model
├── requirements.txt               
└── README.md                      

🛠️ Steps in the Project
1. 📊 Exploratory Data Analysis (EDA)

Distribution of diseases across weather conditions

Correlation between symptoms and weather

Visualizations: heatmaps, histograms, cluster maps

2. ⚙️ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature engineering:

symptom_sum = number of selected symptoms

temp_x_fever = temperature × fever indicator

3. 🤖 Modeling

Baseline models: Logistic Regression, Random Forest, SVM

Advanced models: XGBoost, LightGBM

Hyperparameter tuning with RandomizedSearchCV

Model evaluation: accuracy, precision, recall, F1-score

4. 🔍 Interpretability

Feature importance (tree-based models)

SHAP analysis for local & global interpretability

5. 🌐 Deployment

Streamlit app where user can:

Input weather data + symptoms

Get disease prediction + probability distribution

View top 5 most likely diseases

Final trained model stored in outputs/models/

🚀 How to Run the Project
1. Clone Repository
git clone <repo-link>
cd DS-WEATHER-DISEASE

2. Setup Virtual Environment
python -m venv venv
# Activate
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Train Model (Optional)
python train_and_save.py


This generates:

outputs/models/weather_disease_model.joblib

outputs/models/feature_names.joblib

outputs/models/label_encoder.joblib

5. Run Streamlit App
streamlit run app.py


Then open: 👉 http://localhost:8501

🖼️ Demo Screenshots
🔍 Streamlit App Prediction

(screenshot of app UI)

📊 Example Probability Distribution

(top 5 likely diseases bar chart)

📈 Results

Final selected model: RandomForestClassifier (tuned)

Accuracy: ~85–90%

Key features: fever, temperature, runny nose, cough, high fever

📑 Deliverables

EDA Presentation → outputs/EDA_presentation.pptx

Trained Model → outputs/models/

Streamlit App → app.py

Research/Report Resources → all scripts + outputs

👨‍💻 Contributors

Your Name – Data Science & Development

Friend’s Name – Report, PPT, and Documentation

🏆 Key Highlights

End-to-end ML project from raw dataset → deployment

Includes EDA, preprocessing, modeling, interpretability

Fully demoable via Streamlit app

Faculty-ready PPT, report, reproducible pipeline
