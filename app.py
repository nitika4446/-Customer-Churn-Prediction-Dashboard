# File Name: app.py

# ============================================
# CUSTOMER CHURN PREDICTION APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

# Optional Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction using Machine Learning")

st.write(
    "Predict whether a customer is likely to churn or stay."
)

# ============================================
# CREATE DATASET
# ============================================

np.random.seed(42)

rows = 2000

ages = np.random.randint(18, 70, rows)
balances = np.random.randint(0, 250000, rows)
active_members = np.random.randint(0, 2, rows)
products = np.random.randint(1, 4, rows)

# Better target generation for good accuracy
churn = (
    (ages > 50).astype(int) +
    (balances > 150000).astype(int) +
    (active_members == 0).astype(int) +
    (products == 1).astype(int)
)

churn = (churn >= 2).astype(int)

data = pd.DataFrame({
    "CreditScore": np.random.randint(350, 850, rows),
    "Age": ages,
    "Tenure": np.random.randint(0, 10, rows),
    "Balance": balances,
    "NumOfProducts": products,
    "HasCrCard": np.random.randint(0, 2, rows),
    "IsActiveMember": active_members,
    "EstimatedSalary": np.random.randint(10000, 200000, rows),
    "Gender": np.random.choice(["Male", "Female"], rows),
    "Geography": np.random.choice(
        ["France", "Germany", "India"],
        rows
    ),
    "Exited": churn
})

# ============================================
# DATA PREPROCESSING
# ============================================

# Label Encoding
le_gender = LabelEncoder()

data["Gender"] = le_gender.fit_transform(
    data["Gender"]
)

# One Hot Encoding
data = pd.get_dummies(
    data,
    columns=["Geography"],
    drop_first=True
)

# Features & Target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Store column names
feature_columns = X.columns

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# MODEL TRAINING
# ============================================

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

lgbm_model.fit(X_train, y_train)

# ============================================
# MODEL COMPARISON
# ============================================

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2)
    })

results_df = pd.DataFrame(results)

st.subheader("📊 Model Accuracy Comparison")

st.dataframe(results_df)

# Best Model
best_model = xgb_model

# ============================================
# EDA SECTION
# ============================================

st.subheader("📈 Exploratory Data Analysis")

# Churn Count
fig1, ax1 = plt.subplots(figsize=(6, 4))

sns.countplot(
    x="Exited",
    data=data,
    ax=ax1
)

st.pyplot(fig1)

# Age Distribution
fig2, ax2 = plt.subplots(figsize=(8, 4))

sns.histplot(
    data["Age"],
    kde=True,
    ax=ax2
)

st.pyplot(fig2)

# ============================================
# CONFUSION MATRIX
# ============================================

st.subheader("📌 Confusion Matrix")

predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    predictions
)

fig3, ax3 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax3
)

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

st.pyplot(fig3)

# ============================================
# CLASSIFICATION REPORT
# ============================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ============================================
# STREAMLIT USER INPUT
# ============================================

st.subheader("🔮 Real-Time Customer Churn Prediction")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        350,
        850,
        600
    )

    age = st.number_input(
        "Age",
        18,
        80,
        35
    )

    tenure = st.number_input(
        "Tenure",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Balance",
        0,
        250000,
        50000
    )

with col2:

    products = st.number_input(
        "Number of Products",
        1,
        4,
        2
    )

    card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    active = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    salary = st.number_input(
        "Estimated Salary",
        10000,
        200000,
        70000
    )

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

geography = st.selectbox(
    "Country",
    ["France", "Germany", "India"]
)

# ============================================
# ENCODE USER INPUT
# ============================================

gender_encoded = le_gender.transform([gender])[0]

# IMPORTANT:
# Create SAME columns as training dataset

geo_germany = 1 if geography == "Germany" else 0
geo_india = 1 if geography == "India" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [products],
    "HasCrCard": [card],
    "IsActiveMember": [active],
    "EstimatedSalary": [salary],
    "Gender": [gender_encoded],
    "Geography_Germany": [geo_germany],
    "Geography_India": [geo_india]
})

# Match exact training columns
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# Scale input
input_scaled = scaler.transform(input_data)

# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("Predict Churn"):

    prediction = best_model.predict(
        input_scaled
    )[0]

    probability = best_model.predict_proba(
        input_scaled
    )[0][1]

    st.subheader("📢 Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer is likely to churn.\n\n"
            f"Churn Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Customer is likely to stay.\n\n"
            f"Churn Probability: {probability:.2%}"
        )

# ============================================
# FEATURE IMPORTANCE
# ============================================

st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.write(
    "🚀 Built using Python, Scikit-learn, "
    "XGBoost, LightGBM, Random Forest & Streamlit"
)







