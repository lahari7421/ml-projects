# app_titanic.py
# Run: streamlit run app_titanic.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)
import seaborn as sns

# ── Train Model ───────────────────────────────────────────
@st.cache_resource
def train_model():
    df = sns.load_dataset('titanic')
    df = df[['survived','pclass','sex','age','sibsp','parch','fare']].copy()
    df['age'].fillna(df['age'].median(), inplace=True)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df.dropna(inplace=True)

    X = df.drop('survived', axis=1)
    y = df['survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    return model, scaler, acc, cm, y_test, y_pred

model, scaler, acc, cm, y_test, y_pred = train_model()

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Titanic Survival", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict if a passenger would have survived the Titanic disaster.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Details")
    pclass = st.selectbox("Passenger Class",
                          [1, 2, 3],
                          format_func=lambda x: f"Class {x}")
    sex    = st.radio("Gender", ["Male", "Female"])
    age    = st.slider("Age", 1, 80, 25)

with col2:
    st.subheader("🛳️ Travel Info")
    sibsp  = st.number_input("Siblings / Spouse aboard", 0, 8, 0)
    parch  = st.number_input("Parents / Children aboard", 0, 6, 0)
    fare   = st.slider("Ticket Fare ($)", 0, 500, 50)

st.divider()

if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    sex_val = 0 if sex == "Male" else 1
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f"### ✅ Likely SURVIVED")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    else:
        st.error(f"### ❌ Likely DID NOT SURVIVE")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")

    # Probability bar
    col_a, col_b = st.columns(2)
    col_a.metric("🟢 Survived",     f"{probability[1]*100:.1f}%")
    col_b.metric("🔴 Not Survived", f"{probability[0]*100:.1f}%")

st.divider()

# ── Model Performance Section ─────────────────────────────
with st.expander("📊 View Model Performance"):
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Survived','Survived'],
                yticklabels=['Not Survived','Survived'])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred,
            target_names=['Not Survived','Survived']))

st.caption("Built with Python · Scikit-learn · Streamlit | ML & AI Course — Mini Project 1")