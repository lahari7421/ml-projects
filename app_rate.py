# app_heart.py
# Run: streamlit run app_heart.py
# Dataset: https://www.kaggle.com/datasets/ronitf/heart-disease-uci
# Place heart.csv in same folder
# OR the code below auto-downloads it using ucimlrepo
# pip install ucimlrepo scikit-learn streamlit matplotlib seaborn

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report,
                              roc_curve, roc_auc_score, precision_score,
                              recall_score, f1_score)

# ── Load Dataset ──────────────────────────────────────────
@st.cache_data
def load_data():
    # Try local file first, else download from UCI
    try:
        df = pd.read_csv('heart.csv')
        # Rename if using Kaggle version
        if 'target' not in df.columns and 'num' in df.columns:
            df.rename(columns={'num': 'target'}, inplace=True)
            df['target'] = (df['target'] > 0).astype(int)
    except FileNotFoundError:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        df = pd.concat([heart.data.features,
                        heart.data.targets], axis=1)
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={'num': 'target'}, inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

# ── Train Models ──────────────────────────────────────────
@st.cache_resource
def train_models(df):
    feature_cols = ['age','sex','cp','trestbps','chol',
                    'fbs','restecg','thalach','exang',
                    'oldpeak','slope','ca','thal']

    # Keep only columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    return lr, dt, scaler, X_test, y_test, feature_cols

df = load_data()
lr, dt, scaler, X_test, y_test, feature_cols = train_models(df)

# ── Evaluate ──────────────────────────────────────────────
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    return {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'auc'      : roc_auc_score(y_test, y_prob),
        'cm'       : confusion_matrix(y_test, y_pred),
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'report'   : classification_report(y_test, y_pred,
                        target_names=['No Disease','Heart Disease'])
    }

lr_metrics = get_metrics(lr, X_test, y_test)
dt_metrics = get_metrics(dt, X_test, y_test)

# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Heart Disease Predictor",
                   page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction")
st.markdown("**Full ML Pipeline** — Logistic Regression + Decision Tree with ROC curve analysis")
st.divider()

# ── Tab Layout ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "📋 Data Overview"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Predict
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧑 Personal**")
        age  = st.slider("Age", 20, 80, 50)
        sex  = st.radio("Sex", ["Male","Female"])
        sex_val = 1 if sex == "Male" else 0

    with col2:
        st.markdown("**💓 Cardiac**")
        cp      = st.selectbox("Chest Pain Type",
                    [0,1,2,3],
                    format_func=lambda x:
                        ["Typical Angina","Atypical Angina",
                         "Non-Anginal","Asymptomatic"][x])
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang   = st.radio("Exercise Induced Angina",
                            ["No","Yes"])
        exang_val = 1 if exang == "Yes" else 0
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)

    with col3:
        st.markdown("**🩺 Clinical**")
        trestbps = st.slider("Resting BP (mmHg)", 90, 200, 120)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fbs      = st.radio("Fasting Blood Sugar > 120",
                             ["No","Yes"])
        fbs_val  = 1 if fbs == "Yes" else 0
        restecg  = st.selectbox("Resting ECG",
                    [0,1,2],
                    format_func=lambda x:
                        ["Normal","ST-T Abnormality","LV Hypertrophy"][x])

    # Extra features if in dataset
    col4, col5 = st.columns(2)
    with col4:
        slope = st.selectbox("Slope of Peak ST",
                    [0,1,2],
                    format_func=lambda x:
                        ["Upsloping","Flat","Downsloping"][x])
    with col5:
        ca   = st.slider("Major Vessels Colored", 0, 4, 0)
        thal = st.selectbox("Thal",
                    [1,2,3],
                    format_func=lambda x:
                        ["Normal","Fixed Defect","Reversible Defect"][x-1])

    st.divider()

    model_choice = st.radio("Choose Model",
                             ["Logistic Regression","Decision Tree"],
                             horizontal=True)

    if st.button("❤️ Predict Heart Disease Risk",
                 use_container_width=True, type="primary"):

        # Build input array matching feature_cols
        input_map = {
            'age': age, 'sex': sex_val, 'cp': cp,
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs_val,
            'restecg': restecg, 'thalach': thalach,
            'exang': exang_val, 'oldpeak': oldpeak,
            'slope': slope, 'ca': ca, 'thal': thal
        }
        inp = np.array([[input_map[f] for f in feature_cols]])
        inp_scaled = scaler.transform(inp)

        chosen_model = lr if model_choice == "Logistic Regression" else dt
        pred = chosen_model.predict(inp_scaled)[0]
        prob = chosen_model.predict_proba(inp_scaled)[0]

        st.divider()
        col_r1, col_r2, col_r3 = st.columns(3)

        if pred == 1:
            col_r1.error("### ⚠️ HIGH RISK\nHeart Disease Likely")
        else:
            col_r1.success("### ✅ LOW RISK\nNo Heart Disease Detected")

        col_r2.metric("Risk Probability",
                       f"{prob[1]*100:.1f}%")
        col_r3.metric("Safe Probability",
                       f"{prob[0]*100:.1f}%")

        # Risk level
        risk_pct = prob[1]
        st.progress(risk_pct,
                    text=f"Heart Disease Risk: {risk_pct*100:.1f}%")

        if risk_pct > 0.7:
            st.error("🚨 Consult a cardiologist immediately.")
        elif risk_pct > 0.4:
            st.warning("⚠️ Moderate risk — consult a doctor.")
        else:
            st.success("✅ Low risk — maintain healthy habits.")

        st.caption("⚠️ This is an ML model for educational purposes only. "
                   "Not a medical diagnosis.")

# ══════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Comparison")

    # Metrics table
    metrics_df = pd.DataFrame({
        'Metric'             : ['Accuracy','Precision','Recall','F1','AUC'],
        'Logistic Regression': [f"{lr_metrics['accuracy']:.4f}",
                                f"{lr_metrics['precision']:.4f}",
                                f"{lr_metrics['recall']:.4f}",
                                f"{lr_metrics['f1']:.4f}",
                                f"{lr_metrics['auc']:.4f}"],
        'Decision Tree'      : [f"{dt_metrics['accuracy']:.4f}",
                                f"{dt_metrics['precision']:.4f}",
                                f"{dt_metrics['recall']:.4f}",
                                f"{dt_metrics['f1']:.4f}",
                                f"{dt_metrics['auc']:.4f}"]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col_p1, col_p2 = st.columns(2)

    # Confusion Matrices
    with col_p1:
        st.markdown("**Logistic Regression — Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(lr_metrics['cm'], annot=True, fmt='d',
                    cmap='Blues', ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'])
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        st.pyplot(fig)

    with col_p2:
        st.markdown("**Decision Tree — Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(dt_metrics['cm'], annot=True, fmt='d',
                    cmap='Oranges', ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'])
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        st.pyplot(fig)

    # ROC Curve
    st.markdown("**ROC Curve Comparison**")
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, metrics, color in [
        ("Logistic Regression", lr_metrics, 'steelblue'),
        ("Decision Tree",       dt_metrics, 'darkorange')
    ]:
        fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {metrics['auc']:.3f})")

    ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Classification Reports
    with st.expander("📋 Full Classification Report"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Logistic Regression**")
            st.text(lr_metrics['report'])
        with col_c2:
            st.markdown("**Decision Tree**")
            st.text(dt_metrics['report'])

# ══════════════════════════════════════════════════════════
# TAB 3 — Data Overview
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Dataset Overview")

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Patients", len(df))
    col_d2.metric("Heart Disease",  int(df['target'].sum()))
    col_d3.metric("No Disease",     int((df['target']==0).sum()))

    st.dataframe(df.head(10), use_container_width=True)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['target'].value_counts().plot(kind='bar', ax=ax,
            color=['#2ecc71','#e74c3c'], edgecolor='black')
        ax.set_xticklabels(['No Disease','Heart Disease'], rotation=0)
        ax.set_title("Target Distribution")
        st.pyplot(fig)

    with col_v2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['age'], bins=20, color='steelblue',
                edgecolor='black', alpha=0.8)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        st.pyplot(fig)

    # Correlation heatmap
    st.markdown("**Feature Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[feature_cols + ['target']].corr(),
                annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)

