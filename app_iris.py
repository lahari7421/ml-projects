# app_iris.py
# Run: streamlit run app_iris.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Train Model ───────────────────────────────────────────
@st.cache_resource
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return iris, X_scaled, X_pca, y, kmeans, scaler, pca

iris, X_scaled, X_pca, y_true, kmeans, scaler, pca = train_model()

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Iris Clustering", page_icon="🌸")
st.title("🌸 Iris Flower Clustering")
st.markdown("K-Means unsupervised clustering on the classic Iris dataset.")
st.divider()

# ── Elbow Method Plot ─────────────────────────────────────
with st.expander("📈 Elbow Method — Finding Optimal K"):
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 11), inertia, 'bo-', markersize=8)
    ax.axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
    ax.set_xlabel("K"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method"); ax.legend()
    st.pyplot(fig)
    st.info("The elbow bends at K=3 → 3 species in Iris dataset ✅")

# ── Cluster Visualization ─────────────────────────────────
st.subheader("🔍 Cluster Visualization (PCA 2D)")

clusters = kmeans.predict(X_scaled)
colors   = ['#e74c3c', '#2ecc71', '#3498db']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Predicted
for i in range(3):
    mask = clusters == i
    axes[0].scatter(X_pca[mask,0], X_pca[mask,1],
                    c=colors[i], label=f'Cluster {i}',
                    alpha=0.8, edgecolors='k', lw=0.3, s=70)
centroids_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(centroids_pca[:,0], centroids_pca[:,1],
                c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[0].set_title("K-Means Predicted Clusters")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].legend()

# True
species = ['Setosa','Versicolor','Virginica']
for i in range(3):
    mask = y_true == i
    axes[1].scatter(X_pca[mask,0], X_pca[mask,1],
                    c=colors[i], label=species[i],
                    alpha=0.8, edgecolors='k', lw=0.3, s=70)
axes[1].set_title("True Species Labels")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

# ── Predict New Flower ────────────────────────────────────
st.divider()
st.subheader("🌼 Predict Your Flower's Cluster")

col1, col2 = st.columns(2)
with col1:
    sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
    sepal_w = st.slider("Sepal Width (cm)",  2.0, 5.0, 3.0, 0.1)
with col2:
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_w = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

if st.button("🌸 Predict Cluster", use_container_width=True, type="primary"):
    new_flower = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    new_scaled = scaler.transform(new_flower)
    cluster    = kmeans.predict(new_scaled)[0]
    likely     = species[cluster]

    st.success(f"### 🌸 Cluster {cluster} → Likely **{likely}**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sepal L", f"{sepal_l} cm")
    col_b.metric("Petal L", f"{petal_l} cm")
    col_c.metric("Predicted", likely)

st.caption("Built with Python · Scikit-learn · Streamlit | ML & AI Course — Mini Project 2")