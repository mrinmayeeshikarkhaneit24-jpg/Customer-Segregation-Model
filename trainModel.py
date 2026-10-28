
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    Birch,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

DATA_FILE = Path("Mall_Customers.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
K = 5


def evaluate(X, labels):
    labels = np.asarray(labels)
    noise_count = int(np.sum(labels == -1))
    mask = labels != -1

    X_eval = X[mask]
    y_eval = labels[mask]
    n_clusters = len(np.unique(y_eval))

    result = {
        "clusters": int(len(set(labels) - {-1})),
        "noise_points": noise_count,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }

    if len(X_eval) < 3 or n_clusters < 2 or n_clusters >= len(X_eval):
        return result

    result["silhouette"] = float(silhouette_score(X_eval, y_eval))
    result["davies_bouldin"] = float(
        davies_bouldin_score(X_eval, y_eval)
    )
    result["calinski_harabasz"] = float(
        calinski_harabasz_score(X_eval, y_eval)
    )
    return result


def main():
    df = pd.read_csv(DATA_FILE)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # K-Means elbow curve
    elbow = {}
    for k in range(2, min(11, len(Z))):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(Z)
        elbow[k] = float(model.inertia_)

    k = min(K, len(Z) - 1)

    models = {
        "K-Means": KMeans(
            n_clusters=k, random_state=42, n_init=10
        ),
        "Agglomerative": AgglomerativeClustering(n_clusters=k),
        "Gaussian Mixture": GaussianMixture(
            n_components=k, random_state=42, n_init=5
        ),
        "BIRCH": Birch(n_clusters=k),
    }

    # DBSCAN discovers its own cluster count.
    models["DBSCAN"] = DBSCAN(eps=0.6, min_samples=4)

    # Mean Shift estimates a bandwidth from the dataset.
    bandwidth = estimate_bandwidth(
        Z, quantile=0.2, n_samples=min(500, len(Z))
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    models["Mean Shift"] = MeanShift(bandwidth=bandwidth)

    assignments = {}
    metrics = {}
    fitted_models = {}

    for name, model in models.items():
        if name == "Gaussian Mixture":
            model.fit(Z)
            labels = model.predict(Z)
        else:
            labels = model.fit_predict(Z)

        assignments[name] = np.asarray(labels, dtype=int)
        metrics[name] = evaluate(Z, labels)
        fitted_models[name] = model

        if hasattr(model, "inertia_"):
            metrics[name]["inertia"] = float(model.inertia_)
        else:
            metrics[name]["inertia"] = np.nan

    # Cluster assignments for interactive visualization.
    clustered = df.copy()
    for name, labels in assignments.items():
        safe_name = name.replace(" ", "_")
        clustered[f"{safe_name}_Cluster"] = labels

    # Human-readable profile table for the baseline K-Means model.
    clustered["KMeans_Cluster"] = assignments["K-Means"]
    profiles = clustered.groupby("KMeans_Cluster")[FEATURES].mean().round(2)

    artifact = {
        "scaler": scaler,
        "models": fitted_models,
        "features": FEATURES,
        "metrics": metrics,
        "elbow": elbow,
        "assignments": assignments,
        "clustered": clustered,
        "profiles": profiles,
        "training_rows": len(clustered),
    }

    joblib.dump(artifact, MODEL_DIR / "customer_segmentation.joblib")
    clustered.to_csv(MODEL_DIR / "clustered_customers.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(MODEL_DIR / "model_comparison.csv")

    print("\nModel comparison")
    print(pd.DataFrame(metrics).T.round(4))
    print("\nK-Means profiles")
    print(profiles)
    print("\nSaved models and cluster assignments successfully.")


if __name__ == "__main__":
    main()
