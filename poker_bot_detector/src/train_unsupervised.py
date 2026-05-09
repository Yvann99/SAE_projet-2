"""
Détection non supervisée avec K-Means léger en NumPy.

Objectif :
- regrouper les joueurs en clusters sans utiliser la colonne is_bot ;
- identifier automatiquement le cluster le plus suspect ;
- comparer après coup avec is_bot pour évaluer la qualité de la détection.

Important :
- is_bot n'est PAS utilisé pour entraîner le modèle ;
- is_bot sert uniquement à évaluer les résultats après coup ;
- l'implémentation K-Means est volontairement simple et frugale.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    SIMULATED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    UNSUPERVISED_MODEL_PATH,
    UNSUPERVISED_REPORT_PATH,
    UNSUPERVISED_RESULTS_PATH,
    UNSUPERVISED_CONFUSION_MATRIX_PATH,
    ANOMALY_SCORE_DISTRIBUTION_PATH,
)

# Sous-ensemble volontairement choisi pour le clustering.
# Il combine l'aspect poker/mathématique et le comportement :
# - respect de la stratégie théorique ;
# - distance à la distribution GTO simplifiée ;
# - timing ;
# - régularité des sizings ;
# - tendance à jouer les mains faibles.
UNSUPERVISED_FEATURE_COLUMNS = [
    "decision_time_mean",
    "decision_time_std",
    "bet_size_std",
    "gto_similarity",
    "mean_l1_gto_distance",
]


def load_dataset() -> pd.DataFrame:
    """Charge le dataset simulé."""
    return pd.read_csv(SIMULATED_DATA_PATH)


def _standardize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Standardise les variables pour éviter qu'une échelle domine les autres."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def fit_light_kmeans(X: pd.DataFrame, n_clusters: int = 2, random_state: int = 42, max_iter: int = 100) -> dict:
    """
    Entraîne un K-Means simple en NumPy.

    Cette version évite les modèles lourds et reste très lisible :
    1. standardisation ;
    2. initialisation des centroïdes ;
    3. affectation des points au centroïde le plus proche ;
    4. mise à jour des centroïdes.
    """
    rng = np.random.default_rng(random_state)
    X_values = X.to_numpy(dtype=float)
    X_scaled, mean, std = _standardize(X_values)

    # Initialisation robuste : un point plutôt bas et un point plutôt haut sur
    # l'axe principal de suspicion, sans utiliser is_bot.
    suspicion_axis = X_scaled[:, 3] - X_scaled[:, 4] - 0.5 * X_scaled[:, 1] - 0.5 * X_scaled[:, 2]
    low_index = int(np.argmin(suspicion_axis))
    high_index = int(np.argmax(suspicion_axis))
    centroids = np.vstack([X_scaled[low_index], X_scaled[high_index]])

    for _ in range(max_iter):
        distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            cluster_points = X_scaled[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centroids[cluster_id] = cluster_points.mean(axis=0)
            else:
                new_centroids[cluster_id] = X_scaled[rng.integers(0, len(X_scaled))]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return {
        "type": "light_kmeans",
        "features": UNSUPERVISED_FEATURE_COLUMNS,
        "mean": mean,
        "std": std,
        "centroids": centroids,
    }


def predict_light_kmeans(model: dict, X: pd.DataFrame) -> np.ndarray:
    """Prédit le cluster de chaque ligne avec le K-Means léger."""
    X_values = X.to_numpy(dtype=float)
    X_scaled, _, _ = _standardize(X_values, model["mean"], model["std"])
    centroids = model["centroids"]
    distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1)


def identify_suspicious_cluster(df: pd.DataFrame, clusters) -> int:
    """
    Identifie le cluster le plus suspect sans utiliser is_bot.

    Le cluster suspect est celui qui combine :
    - forte similarité GTO ;
    - faible distance à la distribution théorique ;
    - faible variabilité de timing ;
    - faible variabilité de sizing ;
    - faible tendance à jouer des mains faibles.
    """
    temp = df.copy()
    temp["cluster"] = clusters
    cluster_scores = {}

    for cluster_id in sorted(temp["cluster"].unique()):
        cluster_data = temp[temp["cluster"] == cluster_id]
        score = (
            3.0 * cluster_data["gto_similarity"].mean()
            + 1.5 * cluster_data["mean_gto_action_probability"].mean()
            - 2.2 * cluster_data["mean_l1_gto_distance"].mean()
            - 1.8 * cluster_data["gto_deviation_rate"].mean()
            - 1.4 * cluster_data["weak_hand_play_rate"].mean()
            - 1.2 * cluster_data["trash_hand_vpip"].mean()
            - 0.8 * cluster_data["decision_time_std"].mean()
            - 0.8 * cluster_data["bet_size_std"].mean()
            - abs(cluster_data["fatigue_slope"].mean())
        )
        cluster_scores[cluster_id] = score

    return max(cluster_scores, key=cluster_scores.get)


def compute_suspicion_score(df: pd.DataFrame) -> pd.Series:
    """
    Calcule un score de suspicion métier.

    Plus le score est élevé, plus le joueur semble proche d'un profil bot.
    """
    return (
        3.0 * df["gto_similarity"]
        + 1.5 * df["mean_gto_action_probability"]
        - 2.2 * df["mean_l1_gto_distance"]
        - 1.8 * df["gto_deviation_rate"]
        - 1.4 * df["weak_hand_play_rate"]
        - 1.2 * df["trash_hand_vpip"]
        - 0.8 * df["decision_time_std"]
        - 0.8 * df["bet_size_std"]
        - abs(df["fatigue_slope"])
    )


def create_risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    """Transforme un score de suspicion en niveau de risque."""
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def build_results_dataframe(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Crée le tableau final des résultats."""
    X = df[UNSUPERVISED_FEATURE_COLUMNS]
    clusters = predict_light_kmeans(model, X)
    suspicious_cluster = identify_suspicious_cluster(df, clusters)
    predicted_suspect = (clusters == suspicious_cluster).astype(int)
    suspicion_scores = compute_suspicion_score(df)

    results = df[["player_id", TARGET_COLUMN]].copy()
    results["cluster"] = clusters
    results["predicted_suspect"] = predicted_suspect
    results["anomaly_score"] = suspicion_scores

    medium_threshold = results["anomaly_score"].quantile(0.70)
    high_threshold = results["anomaly_score"].quantile(0.90)
    results["risk_level"] = results["anomaly_score"].apply(
        lambda score: create_risk_level(score, medium_threshold, high_threshold)
    )

    return results


def save_model(model: dict) -> None:
    """Sauvegarde le modèle non supervisé."""
    UNSUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, UNSUPERVISED_MODEL_PATH)


def save_results(results: pd.DataFrame) -> None:
    """Sauvegarde les résultats de détection."""
    UNSUPERVISED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(UNSUPERVISED_RESULTS_PATH, index=False)


def save_report(results: pd.DataFrame) -> None:
    """Sauvegarde les métriques d'évaluation."""
    UNSUPERVISED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    y_true = results[TARGET_COLUMN]
    y_pred = results["predicted_suspect"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=["Humain", "Bot/Suspect"])
    cluster_table = pd.crosstab(
        results["cluster"],
        results[TARGET_COLUMN],
        rownames=["cluster"],
        colnames=["is_bot"],
    )

    content = f"""
ÉVALUATION DU MODÈLE NON SUPERVISÉ - K-MEANS LÉGER

Rappel important :
Le modèle n'utilise pas la colonne is_bot pendant l'entraînement.
La colonne is_bot sert uniquement à vérifier après coup si les clusters détectés
correspondent aux bots simulés.

Variables utilisées pour le clustering :
{chr(10).join('- ' + feature for feature in UNSUPERVISED_FEATURE_COLUMNS)}

Accuracy : {accuracy:.4f}
Precision : {precision:.4f}
Recall : {recall:.4f}
F1-score : {f1:.4f}

Rapport détaillé :

{report}

Répartition réelle humains/bots par cluster :

{cluster_table.to_string()}

Répartition des niveaux de risque :

{results['risk_level'].value_counts().to_string()}
"""

    with open(UNSUPERVISED_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(content)

    print(content)


def save_confusion_matrix(results: pd.DataFrame) -> None:
    """Sauvegarde la matrice de confusion du modèle non supervisé."""
    UNSUPERVISED_CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    y_true = results[TARGET_COLUMN]
    y_pred = results["predicted_suspect"]
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Humain", "Bot/Suspect"])
    display.plot()
    plt.title("Matrice de confusion - K-Means léger")
    plt.savefig(UNSUPERVISED_CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close()


def save_anomaly_score_distribution(results: pd.DataFrame) -> None:
    """Sauvegarde la distribution des scores de suspicion."""
    ANOMALY_SCORE_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    humans = results[results[TARGET_COLUMN] == 0]["anomaly_score"]
    bots = results[results[TARGET_COLUMN] == 1]["anomaly_score"]
    plt.figure(figsize=(8, 5))
    plt.hist(humans, bins=30, alpha=0.7, label="Humains")
    plt.hist(bots, bins=30, alpha=0.7, label="Bots")
    plt.title("Distribution des scores de suspicion")
    plt.xlabel("Score de suspicion")
    plt.ylabel("Nombre de joueurs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANOMALY_SCORE_DISTRIBUTION_PATH, bbox_inches="tight")
    plt.close()


def train_unsupervised_model():
    """Pipeline complet de détection non supervisée."""
    df = load_dataset()
    X = df[UNSUPERVISED_FEATURE_COLUMNS]

    model = fit_light_kmeans(X, n_clusters=2, random_state=RANDOM_STATE)
    save_model(model)

    results = build_results_dataframe(df, model)
    save_results(results)
    save_report(results)
    save_confusion_matrix(results)
    save_anomaly_score_distribution(results)

    return model, results


if __name__ == "__main__":
    model, results = train_unsupervised_model()
    print("Modèle non supervisé K-Means léger entraîné avec succès.")
    print(f"Modèle sauvegardé dans : {UNSUPERVISED_MODEL_PATH}")
    print(f"Résultats sauvegardés dans : {UNSUPERVISED_RESULTS_PATH}")
