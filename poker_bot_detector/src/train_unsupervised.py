import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# On importe les paramètres depuis ton nouveau config.py
from train_config import (
    SIMULATED_DATA_PATH, UNSUPERVISED_FEATURE_COLUMNS, TARGET_COLUMN,
    RANDOM_STATE, UNSUPERVISED_CONTAMINATION, UNSUPERVISED_MODEL_PATH,
    UNSUPERVISED_REPORT_PATH, UNSUPERVISED_RESULTS_PATH,
    UNSUPERVISED_CONFUSION_MATRIX_PATH, ANOMALY_SCORE_DISTRIBUTION_PATH,
)

def load_dataset():
    return pd.read_csv(SIMULATED_DATA_PATH)

def build_isolation_forest_model() -> Pipeline:
    """Pipeline : Standardisation + Isolation Forest"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("isolation_forest", IsolationForest(
                n_estimators=100,
                contamination=UNSUPERVISED_CONTAMINATION,
                max_samples=256,
                random_state=RANDOM_STATE,
                n_jobs=-1, # Utilise tous les coeurs de ton Mac
            )),
        ]
    )

def create_risk_level(score, medium_threshold, high_threshold):
    if score >= high_threshold: return "high"
    if score >= medium_threshold: return "medium"
    return "low"

def build_results_dataframe(df, model):
    X = df[UNSUPERVISED_FEATURE_COLUMNS]
    isolation_predictions = model.predict(X)
    
    # Conversion : IF sort -1 pour anomalie, on veut 1 pour notre rapport
    predicted_suspect = (isolation_predictions == -1).astype(int)
    
    # Score de suspicion (on inverse la decision_function)
    anomaly_scores = -model.decision_function(X)

    results = df[["player_id", TARGET_COLUMN]].copy()
    results["predicted_suspect"] = predicted_suspect
    results["anomaly_score"] = anomaly_scores

    # Définition des seuils de risque basés sur les quantiles
    results["risk_level"] = results["anomaly_score"].apply(
        lambda s: create_risk_level(s, results["anomaly_score"].quantile(0.85), results["anomaly_score"].quantile(0.95))
    )
    return results

def save_report(results):
    UNSUPERVISED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = results[TARGET_COLUMN], results["predicted_suspect"]
    
    content = f"""
ÉVALUATION NON SUPERVISÉE (ISOLATION FOREST)
Variables utilisées : {', '.join(UNSUPERVISED_FEATURE_COLUMNS)}

Accuracy  : {accuracy_score(y_true, y_pred):.4f}
Precision : {precision_score(y_true, y_pred):.4f}
Recall    : {recall_score(y_true, y_pred):.4f}
F1-score  : {f1_score(y_true, y_pred):.4f}

Rapport détaillé :
{classification_report(y_true, y_pred, target_names=['Humain', 'Suspect'])}
"""
    with open(UNSUPERVISED_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)

def save_visuals(results):
    """Génère les graphiques pour le rapport final"""
    # Matrice de Confusion
    cm = confusion_matrix(results[TARGET_COLUMN], results["predicted_suspect"])
    ConfusionMatrixDisplay(cm, display_labels=["Humain", "Suspect"]).plot()
    plt.title("Matrice de confusion - Isolation Forest")
    plt.savefig(UNSUPERVISED_CONFUSION_MATRIX_PATH)
    plt.close()

    # Distribution des scores
    plt.figure(figsize=(8, 5))
    for label, name in [(0, "Humains"), (1, "Bots")]:
        subset = results[results[TARGET_COLUMN] == label]
        plt.hist(subset["anomaly_score"], bins=30, alpha=0.6, label=name)
    plt.title("Répartition des scores d'anomalie")
    plt.legend()
    plt.savefig(ANOMALY_SCORE_DISTRIBUTION_PATH)
    plt.close()

def train_unsupervised_model():
    df = load_dataset()
    X = df[UNSUPERVISED_FEATURE_COLUMNS]

    model = build_isolation_forest_model()
    model.fit(X)

    # Sauvegardes
    UNSUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, UNSUPERVISED_MODEL_PATH)
    
    results = build_results_dataframe(df, model)
    UNSUPERVISED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(UNSUPERVISED_RESULTS_PATH, index=False)
    
    save_report(results)
    save_visuals(results)
    return model, results

if __name__ == "__main__":
    train_unsupervised_model()