from pathlib import Path
import sys
import pandas as pd

# Permet d'importer les fichiers du dossier src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import (
    UNSUPERVISED_MODEL_PATH,
    UNSUPERVISED_REPORT_PATH,
    UNSUPERVISED_RESULTS_PATH,
    UNSUPERVISED_CONFUSION_MATRIX_PATH,
    ANOMALY_SCORE_DISTRIBUTION_PATH,
)


def check_file_exists(path, name):
    """Vérifie qu'un fichier existe."""
    assert path.exists(), f"Fichier manquant : {name}"


def check_file_not_empty(path, name):
    """Vérifie qu'un fichier n'est pas vide."""
    assert path.stat().st_size > 0, f"Fichier vide : {name}"


def check_results_columns(df):
    """Vérifie que le fichier de résultats contient les bonnes colonnes."""

    expected_columns = [
        "player_id",
        "is_bot",
        "cluster",
        "predicted_suspect",
        "anomaly_score",
        "risk_level",
    ]

    for column in expected_columns:
        assert column in df.columns, f"Colonne manquante : {column}"


def check_prediction_values(df):
    """Vérifie que les prédictions sont bien codées en 0 ou 1."""

    allowed_values = {0, 1}
    actual_values = set(df["predicted_suspect"].unique())

    assert actual_values.issubset(allowed_values), (
        f"Valeurs incorrectes dans predicted_suspect : {actual_values}"
    )


def check_cluster_values(df):
    """Vérifie que K-Means a bien créé 2 clusters."""

    clusters = set(df["cluster"].unique())

    assert len(clusters) == 2, (
        f"K-Means devrait produire 2 clusters, obtenu : {clusters}"
    )


def check_risk_levels(df):
    """Vérifie que les niveaux de risque sont corrects."""

    allowed_values = {"low", "medium", "high"}
    actual_values = set(df["risk_level"].unique())

    assert actual_values.issubset(allowed_values), (
        f"Niveaux de risque incorrects : {actual_values}"
    )


def check_no_missing_values(df):
    """Vérifie qu'il n'y a pas de valeurs manquantes."""

    assert df.isnull().sum().sum() == 0, (
        "Le fichier de résultats contient des valeurs manquantes."
    )


def check_file_size(df):
    """Vérifie que le fichier contient bien 1000 joueurs."""

    assert len(df) == 1000, (
        f"Le fichier de résultats doit contenir 1000 lignes, obtenu : {len(df)}"
    )


def main():
    print("Vérification des fichiers générés par la partie 3...")

    files_to_check = [
        (UNSUPERVISED_MODEL_PATH, "unsupervised_model.pkl"),
        (UNSUPERVISED_REPORT_PATH, "unsupervised_metrics.txt"),
        (UNSUPERVISED_RESULTS_PATH, "unsupervised_detection_results.csv"),
        (UNSUPERVISED_CONFUSION_MATRIX_PATH, "unsupervised_confusion_matrix.png"),
        (ANOMALY_SCORE_DISTRIBUTION_PATH, "anomaly_score_distribution.png"),
    ]

    for path, name in files_to_check:
        check_file_exists(path, name)
        check_file_not_empty(path, name)
        print(f"{name} : OK")

    print()
    print("Vérification du fichier de résultats...")

    results = pd.read_csv(UNSUPERVISED_RESULTS_PATH)

    check_file_size(results)
    check_results_columns(results)
    check_prediction_values(results)
    check_cluster_values(results)
    check_risk_levels(results)
    check_no_missing_values(results)

    print("Fichier de résultats : OK")

    print()
    print("Tous les contrôles de la partie 3 sont validés.")
    print("Le modèle non supervisé fonctionne correctement.")


if __name__ == "__main__":
    main()