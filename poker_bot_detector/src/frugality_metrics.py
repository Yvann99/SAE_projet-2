"""
Mesure de la frugalité du projet.

Objectif :
- mesurer le coût simple du système ;
- montrer que l'approche reste légère ;
- produire un rapport exploitable dans le dossier outputs/reports.

Métriques utilisées :
- nombre de variables ;
- taille du dataset ;
- temps d'entraînement du modèle supervisé ;
- temps d'entraînement du modèle non supervisé ;
- taille des modèles sauvegardés ;
- nombre de lignes traitées.
"""

import time
from pathlib import Path

import pandas as pd

from config import (
    SIMULATED_DATA_PATH,
    SIMULATED_HANDS_PATH,
    FEATURE_COLUMNS,
    SUPERVISED_MODEL_PATH,
    UNSUPERVISED_MODEL_PATH,
    FRUGALITY_REPORT_PATH,
)

from train_supervised import train_supervised_model
from train_unsupervised import train_unsupervised_model


def get_file_size_kb(path: Path) -> float:
    """
    Retourne la taille d'un fichier en kilo-octets.
    """

    if not path.exists():
        return 0.0

    return path.stat().st_size / 1024


def measure_training_time(function_to_measure):
    """
    Mesure le temps d'exécution d'une fonction.

    On l'utilise pour mesurer :
    - l'entraînement supervisé ;
    - l'entraînement non supervisé.
    """

    start_time = time.perf_counter()
    function_to_measure()
    end_time = time.perf_counter()

    return end_time - start_time


def generate_frugality_report():
    """
    Génère un rapport de frugalité complet.
    """

    df = pd.read_csv(SIMULATED_DATA_PATH)

    n_rows = len(df)
    n_features = len(FEATURE_COLUMNS)
    dataset_size_kb = get_file_size_kb(SIMULATED_DATA_PATH)
    hands_dataset_size_kb = get_file_size_kb(SIMULATED_HANDS_PATH)
    hands_rows = len(pd.read_csv(SIMULATED_HANDS_PATH)) if SIMULATED_HANDS_PATH.exists() else 0

    print("Mesure du temps d'entraînement du modèle supervisé...")
    supervised_training_time = measure_training_time(train_supervised_model)

    print("Mesure du temps d'entraînement du modèle non supervisé...")
    unsupervised_training_time = measure_training_time(train_unsupervised_model)

    supervised_model_size_kb = get_file_size_kb(SUPERVISED_MODEL_PATH)
    unsupervised_model_size_kb = get_file_size_kb(UNSUPERVISED_MODEL_PATH)

    total_model_size_kb = supervised_model_size_kb + unsupervised_model_size_kb

    report = f"""
RAPPORT DE FRUGALITÉ DU SYSTÈME

1. Données utilisées

Nombre de joueurs analysés : {n_rows}
Nombre de mains préflop simulées : {hands_rows}
Nombre de variables utilisées : {n_features}
Taille du dataset joueurs : {dataset_size_kb:.2f} KB
Taille du dataset mains : {hands_dataset_size_kb:.2f} KB

Variables utilisées :
{chr(10).join("- " + feature for feature in FEATURE_COLUMNS)}

2. Modèle supervisé : Random Forest

Temps d'entraînement : {supervised_training_time:.4f} secondes
Taille du modèle sauvegardé : {supervised_model_size_kb:.2f} KB

3. Modèle non supervisé : K-Means

Temps d'entraînement : {unsupervised_training_time:.4f} secondes
Taille du modèle sauvegardé : {unsupervised_model_size_kb:.2f} KB

4. Synthèse frugalité

Taille totale des modèles : {total_model_size_kb:.2f} KB
Nombre total de variables : {n_features}

Interprétation :

Le système utilise un nombre volontairement limité de variables comportementales agrégées à partir des mains préflop simulées.
Les modèles choisis sont simples et rapides à entraîner :
- Random Forest pour la détection supervisée ;
- K-Means pour la détection non supervisée.

Cette architecture respecte une logique d'IA frugale, car elle évite les modèles lourds
de type deep learning et privilégie des algorithmes classiques, interprétables et peu coûteux.

5. Conclusion

Le projet montre qu'il est possible de construire un système de détection de bots
simple, rapide et peu gourmand en ressources, tout en conservant une bonne capacité
de détection sur des données simulées.

Les résultats doivent cependant être interprétés avec prudence, car le dataset est simulé.
Une validation sur données réelles serait nécessaire pour confirmer la robustesse du système.
"""

    FRUGALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(FRUGALITY_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print(f"Rapport de frugalité sauvegardé dans : {FRUGALITY_REPORT_PATH}")


if __name__ == "__main__":
    generate_frugality_report()