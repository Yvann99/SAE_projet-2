"""
Entraînement modèle supervisé avec Interprétabilité (SHAP & Visualisation)

Modèle utilisé : Random Forest
Objectif : apprendre à distinguer les humains des bots et expliquer les décisions.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree  # Pour dessiner un arbre

from config import (
    SIMULATED_DATA_PATH,
    SUPERVISED_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

def load_dataset() -> pd.DataFrame:
    return pd.read_csv(SIMULATED_DATA_PATH)

def split_dataset(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1, # Utilise tous les cœurs pour plus de rapidité
    )
    model.fit(X_train, y_train)
    return model

def save_model(model: RandomForestClassifier) -> None:
    SUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SUPERVISED_MODEL_PATH)

# --- NOUVELLE FONCTION : VISUALISATION D'UN ARBRE ---
def visualize_single_tree(model, feature_cols):
    """Génère une image du premier arbre de la forêt."""
    plt.figure(figsize=(20, 10))
    # On visualise le premier arbre (index 0) avec une profondeur limitée pour la lisibilité
    plot_tree(model.estimators_[0], 
              feature_names=feature_cols, 
              class_names=["Humain", "Bot"], 
              filled=True, 
              max_depth=3, 
              fontsize=10)
    
    plt.title("Structure logique d'un arbre de décision (extrait de la forêt)")
    # Assure-toi que le dossier outputs/figures existe
    plt.savefig("outputs/figures/random_forest_tree_sample.png", bbox_inches="tight")
    plt.close()
    print("Visualisation de l'arbre sauvegardée dans outputs/figures/")

# --- NOUVELLE FONCTION : INTERPRÉTABILITÉ SHAP ---
def apply_shap_explanation(model, X_test):
    """Calcule et sauvegarde l'explication SHAP des variables."""
    # Création de l'explainer spécifique aux modèles à base d'arbres
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Graphique global (Summary Plot) : quelles variables impactent le plus le modèle
    plt.figure(figsize=(10, 6))
    # shap_values[1] correspond à la classe "Bot"
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.title("Importance des variables via SHAP (Impact sur la détection Bot)")
    plt.savefig("outputs/figures/shap_summary_plot.png", bbox_inches="tight")
    plt.close()
    print("Graphique SHAP sauvegardé dans outputs/figures/")

def train_supervised_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = train_random_forest(X_train, y_train)
    save_model(model)

    # Lancement des visualisations
    visualize_single_tree(model, FEATURE_COLUMNS)
    apply_shap_explanation(model, X_test)

    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Petit check pour créer le dossier figure s'il manque
    import os
    os.makedirs("outputs/figures", exist_ok=True)

    model, X_train, X_test, y_train, y_test = train_supervised_model()

    print("\n--- SYNTHÈSE ---")
    print(f"Modèle supervisé entraîné avec succès.")
    print(f"Nombre de joueurs pour l'entraînement : {len(X_train)}")
    print(f"Nombre de joueurs pour le test : {len(X_test)}")
    print(f"Modèle sauvegardé dans : {SUPERVISED_MODEL_PATH}")