"""
Entraînement modèle supervisé - Random Forest

Objectif : Apprendre à distinguer les humains des bots.
Interprétabilité : Visualisation de la structure logique d'un arbre.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from config import (
    SIMULATED_DATA_PATH,
    SUPERVISED_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

def load_dataset() -> pd.DataFrame:
    """Charge les données simulées."""
    return pd.read_csv(SIMULATED_DATA_PATH)

def split_dataset(df: pd.DataFrame):
    """Prépare les sets d'entraînement et de test avec stratification."""
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
    """Entraîne la forêt aléatoire."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1, # Utilise tous les cœurs du MacBook Air
    )
    model.fit(X_train, y_train)
    return model

def save_model(model: RandomForestClassifier) -> None:
    """Sauvegarde le modèle au format .pkl."""
    SUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SUPERVISED_MODEL_PATH)

def visualize_single_tree(model, feature_cols):
    """Génère une image claire de la logique d'un arbre."""
    # On s'assure que le dossier existe
    os.makedirs("outputs/figures", exist_ok=True)
    
    plt.figure(figsize=(20, 10))
    # On affiche l'arbre n°0. max_depth=3 pour garder l'image lisible.
    plot_tree(model.estimators_[0], 
              feature_names=feature_cols, 
              class_names=["Humain", "Bot"], 
              filled=True, 
              max_depth=3, 
              fontsize=10,
              precision=2)
    
    plt.title("Logique de décision : Extrait d'un arbre de la Random Forest", fontsize=15)
    plt.savefig("outputs/figures/random_forest_tree_sample.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ Visualisation de l'arbre sauvegardée dans outputs/figures/")

def train_supervised_model():
    """Fonction principale d'entraînement et de visualisation."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = train_random_forest(X_train, y_train)
    save_model(model)

    # On remplace SHAP par la visualisation directe, beaucoup plus légère
    visualize_single_tree(model, FEATURE_COLUMNS)

    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Exécution
    model, X_train, X_test, y_train, y_test = train_supervised_model()

    print("\n" + "="*30)
    print("SUCCÈS DE L'ENTRAÎNEMENT")
    print("="*30)
    print(f"Joueurs analysés : {len(X_train) + len(X_test)}")
    print(f"Modèle sauvegardé : {SUPERVISED_MODEL_PATH}")
    print("="*30)