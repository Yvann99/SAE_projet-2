"""
Entraînement modèle supervisé

Modèle utilisé : Random Forest

Objectif : apprendre à distinguer les humains des bots à partir des statistiques simulée
"""

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier # modèle de ramdom forest
from sklearn.model_selection import train_test_split # découper données en deux partie

from config import (
    SIMULATED_DATA_PATH,
    SUPERVISED_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
) #on récupère les paramètres/fichier centraux du projet


def load_dataset() -> pd.DataFrame: # charge le dataset
    return pd.read_csv(SIMULATED_DATA_PATH)


def split_dataset(df: pd.DataFrame): #Sépare les données en train/test

#colonnes que le modèle utilise pour apprendre
    X = df[FEATURE_COLUMNS] # crée colonne feature
    y = df[TARGET_COLUMN] #colonne target = vrai rep: 1 = bot, 0 = humain

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE, # taille des données de test par rapp global
        random_state=RANDOM_STATE, # decoupage reproductible
        stratify=y, # meme proportion entre train et test
    ) # on retourne les données découpé  : X_train = var entrainement modèle, X_test = var test modèle, y_train = vraies rep données entrainement, y_test = vrai rep données de test


def train_random_forest(X_train, y_train) -> RandomForestClassifier: # entrainement modèle avce : X_train → variables d’entraînement, y_train → vraies réponses et ressort : un modèle Random Forest entraîné

    model = RandomForestClassifier(
        n_estimators=100, # nb arbres de décisions
        max_depth=6, # complexité modèle
        random_state=RANDOM_STATE, #tj meme modèle a chaque exé
        n_jobs=1, # utilise tout les coeur dispo du processeur
    ) # ici le modèle n'a pas encore appris, on prépare suelemnt les para pour la randm forest

    model.fit(X_train, y_train) #le modèle apprend ici : apprend à associer certains comportements à la classe bot ou humain

    return model


def save_model(model: RandomForestClassifier) -> None: # sauvegarde du modèle entrainé

    SUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True) # crée le dossier de sauvegarde si nécessaire, parents=True : Crée aussi les dossiers parents si besoin, exist_ok=True : Évite une erreur si le dossier existe déjà
    joblib.dump(model, SUPERVISED_MODEL_PATH) # sauvegarde le modèle dans un fichier .pkl


def train_supervised_model(): # on éxécute tout ici 

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    model = train_random_forest(X_train, y_train)

    save_model(model)

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train_supervised_model()

    print("Modèle supervisé entraîné avec succès.")
    print(f"Nombre de lignes train : {len(X_train)}")
    print(f"Nombre de lignes test : {len(X_test)}")
    print(f"Modèle sauvegardé dans : {SUPERVISED_MODEL_PATH}")