"""
on test modèle supervisé.

ici on charge le modèle Random Forest déjà entraîné,
on recharge les données,
on refait le même découpage train/test,
on prédit sur les données de test,
et on génère les métriques, les graphiques et le rapport.

On calcule :
accuracy  → taux global de bonnes prédictions
precision → parmi les joueurs prédits bots, combien sont vraiment des bots
recall    → parmi les vrais bots, combien ont été détectés
F1-score  → compromis entre precision et recall
matrice de confusion → tableau des bonnes/mauvaises classifications
importance des variables → variables les plus utilisées par le modèle
"""

import joblib # charger ou sauvergarder modèle en python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, #pourcentage total bonnes prédictions
    precision_score, # vrais bots détectés / total des joueurs prédits bots
    recall_score, #vrais bots détectés / total des vrais bots
    f1_score, #moyenne équilibrée entre precision et recall.
    confusion_matrix, #prédictions de bot/humain sous forme de tableau
    classification_report, #rapport complet 
    ConfusionMatrixDisplay, #transforme la matrice de confusion en graphique propre.
)

from config import (
    SIMULATED_DATA_PATH, #Chemin du dataset
    SUPERVISED_MODEL_PATH, #Chemin du modèle Random Forest déjà sauvegardé
    SUPERVISED_REPORT_PATH, # Chemin du rapport texte
    CONFUSION_MATRIX_PATH, # Chemin de la matrice de confusion
    FEATURE_IMPORTANCE_PATH, # Chemin du graph d’importance des var
    FEATURE_COLUMNS, # toutes les colonnes explicatives
    TARGET_COLUMN, #Nom colonne cible
    TEST_SIZE, #Proportion des données qu'on garde pour test
    RANDOM_STATE, #garantit que le découpage train/test reste toujours le même
) #importe les paramètres centraux du projet grace au chemin d'accès rapide

from sklearn.model_selection import train_test_split


def load_data():  #Charge données et refait le même split train/test

    df = pd.read_csv(SIMULATED_DATA_PATH) #fichier CSV avec joueurs simulés

    X = df[FEATURE_COLUMNS] #contient colonnes de FEATURE_COLUMNS
    y = df[TARGET_COLUMN] # colonne cible

    return train_test_split(
        X,# var d'entrée
        y, # vrai rep
        test_size=TEST_SIZE, # si test_size = 0.2 par ex = 20% données servent au test et 80% servent a l'entrainement (train)
        random_state=RANDOM_STATE, #garantit que le découpage est toujours le même
        stratify=y, # conserve la même proportion humains/bots dans le train et dans le test
    )


def load_model(): # sert à charger la Random Forest déjà entraîné

    return joblib.load(SUPERVISED_MODEL_PATH) # renvoié modèle entrainé


def save_confusion_matrix(y_test, y_pred): # ici y_test → vraies classes et y_pred → classes prédites par le modèle

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True) #crée le dossier parent si nécessaire, parents=True : Permet de créer tous les dossiers intermédiaires, exist_ok=True : Évite une erreur si le dossier existe déjà

    cm = confusion_matrix(y_test, y_pred) #on calcule la matrice de confusion

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, # ?
        display_labels=["Humain", "Bot"] # nom sur le graph
    ) # ici on crée un objet pour afficher la matrice de confusion proprement

    display.plot()
    plt.title("Matrice de confusion - Random Forest")
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close() # graph


def save_feature_importance(model): #crée le graphique d’importance des variables

    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True) #crée le dossier outputs/figures/ si nécessaire.

    importances = model.feature_importances_ #récupère l’importance des variables calculée par la Random Forest : une Random Forest est composée de plusieurs arbres de décision, elle peut estimer quelles variables ont le plus contribué aux décisions, plus l’importance est grande, plus la variable a été utile au modèle

    importance_df = pd.DataFrame({ 
        "feature": FEATURE_COLUMNS,
        "importance": importances,#crée un DataFrame avec deux colonnes : feature = nom des features et importance = valeurs
    }).sort_values(by="importance", ascending=True) # trie les variables de la moins importante à la plus importante

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Importance des variables - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, bbox_inches="tight")
    plt.close() #graph


def save_report(y_test, y_pred):

    SUPERVISED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True) # on crée dossier si nécéssaire

    accuracy = accuracy_score(y_test, y_pred) #on calcul accuracy : mesure le taux global de bonnes prédictions
    precision = precision_score(y_test, y_pred) #calcule la précision pour la classe positive 1 = bot : parmi les joueurs prédits bots, combien sont vraiment des bots = vrai positif
    recall = recall_score(y_test, y_pred) #calcule le rappel : parmi les vrais bots, combien ont été détectés ?
    f1 = f1_score(y_test, y_pred) #ndicateur synthétique entre precision et recall

    report = classification_report(
        y_test,
        y_pred, 
        target_names=["Humain", "Bot"]
    ) # création rapport détaillé de l'études

    content = f"""
ÉVALUATION DU MODÈLE SUPERVISÉ - RANDOM FOREST

Accuracy : {accuracy:.4f}
Precision : {precision:.4f}
Recall : {recall:.4f}
F1-score : {f1:.4f}

Rapport détaillé :

{report}
"""
    #texte complet du rapport

    with open(SUPERVISED_REPORT_PATH, "w", encoding="utf-8") as file:  # on ouvre fichier en écriture w= write, utf8 = accent fr
        file.write(content) # écrit rap dans fichier

    print(content) # on affiche le rapport


def evaluate_supervised_model(): #fonction orchestre toute l’évaluation des bot / humain, vérif ect

    X_train, X_test, y_train, y_test = load_data() 
    model = load_model()

    y_pred = model.predict(X_test)

    save_report(y_test, y_pred)
    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(model)

    print(f"Rapport sauvegardé dans : {SUPERVISED_REPORT_PATH}")
    print(f"Matrice de confusion sauvegardée dans : {CONFUSION_MATRIX_PATH}")
    print(f"Importance des variables sauvegardée dans : {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    evaluate_supervised_model()