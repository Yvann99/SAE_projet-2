"""
Point d'entrée principal du projet.

Ce fichier lance :
1. la génération des datasets joueur et main par main ;
2. l'entraînement du modèle supervisé ;
3. l'évaluation du modèle supervisé ;
4. la détection non supervisée ;
5. le rapport de frugalité.
"""

from config import (
    N_PLAYERS,
    BOT_RATIO,
    RANDOM_STATE,
    SIMULATED_DATA_PATH,
    SIMULATED_HANDS_PATH,
)
from generate_dataset import generate_dataset, save_dataset, save_hands_dataset
from train_supervised import train_supervised_model
from evaluate_model import evaluate_supervised_model
from train_unsupervised import train_unsupervised_model
from frugality_metrics import generate_frugality_report


def main():
    print("=== ÉTAPE 1 : GÉNÉRATION DES DONNÉES PRÉFLOP ===")

    players_dataset, hands_dataset = generate_dataset(
        n_players=N_PLAYERS,
        bot_ratio=BOT_RATIO,
        random_state=RANDOM_STATE,
    )

    save_dataset(players_dataset, SIMULATED_DATA_PATH)
    save_hands_dataset(hands_dataset, SIMULATED_HANDS_PATH)

    print("Datasets générés avec succès.")
    print(f"Dataset joueurs : {SIMULATED_DATA_PATH}")
    print(f"Dataset mains : {SIMULATED_HANDS_PATH}")
    print(f"Nombre total de joueurs : {len(players_dataset)}")
    print(f"Nombre total de mains simulées : {len(hands_dataset)}")
    print(f"Nombre d'humains : {(players_dataset['is_bot'] == 0).sum()}")
    print(f"Nombre de bots : {(players_dataset['is_bot'] == 1).sum()}")

    print()
    print("=== ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE SUPERVISÉ ===")
    train_supervised_model()
    print("Modèle supervisé entraîné avec succès.")

    print()
    print("=== ÉTAPE 3 : ÉVALUATION DU MODÈLE SUPERVISÉ ===")
    evaluate_supervised_model()

    print()
    print("=== ÉTAPE 4 : DÉTECTION NON SUPERVISÉE ===")
    train_unsupervised_model()
    print("Modèle non supervisé entraîné avec succès.")

    print()
    print("=== ÉTAPE 5 : RAPPORT DE FRUGALITÉ ===")
    generate_frugality_report()
    print("Rapport de frugalité généré avec succès.")

    print()
    print("Pipeline terminé avec succès.")


if __name__ == "__main__":
    main()
