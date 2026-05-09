import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import N_PLAYERS, BOT_RATIO, SIMULATED_DATA_PATH, SIMULATED_HANDS_PATH
from generate_dataset import generate_dataset, save_dataset, save_hands_dataset


def check_file_exists(path, name):
    assert path.exists(), f"Fichier manquant : {name}"


def check_shape(players_df):
    assert len(players_df) == N_PLAYERS, f"Le dataset joueurs doit contenir {N_PLAYERS} lignes."


def check_player_columns(df):
    expected_columns = [
        "player_id",
        "is_bot",
        "vpip",
        "pfr",
        "af",
        "action_entropy",
        "decision_time_mean",
        "decision_time_std",
        "bet_size_mean",
        "bet_size_std",
        "gto_similarity",
        "gto_deviation_rate",
        "mean_gto_action_probability",
        "mean_l1_gto_distance",
        "std_l1_gto_distance",
        "avg_hand_strength",
        "weak_hand_play_rate",
        "trash_hand_vpip",
        "premium_hand_play_rate",
        "strong_hand_aggression_rate",
        "marginal_hand_error_rate",
        "gto_fold_follow_rate",
        "gto_call_follow_rate",
        "gto_raise_follow_rate",
        "open_raise_accuracy",
        "defense_accuracy",
        "threebet_accuracy",
        "blind_defense_rate",
        "blind_defense_accuracy",
        "sb_steal_attempt_rate",
        "pot_odds_call_accuracy",
        "hands_played",
        "sessions_played",
        "fatigue_slope",
    ]
    for column in expected_columns:
        assert column in df.columns, f"Colonne joueur manquante : {column}"


def check_hands_columns(df):
    expected_columns = [
        "player_id",
        "hand_id",
        "is_bot",
        "player_profile",
        "table_size",
        "position",
        "small_blind_bb",
        "big_blind_bb",
        "posted_blind_bb",
        "stack_bb",
        "effective_stack_bb",
        "previous_action",
        "facing_raise",
        "facing_3bet",
        "pot_size_bb",
        "current_bet_bb",
        "amount_to_call_bb",
        "pot_odds",
        "card_1",
        "card_2",
        "hand_class",
        "hand_family",
        "hand_strength",
        "gto_fold_probability",
        "gto_call_probability",
        "gto_raise_probability",
        "gto_action",
        "player_action",
        "chosen_action_probability",
        "gto_l1_distance",
        "is_gto_correct",
        "decision_time",
        "bet_size_bb",
        "bet_size",
    ]
    for column in expected_columns:
        assert column in df.columns, f"Colonne main manquante : {column}"


def check_missing_values(df, name):
    assert df.isnull().sum().sum() == 0, f"{name} contient des valeurs manquantes."


def check_labels(df):
    assert set(df["is_bot"].unique()) == {0, 1}, "is_bot doit contenir uniquement 0 et 1."


def check_bot_ratio(df):
    expected_bots = int(N_PLAYERS * BOT_RATIO)
    actual_bots = int(df["is_bot"].sum())
    assert actual_bots == expected_bots, f"Nombre de bots incorrect : attendu {expected_bots}, obtenu {actual_bots}"


def check_value_ranges(df):
    probability_columns = [
        "vpip",
        "pfr",
        "gto_similarity",
        "gto_deviation_rate",
        "mean_gto_action_probability",
        "weak_hand_play_rate",
        "trash_hand_vpip",
        "premium_hand_play_rate",
        "strong_hand_aggression_rate",
        "marginal_hand_error_rate",
        "gto_fold_follow_rate",
        "gto_call_follow_rate",
        "gto_raise_follow_rate",
        "open_raise_accuracy",
        "defense_accuracy",
        "threebet_accuracy",
        "blind_defense_rate",
        "blind_defense_accuracy",
        "sb_steal_attempt_rate",
        "pot_odds_call_accuracy",
        "action_entropy",
    ]
    for column in probability_columns:
        assert df[column].between(0, 1).all(), f"{column} doit être entre 0 et 1."

    assert (df["mean_l1_gto_distance"] >= 0).all(), "Distance GTO positive attendue."
    assert (df["std_l1_gto_distance"] >= 0).all(), "Écart-type de distance GTO positif attendu."
    assert (df["decision_time_mean"] > 0).all(), "Temps moyen de décision positif attendu."
    assert (df["decision_time_std"] > 0).all(), "Variabilité du temps positive attendue."
    assert (df["hands_played"] > 0).all(), "Nombre de mains positif attendu."
    assert (df["sessions_played"] > 0).all(), "Nombre de sessions positif attendu."


def check_hands_integrity(hands_df):
    allowed_actions = {"fold", "call", "raise"}
    allowed_positions = {"UTG", "MP", "CO", "BTN", "SB", "BB"}
    allowed_previous_actions = {"unopened", "limped", "open_raise", "three_bet"}

    assert set(hands_df["gto_action"].unique()).issubset(allowed_actions), "Action GTO invalide."
    assert set(hands_df["player_action"].unique()).issubset(allowed_actions), "Action joueur invalide."
    assert set(hands_df["position"].unique()).issubset(allowed_positions), "Position invalide."
    assert set(hands_df["previous_action"].unique()).issubset(allowed_previous_actions), "Contexte préflop invalide."
    assert (hands_df["card_1"] != hands_df["card_2"]).all(), "Une main contient deux fois la même carte."
    assert hands_df["is_gto_correct"].isin([0, 1]).all(), "is_gto_correct doit valoir 0 ou 1."
    assert hands_df["facing_raise"].isin([0, 1]).all(), "facing_raise doit valoir 0 ou 1."
    assert hands_df["facing_3bet"].isin([0, 1]).all(), "facing_3bet doit valoir 0 ou 1."
    assert (hands_df["decision_time"] > 0).all(), "Temps de décision positif attendu."
    assert (hands_df["bet_size_bb"] >= 0).all(), "Taille de mise positive ou nulle attendue."
    assert hands_df["pot_odds"].between(0, 1).all(), "Pot odds entre 0 et 1 attendues."

    probability_sum = (
        hands_df["gto_fold_probability"]
        + hands_df["gto_call_probability"]
        + hands_df["gto_raise_probability"]
    )
    assert probability_sum.between(0.999, 1.001).all(), "Les fréquences GTO doivent sommer à 1."


def check_business_logic(players_df):
    humans = players_df[players_df["is_bot"] == 0]
    bots = players_df[players_df["is_bot"] == 1]

    assert bots["gto_similarity"].mean() > humans["gto_similarity"].mean(), "Les bots doivent suivre davantage la stratégie théorique."
    assert bots["gto_deviation_rate"].mean() < humans["gto_deviation_rate"].mean(), "Les bots doivent moins dévier du GTO."
    assert bots["mean_gto_action_probability"].mean() > humans["mean_gto_action_probability"].mean(), "Les bots doivent choisir des actions plus probables théoriquement."
    assert bots["mean_l1_gto_distance"].mean() < humans["mean_l1_gto_distance"].mean(), "Les bots doivent être plus proches de la distribution théorique."
    assert bots["weak_hand_play_rate"].mean() < humans["weak_hand_play_rate"].mean(), "Les bots doivent moins jouer les mains faibles."
    assert bots["trash_hand_vpip"].mean() < humans["trash_hand_vpip"].mean(), "Les bots doivent moins jouer les poubelles."
    assert bots["decision_time_std"].mean() < humans["decision_time_std"].mean(), "Les bots doivent avoir un timing plus régulier."
    assert bots["bet_size_std"].mean() < humans["bet_size_std"].mean(), "Les bots doivent avoir des sizings plus réguliers."


def main():
    print("Génération des datasets joueurs et mains...")
    players_df, hands_df = generate_dataset(
        n_players=N_PLAYERS,
        bot_ratio=BOT_RATIO,
        random_state=42,
    )

    save_dataset(players_df, SIMULATED_DATA_PATH)
    save_hands_dataset(hands_df, SIMULATED_HANDS_PATH)

    print("Vérification des fichiers...")
    check_file_exists(SIMULATED_DATA_PATH, "simulated_players.csv")
    check_file_exists(SIMULATED_HANDS_PATH, "simulated_hands.csv")

    print("Chargement des CSV...")
    players_df = pd.read_csv(SIMULATED_DATA_PATH)
    hands_df = pd.read_csv(SIMULATED_HANDS_PATH)

    print("Vérification du dataset joueurs...")
    check_shape(players_df)
    check_player_columns(players_df)
    check_missing_values(players_df, "simulated_players.csv")
    check_labels(players_df)
    check_bot_ratio(players_df)
    check_value_ranges(players_df)
    check_business_logic(players_df)

    print("Vérification du dataset mains...")
    check_hands_columns(hands_df)
    check_missing_values(hands_df, "simulated_hands.csv")
    check_hands_integrity(hands_df)

    print()
    print("Tous les contrôles de la partie 1 sont validés.")
    print("Les mains préflop, blinds, pot odds et variables agrégées sont cohérentes.")


if __name__ == "__main__":
    main()
