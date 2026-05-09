"""
Configuration centrale du projet.

Ce fichier regroupe les chemins, paramètres et colonnes utilisés par tout le projet.
"""

from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossiers principaux
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Fichiers de données
SIMULATED_DATA_PATH = RAW_DATA_DIR / "simulated_players.csv"
SIMULATED_HANDS_PATH = RAW_DATA_DIR / "simulated_hands.csv"

# Paramètres de simulation
RANDOM_STATE = 42
N_PLAYERS = 1000
BOT_RATIO = 0.25

if not 0 < BOT_RATIO < 1:
    raise ValueError("BOT_RATIO doit être compris entre 0 et 1.")

# Paramètres machine learning
TEST_SIZE = 0.2

# Variables utilisées par les modèles.
# Elles sont toutes agrégées à partir des mains préflop simulées.
FEATURE_COLUMNS = [
    # Style général
    "vpip",
    "pfr",
    "af",
    "action_entropy",

    # Timing et sizing
    "decision_time_mean",
    "decision_time_std",
    "bet_size_mean",
    "bet_size_std",

    # Théorie préflop / GTO simplifiée
    "gto_similarity",
    "gto_deviation_rate",
    "mean_gto_action_probability",
    "mean_l1_gto_distance",
    "std_l1_gto_distance",

    # Qualité de jouabilité des mains
    "avg_hand_strength",
    "weak_hand_play_rate",
    "trash_hand_vpip",
    "premium_hand_play_rate",
    "strong_hand_aggression_rate",
    "marginal_hand_error_rate",

    # Analyse des spots préflop
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

    # Volume et stabilité dans le temps
    "hands_played",
    "sessions_played",
    "fatigue_slope",
]

TARGET_COLUMN = "is_bot"

# Partie supervisée
SUPERVISED_MODEL_PATH = MODELS_DIR / "supervised_model.pkl"
SUPERVISED_REPORT_PATH = REPORTS_DIR / "supervised_metrics.txt"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

# Partie non supervisée
UNSUPERVISED_MODEL_PATH = MODELS_DIR / "unsupervised_model.pkl"
UNSUPERVISED_REPORT_PATH = REPORTS_DIR / "unsupervised_metrics.txt"
UNSUPERVISED_RESULTS_PATH = PREDICTIONS_DIR / "unsupervised_detection_results.csv"
UNSUPERVISED_CONFUSION_MATRIX_PATH = FIGURES_DIR / "unsupervised_confusion_matrix.png"
ANOMALY_SCORE_DISTRIBUTION_PATH = FIGURES_DIR / "anomaly_score_distribution.png"

# Rapport de frugalité
FRUGALITY_REPORT_PATH = REPORTS_DIR / "frugality_report.txt"
