"""
Configuration centrale du projet Poker Bot Detector.
Ce fichier centralise les chemins, les paramètres de simulation et les variables de modèle.
"""

from pathlib import Path

# 1. GESTION DES CHEMINS (Arborescence du projet)
# On part du fichier actuel (src/config.py) pour remonter à la racine
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

# 2. PARAMÈTRES DE SIMULATION & REPRODUCTIBILITÉ
RANDOM_STATE = 42
N_PLAYERS = 2000          # Nombre total de joueurs
BOT_RATIO = 0.05          # 5% de bots (crée une anomalie statistique)
TEST_SIZE = 0.2           # 20% des données pour le test (supervisé)

# Paramètre spécifique à l'Isolation Forest (Non supervisé)
UNSUPERVISED_CONTAMINATION = BOT_RATIO 

# 3. VARIABLES DU MODÈLE SUPERVISÉ (Random Forest)
# Liste exhaustive pour capter toutes les nuances
FEATURE_COLUMNS = [
    "vpip", "pfr", "af", "action_entropy",
    "decision_time_mean", "decision_time_std",
    "bet_size_mean", "bet_size_std",
    "gto_similarity", "gto_deviation_rate",
    "mean_gto_action_probability", "mean_l1_gto_distance",
    "std_l1_gto_distance", "avg_hand_strength",
    "weak_hand_play_rate", "trash_hand_vpip",
    "premium_hand_play_rate", "strong_hand_aggression_rate",
    "marginal_hand_error_rate", "gto_fold_follow_rate",
    "gto_call_follow_rate", "gto_raise_follow_rate",
    "open_raise_accuracy", "defense_accuracy",
    "threebet_accuracy", "blind_defense_rate",
    "blind_defense_accuracy", "sb_steal_attempt_rate",
    "pot_odds_call_accuracy", "hands_played",
    "sessions_played", "fatigue_slope"
]

TARGET_COLUMN = "is_bot"

# 4. VARIABLES DU MODÈLE NON SUPERVISÉ (Isolation Forest)
# Sélection restreinte pour réduire le bruit
UNSUPERVISED_FEATURE_COLUMNS = [
    "decision_time_mean",
    "decision_time_std",
    "bet_size_std",
    "gto_similarity",
    "mean_l1_gto_distance",
    "trash_hand_vpip"
]

# 5. CHEMINS DE SAUVEGARDE DES SORTIES
# Partie supervisée (Random Forest)
SUPERVISED_MODEL_PATH = MODELS_DIR / "supervised_model.pkl"
SUPERVISED_REPORT_PATH = REPORTS_DIR / "supervised_metrics.txt"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

# Partie non supervisée (Isolation Forest)
UNSUPERVISED_MODEL_PATH = MODELS_DIR / "unsupervised_model.pkl"
UNSUPERVISED_REPORT_PATH = REPORTS_DIR / "unsupervised_metrics.txt"
UNSUPERVISED_RESULTS_PATH = PREDICTIONS_DIR / "unsupervised_detection_results.csv"
UNSUPERVISED_CONFUSION_MATRIX_PATH = FIGURES_DIR / "unsupervised_confusion_matrix.png"
ANOMALY_SCORE_DISTRIBUTION_PATH = FIGURES_DIR / "anomaly_score_distribution.png"