"""
Interface Streamlit du projet Poker Bot Detector.

Objectif :
- afficher les mains préflop simulées ;
- présenter les datasets agrégés ;
- présenter les résultats supervisés et non supervisés ;
- afficher le rapport de frugalité ;
- lister les joueurs suspects.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import (
    SIMULATED_DATA_PATH,
    SIMULATED_HANDS_PATH,
    SUPERVISED_REPORT_PATH,
    UNSUPERVISED_REPORT_PATH,
    FRUGALITY_REPORT_PATH,
    UNSUPERVISED_RESULTS_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    UNSUPERVISED_CONFUSION_MATRIX_PATH,
    ANOMALY_SCORE_DISTRIBUTION_PATH,
)

st.set_page_config(
    page_title="Poker Bot Detector",
    page_icon="♠️",
    layout="wide",
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Fichier introuvable : {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_text(path: Path) -> str:
    if not path.exists():
        return f"Fichier introuvable : {path}"
    return path.read_text(encoding="utf-8")


def display_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image introuvable : {path}")


def main():
    st.title("Poker Bot Detector — IA frugale")
    st.markdown(
        """
        Cette interface présente un système de détection de bots au poker en ligne.

        La simulation prend maintenant en compte les **deux cartes privées préflop** :
        chaque main reçoit une force théorique, une action attendue et une action réelle simulée.
        Les modèles apprennent ensuite à partir des statistiques agrégées issues de ces mains.
        """
    )

    dataset = load_csv(SIMULATED_DATA_PATH)
    hands_dataset = load_csv(SIMULATED_HANDS_PATH)
    unsupervised_results = load_csv(UNSUPERVISED_RESULTS_PATH)

    if dataset.empty:
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Vue générale",
            "Mains préflop",
            "Modèle supervisé",
            "Modèle non supervisé",
            "Frugalité",
            "Joueurs suspects",
        ]
    )

    with tab1:
        st.header("Vue générale du dataset joueur")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Joueurs", len(dataset))
        col2.metric("Humains", int((dataset["is_bot"] == 0).sum()))
        col3.metric("Bots", int((dataset["is_bot"] == 1).sum()))
        col4.metric("Mains simulées", len(hands_dataset) if not hands_dataset.empty else 0)

        st.subheader("Aperçu du dataset agrégé par joueur")
        st.dataframe(dataset.head(20), use_container_width=True)

        st.subheader("Statistiques moyennes par type de joueur")
        summary_columns = [
            "vpip",
            "pfr",
            "decision_time_mean",
            "decision_time_std",
            "bet_size_std",
            "gto_similarity",
            "mean_gto_action_probability",
            "mean_l1_gto_distance",
            "weak_hand_play_rate",
            "trash_hand_vpip",
            "premium_hand_play_rate",
            "open_raise_accuracy",
            "defense_accuracy",
            "blind_defense_accuracy",
            "pot_odds_call_accuracy",
            "action_entropy",
        ]
        summary = dataset.groupby("is_bot")[summary_columns].mean()
        summary.index = summary.index.map({0: "Humain", 1: "Bot"})
        st.dataframe(summary.round(3), use_container_width=True)

    with tab2:
        st.header("Mains préflop simulées")

        if hands_dataset.empty:
            st.warning("Le dataset des mains est introuvable.")
        else:
            st.markdown(
                """
                Chaque ligne représente une décision préflop : deux cartes, la position,
                les blindes, la mise à payer, les pot odds, une distribution théorique
                fold/call/raise et l'action réellement simulée.
                """
            )

            player_options = sorted(hands_dataset["player_id"].unique())
            selected_player = st.selectbox("Choisir un joueur", player_options[:500])

            player_hands = hands_dataset[hands_dataset["player_id"] == selected_player]
            st.dataframe(player_hands.head(50), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mains du joueur", len(player_hands))
            col2.metric("Taux de respect GTO", f"{player_hands['is_gto_correct'].mean():.2%}")
            col3.metric("Force moyenne", f"{player_hands['hand_strength'].mean():.2f}")

            st.subheader("Actions du joueur sélectionné")
            st.bar_chart(player_hands["player_action"].value_counts())

    with tab3:
        st.header("Modèle supervisé — Random Forest")
        st.markdown(
            """
            Le modèle supervisé apprend directement à distinguer les humains et les bots
            à partir des variables agrégées issues des mains préflop.
            """
        )

        st.text(load_text(SUPERVISED_REPORT_PATH))
        col1, col2 = st.columns(2)
        with col1:
            display_image(CONFUSION_MATRIX_PATH, "Matrice de confusion — Random Forest")
        with col2:
            display_image(FEATURE_IMPORTANCE_PATH, "Importance des variables")

    with tab4:
        st.header("Modèle non supervisé — K-Means")
        st.markdown(
            """
            Le modèle non supervisé n'utilise pas `is_bot` pendant l'entraînement.
            Il regroupe les joueurs selon leurs statistiques préflop, puis le cluster suspect
            est identifié avec une logique métier.
            """
        )

        st.text(load_text(UNSUPERVISED_REPORT_PATH))
        col1, col2 = st.columns(2)
        with col1:
            display_image(UNSUPERVISED_CONFUSION_MATRIX_PATH, "Matrice de confusion — K-Means")
        with col2:
            display_image(ANOMALY_SCORE_DISTRIBUTION_PATH, "Distribution des scores de suspicion")

    with tab5:
        st.header("Rapport de frugalité")
        st.text(load_text(FRUGALITY_REPORT_PATH))

    with tab6:
        st.header("Joueurs suspects détectés")

        if unsupervised_results.empty:
            st.warning("Le fichier de résultats non supervisés est introuvable.")
            st.stop()

        risk_filter = st.selectbox("Filtrer par niveau de risque", ["Tous", "high", "medium", "low"])
        results_display = unsupervised_results.copy()

        if risk_filter != "Tous":
            results_display = results_display[results_display["risk_level"] == risk_filter]

        results_display = results_display.sort_values(by="anomaly_score", ascending=False)
        st.dataframe(results_display.head(100), use_container_width=True)

        st.subheader("Répartition des niveaux de risque")
        st.bar_chart(unsupervised_results["risk_level"].value_counts())


if __name__ == "__main__":
    main()
