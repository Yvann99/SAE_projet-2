from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


def check_file_exists(path, name):
    assert path.exists(), f"Fichier manquant : {name}"


def check_file_not_empty(path, name):
    assert path.stat().st_size > 0, f"Fichier vide : {name}"


def check_app_content(path):
    content = path.read_text(encoding="utf-8")

    expected_keywords = [
        "streamlit",
        "Poker Bot Detector",
        "Random Forest",
        "K-Means",
        "Frugalité",
        "Joueurs suspects",
        "st.tabs",
        "st.dataframe",
    ]

    for keyword in expected_keywords:
        assert keyword in content, f"Mot-clé manquant dans l'application : {keyword}"


def main():
    print("Vérification de l'interface Streamlit...")

    check_file_exists(STREAMLIT_APP_PATH, "streamlit_app.py")
    check_file_not_empty(STREAMLIT_APP_PATH, "streamlit_app.py")
    check_app_content(STREAMLIT_APP_PATH)

    print("streamlit_app.py : OK")
    print()
    print("Tous les contrôles de la partie 5 sont validés.")
    print("L'interface Streamlit est prête.")


if __name__ == "__main__":
    main()