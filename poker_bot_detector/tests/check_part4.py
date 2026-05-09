from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import (
    FRUGALITY_REPORT_PATH,
    SUPERVISED_MODEL_PATH,
    UNSUPERVISED_MODEL_PATH,
)


def check_file_exists(path, name):
    assert path.exists(), f"Fichier manquant : {name}"


def check_file_not_empty(path, name):
    assert path.stat().st_size > 0, f"Fichier vide : {name}"


def check_report_content(path):
    content = path.read_text(encoding="utf-8")

    expected_keywords = [
        "RAPPORT DE FRUGALITÉ",
        "Nombre de joueurs analysés",
        "Nombre de variables utilisées",
        "Random Forest",
        "K-Means",
        "Taille totale des modèles",
        "IA frugale",
    ]

    for keyword in expected_keywords:
        assert keyword in content, f"Mot-clé manquant dans le rapport : {keyword}"


def main():
    print("Vérification des fichiers de la partie 4...")

    files_to_check = [
        (FRUGALITY_REPORT_PATH, "frugality_report.txt"),
        (SUPERVISED_MODEL_PATH, "supervised_model.pkl"),
        (UNSUPERVISED_MODEL_PATH, "unsupervised_model.pkl"),
    ]

    for path, name in files_to_check:
        check_file_exists(path, name)
        check_file_not_empty(path, name)
        print(f"{name} : OK")

    print()
    print("Vérification du contenu du rapport de frugalité...")

    check_report_content(FRUGALITY_REPORT_PATH)

    print("Contenu du rapport : OK")

    print()
    print("Tous les contrôles de la partie 4 sont validés.")
    print("Le rapport de frugalité fonctionne correctement.")


if __name__ == "__main__":
    main()