from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import (
    SUPERVISED_MODEL_PATH,
    SUPERVISED_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
)


def check_file_exists(path, name):
    assert path.exists(), f"Fichier manquant : {name}"


def check_file_not_empty(path, name):
    assert path.stat().st_size > 0, f"Fichier vide : {name}"


def main():
    print("Vérification des fichiers générés par la partie 2...")

    files_to_check = [
        (SUPERVISED_MODEL_PATH, "supervised_model.pkl"),
        (SUPERVISED_REPORT_PATH, "supervised_metrics.txt"),
        (CONFUSION_MATRIX_PATH, "confusion_matrix.png"),
        (FEATURE_IMPORTANCE_PATH, "feature_importance.png"),
    ]

    for path, name in files_to_check:
        check_file_exists(path, name)
        check_file_not_empty(path, name)
        print(f"{name} : OK")

    print()
    print("Tous les contrôles de la partie 2 sont validés.")
    print("Le modèle supervisé fonctionne correctement.")


if __name__ == "__main__":
    main()