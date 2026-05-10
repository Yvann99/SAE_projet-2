# Poker Bot Detector — IA frugale préflop

Projet étudiant de détection de bots au poker en ligne avec une approche simple, interprétable et frugale.

## Objectif

Le projet simule des joueurs humains et des bots à partir de décisions **préflop**. Pour chaque main :

- deux cartes privées sont tirées dans un paquet de 52 cartes ;
- la main est transformée en classe préflop (`AA`, `AKs`, `AKo`, `76s`, etc.) ;
- une force de main est calculée ;
- la position est simulée (`UTG`, `MP`, `CO`, `BTN`, `SB`, `BB`) ;
- le contexte préflop est simulé : blindes, pot, mise à payer, action précédente, pot odds ;
- une stratégie théorique simplifiée est produite sous forme de fréquences `fold/call/raise` ;
- l'action du joueur est simulée selon son profil humain ou bot ;
- les écarts à la stratégie théorique sont agrégés au niveau joueur.

Les modèles apprennent ensuite à distinguer les humains et les bots à partir des statistiques agrégées issues de ces décisions préflop.

## Pipeline

```text
1. Génération des mains préflop simulées
2. Simulation des positions, blindes, pot odds et actions précédentes
3. Calcul d'une stratégie théorique simplifiée fold/call/raise
4. Simulation du comportement humain ou bot
5. Agrégation des décisions au niveau joueur
6. Entraînement supervisé avec Random Forest
7. Détection non supervisée avec K-Means léger
8. Rapport de frugalité
9. Interface Streamlit de démonstration
```

## Fichiers générés

```text
data/raw/simulated_hands.csv
    Dataset détaillé main par main.

data/raw/simulated_players.csv
    Dataset agrégé par joueur, utilisé par les modèles.

outputs/models/supervised_model.pkl
outputs/models/unsupervised_model.pkl
outputs/reports/supervised_metrics.txt
outputs/reports/unsupervised_metrics.txt
outputs/reports/frugality_report.txt
outputs/figures/confusion_matrix.png
outputs/figures/feature_importance.png
outputs/figures/unsupervised_confusion_matrix.png
outputs/figures/anomaly_score_distribution.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer le pipeline complet

```bash
python src/main.py
```

## Lancer les tests

```bash
python tests/check_part1.py
python tests/check_part2.py
python tests/check_part3.py
python tests/check_part4.py
python tests/check_part5.py
```

## Lancer l'interface

```bash
python -m streamlit run app/streamlit_app.py
```

## Variables principales

Les variables utilisées par les modèles sont agrégées à partir des décisions préflop simulées :

- `vpip` : fréquence d'entrée volontaire dans le pot ;
- `pfr` : fréquence de relance préflop ;
- `af` : facteur d'agressivité ;
- `gto_similarity` : taux de respect de l'action théorique dominante ;
- `gto_deviation_rate` : taux d'écart à l'action théorique dominante ;
- `mean_gto_action_probability` : probabilité théorique moyenne de l'action réellement jouée ;
- `mean_l1_gto_distance` : distance moyenne entre l'action jouée et la distribution théorique `fold/call/raise` ;
- `weak_hand_play_rate` : fréquence de jeu des mains faibles ;
- `trash_hand_vpip` : fréquence de jeu des mains très faibles ;
- `premium_hand_play_rate` : fréquence de jeu des mains premium ;
- `strong_hand_aggression_rate` : fréquence de raise avec mains fortes ;
- `marginal_hand_error_rate` : taux d'erreur sur les mains marginales ;
- `gto_fold_follow_rate` : capacité à folder quand la théorie recommande fold ;
- `gto_call_follow_rate` : capacité à call quand la théorie recommande call ;
- `gto_raise_follow_rate` : capacité à raise quand la théorie recommande raise ;
- `open_raise_accuracy` : précision dans les spots d'ouverture ;
- `defense_accuracy` : précision face à une relance ;
- `threebet_accuracy` : précision face à un 3-bet ;
- `blind_defense_rate` : fréquence de défense des blindes ;
- `blind_defense_accuracy` : qualité de défense des blindes ;
- `sb_steal_attempt_rate` : fréquence de tentative de vol en petite blinde ;
- `pot_odds_call_accuracy` : cohérence des calls dans les spots avec pot odds favorables ;
- `decision_time_mean` et `decision_time_std` : temps moyen et régularité des décisions ;
- `bet_size_mean` et `bet_size_std` : taille moyenne et régularité des mises ;
- `action_entropy` : diversité des actions. L'Entropy Action permet de quantifier la richesse stratégique d'un joueur. Un bot se distingue par une entropie anormalement stable, là où l'humain présente des variations liées à la psychologie (fatigue, tilt).

## Limite méthodologique

Les données sont simulées. Les résultats valident la cohérence du pipeline, mais une validation sur données réelles serait nécessaire pour conclure sur une performance opérationnelle.

La stratégie préflop utilisée est une approximation pédagogique : elle ne remplace pas un solveur GTO réel. Elle sert à créer un cadre mathématique compréhensible pour comparer les décisions humaines et les décisions automatisées.
