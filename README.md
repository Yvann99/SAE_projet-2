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
## À noter dans la simulation des comportements humains/bots : 
L'humain : La fonction reçoit hand_index et n_hands. Pourquoi ? Pour simuler la dégradation des performances. Plus l'humain joue longtemps (plus le ratio hand_index / n_hands est élevé), plus il va "tilter", s'ennuyer ou faire des erreurs de concentration. Son niveau de bruit (noise_level) augmente avec le temps.

Le bot : Il n'a pas ces paramètres. Pour un automate, la 1ère main et la 10 000ème main sont traitées avec la même précision glaciale. Il est invariant dans le temps.

La distance L1 plûtot qu'un vrai faux pour avoir une valeur plus précise à comparer :
Pour comparer une action réelle à une distribution de probabilités, il faut qu'elles parlent la même langue.

La GTO est déjà un vecteur de probabilités : [Fold%, Call%, Raise%].

Exemple : [0.1, 0.7, 0.2]

L'action du joueur est transformée en vecteur binaire :

Si le joueur a fait Call, son vecteur devient [0.0, 1.0, 0.0].

2. Le calcul de l'écart (La somme des différences absolues)

On soustrait chaque élément du vecteur joueur à celui du vecteur GTO, et on prend la valeur absolue (pour que les écarts ne s'annulent pas entre eux).

Distance=∣Fold joueur- Fold GTO∣+∣Call joueur−Call GTO∣+∣Raise joueur−Raise GTO∣
3. Exemple concret
Imaginons un spot où la GTO recommande de souvent folder, mais de bluffer de temps en temps :
GTO freqs : Fold: 0.8, Call: 0.0, Raise: 0.2
Action Joueur : Le joueur décide de Call (une erreur, car la GTO ne recommandait jamais de payer ici). Son vecteur est [0, 1, 0].
Le calcul :
Écart Fold : ∣0−0.8∣=0.8
Écart Call : ∣1−0.0∣=1.0
Écart Raise : ∣0−0.2∣=0.2
Total Distance L1 : 0.8+1.0+0.2=2.0
Pourquoi utiliser la distance L1 plutôt que juste "Vrai/Faux" ?

Nuance de l'erreur : Faire un "Call" alors que la GTO disait de "Call" à 40% et "Fold" à 60% donnera une petite distance. Faire un "Call" alors que la GTO disait "Fold" à 100% donnera une distance maximale (2.0).

Signature du Bot : Un bot GTO aura une distance L1 moyenne très proche de 0 sur le long terme (car il finit par équilibrer ses actions selon les fréquences). Un humain aura une distance L1 moyenne beaucoup plus élevée et instable.

Sensibilité aux fréquences : C'est le seul moyen de détecter si un joueur "mixe" ses mains correctement (ex: bluffer exactement la fréquence recommandée).

C'est cette valeur numérique que ton Random Forest va analyser. S'il voit une moyenne de distance L1 de 0.05, il "saura" presque à coup sûr que c'est un bot.

Reproductibilité et Contrôle du Hasard

Afin de garantir la validité scientifique des résultats et la stabilité du pipeline de données, le projet utilise un paramètre random_state (Seed) systématique. Ce mécanisme permet de transformer le processus de simulation pseudo-aléatoire en une expérience déterministe : le même jeu de données (cartes, temps de décision, profils de joueurs) sera généré à chaque exécution. Cette approche est cruciale pour le débogage, la comparaison objective des performances des modèles de Machine Learning et la reproductibilité des tests sur n'importe quel environnement de calcul.

### Analyse de la structure décisionnelle (Plot Tree)

Pour assurer la transparence du modèle et lever l'effet « boîte noire », le projet intègre un module de visualisation de la structure logique des arbres de décision. Cette méthode permet d'extraire un estimateur individuel de la forêt aléatoire afin d'auditer les règles de classification apprises.

La lecture de l'arbre s'articule autour de trois indicateurs clés :

* **Seuils de segmentation :** Chaque nœud identifie la variable la plus discriminante (ex: `decision_time_std`) et le seuil précis permettant de séparer les profils.
* **Indice de Gini :** Ce coefficient mesure la pureté des nœuds. Un indice proche de zéro indique que le modèle a réussi à isoler parfaitement un groupe de joueurs (humains ou bots).
* **Distribution des classes :** La visualisation permet de confirmer que les décisions du modèle reposent sur des critères comportementaux cohérents, tels que la régularité anormale du rythme de jeu ou l'absence de dégradation de la performance au cours du temps.

Cette capacité d'audit est fondamentale pour valider la robustesse de la détection et garantir que l'algorithme ne s'appuie pas sur des corrélations biaisées.

## Analyse du Modèle Non Supervisé : Isolation Forest

L'approche non supervisée a été intégralement refondue pour passer d'un partitionnement par K-Means à une détection par **Isolation Forest**, algorithme de référence en cybersécurité pour l'identification de comportements aberrants.

### Évolutions Techniques
* **Algorithme d'isolement :** Contrairement au clustering classique, l'Isolation Forest isole les observations suspectes par partitionnement aléatoire. Les profils automatisés, ayant des statistiques extrêmes, sont isolés plus précocement que la masse des profils humains.
* **Frugalité et Sélection de Variables :** Pour réduire le "bruit" statistique, le modèle se concentre sur 6 variables critiques (frugalité) liées au timing de décision, à la variance des mises et à la stricte adhérence aux fréquences GTO.
* **Réalisme du Dataset :** La simulation a été ajustée pour refléter la réalité du marché (2000 joueurs, 5 % de bots). Cette faible proportion transforme la détection de bots en un véritable problème d'identification d'anomalies.

### Performances et Points Forts
* **Efficacité "à l'aveugle" :** Le modèle atteint un F1-Score élevé sans jamais avoir accès aux étiquettes réelles pendant l'entraînement. Il identifie les signatures robotiques par pure analyse de structure.
* **Maîtrise des Faux Positifs :** Le taux d'erreur sur les profils humains est inférieur à 1 %. Dans un scénario réel, ces profils seraient soumis à une vérification manuelle plutôt qu'à un bannissement automatique.
* **Optimisation des Ressources :** Le pipeline est extrêmement léger, s'entraînant en une fraction de seconde avec une empreinte mémoire minimale.

### Limites et Biais Identifiés
* **Biais de Simulation :** L'algorithme identifie la signature mathématique générée par nos propres scripts de simulation. Un bot de production intégrant du bruit aléatoire complexe pourrait réduire cette détectabilité.
* **Variables Spécifiques :** Le choix des variables d'élite repose sur notre connaissance préalable des comportements simulés.
* **Paramètre de Contamination :** L'algorithme utilise un taux de contamination prédéfini (5 %), une donnée rarement connue avec précision dans un environnement de production réel.

### Note Stratégique : Le Dilemme du Bot
L'efficacité du système ne réside pas seulement dans le bannissement, mais dans la contrainte imposée au tricheur. Pour échapper à une détection basée sur la perfection statistique, un créateur de bot doit introduire des erreurs volontaires. Or, au poker, toute déviation de l'optimalité mathématique réduit l'espérance de gain (EV). 

**Notre système impose un dilemme économique :** soit le bot joue de manière optimale et s'expose à une détection immédiate, soit il simule l'erreur humaine pour rester furtif, perdant ainsi sa rentabilité financière.

## Limite méthodologique

Les données sont simulées. Les résultats valident la cohérence du pipeline, mais une validation sur données réelles serait nécessaire pour conclure sur une performance opérationnelle.

La stratégie préflop utilisée est une approximation pédagogique : elle ne remplace pas un solveur GTO réel. Elle sert à créer un cadre mathématique compréhensible pour comparer les décisions humaines et les décisions automatisées.
