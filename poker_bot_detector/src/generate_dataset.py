"""
Génération d'un dataset simulé basé sur des décisions préflop de poker.

Objectif de cette version : être plus proche d'une vraie analyse poker.
Pour chaque main, on simule :
- deux cartes privées ;
- la position du joueur, notamment SB et BB ;
- le contexte préflop : pot, grosse blinde, mise à payer, action précédente ;
- une stratégie théorique simplifiée sous forme de fréquences fold/call/raise ;
- l'action réellement jouée par un humain ou un bot ;
- l'écart entre l'action jouée et la stratégie théorique.

Une ligne dans simulated_hands.csv = une décision préflop.
Une ligne dans simulated_players.csv = un joueur résumé par ses statistiques agrégées.
"""

import numpy as np
import pandas as pd

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]
DECK = np.array([rank + suit for rank in RANKS for suit in SUITS])

RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
ACTIONS = ["fold", "call", "raise"]
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0

HUMAN_PROFILES = [
    "recreational_loose",
    "tight_regular",
    "aggressive_regular",
    "beginner",
    "solid_regular",
]

BOT_PROFILES = [
    "gto_strict",
    "humanized_gto",
    "tight_grinder_bot",
]


def draw_two_cards(rng: np.random.Generator) -> tuple[str, str]:
    """Tire deux cartes différentes dans un paquet de 52 cartes."""
    cards = rng.choice(DECK, size=2, replace=False)
    return str(cards[0]), str(cards[1])


def get_card_rank(card: str) -> str:
    """Récupère le rang d'une carte : As -> A, Kh -> K."""
    return card[0]


def get_card_suit(card: str) -> str:
    """Récupère la couleur d'une carte : As -> s, Kh -> h."""
    return card[1]


def get_hand_class(card_1: str, card_2: str) -> str:
    """
    Convertit deux cartes en classe de main préflop.

    Exemples :
    - As Kh -> AKo ;
    - As Ks -> AKs ;
    - Qh Qd -> QQ.
    """
    rank_1 = get_card_rank(card_1)
    rank_2 = get_card_rank(card_2)
    suit_1 = get_card_suit(card_1)
    suit_2 = get_card_suit(card_2)

    value_1 = RANK_VALUE[rank_1]
    value_2 = RANK_VALUE[rank_2]

    if rank_1 == rank_2:
        return rank_1 + rank_2

    if value_2 > value_1:
        rank_1, rank_2 = rank_2, rank_1
        suit_1, suit_2 = suit_2, suit_1

    suited_marker = "s" if suit_1 == suit_2 else "o"
    return rank_1 + rank_2 + suited_marker


def compute_hand_strength(hand_class: str) -> float:
    """
    Calcule une force préflop simplifiée.

    Ce n'est pas un solver GTO complet. C'est une approximation pédagogique basée sur :
    - les paires ;
    - la hauteur des cartes ;
    - les cartes suited ;
    - la connectivité ;
    - les mains dominées et très faibles.
    """
    if len(hand_class) == 2:
        value = RANK_VALUE[hand_class[0]]
        return float(45 + value * 4)

    high_rank = hand_class[0]
    low_rank = hand_class[1]
    suited_marker = hand_class[2]

    high_value = RANK_VALUE[high_rank]
    low_value = RANK_VALUE[low_rank]

    score = high_value * 3 + low_value * 2

    if high_rank == "A":
        score += 10
    elif high_rank == "K":
        score += 7
    elif high_rank == "Q":
        score += 5

    if suited_marker == "s":
        score += 6

    gap = abs(high_value - low_value)
    if gap == 1:
        score += 7
    elif gap == 2:
        score += 4
    elif gap == 3:
        score += 2

    # Malus pour mains faibles, offsuit et déconnectées.
    if suited_marker == "o" and high_value < 12 and gap >= 5:
        score -= 9

    # Petit bonus pour broadways.
    if high_value >= 10 and low_value >= 10:
        score += 4

    return float(np.clip(score, 5, 100))


def classify_hand_family(hand_class: str, hand_strength: float) -> str:
    """Classe une main dans une famille poker lisible."""
    if len(hand_class) == 2:
        value = RANK_VALUE[hand_class[0]]
        if value >= 12:
            return "premium_pair"
        if value >= 7:
            return "medium_pair"
        return "low_pair"

    high_rank = hand_class[0]
    low_rank = hand_class[1]
    suited = hand_class[2] == "s"
    high_value = RANK_VALUE[high_rank]
    low_value = RANK_VALUE[low_rank]
    gap = abs(high_value - low_value)

    if high_rank == "A" and low_value >= 10:
        return "premium_broadway"
    if high_value >= 10 and low_value >= 10:
        return "broadway"
    if high_rank == "A" and suited:
        return "suited_ace"
    if suited and gap <= 2 and low_value >= 5:
        return "suited_connector"
    if hand_strength < 35:
        return "trash"
    if 35 <= hand_strength < 55:
        return "marginal"
    return "standard"


def position_bonus(position: str) -> float:
    """Bonus de jouabilité selon la position."""
    return {
        "UTG": -9.0,
        "MP": -5.0,
        "CO": 1.0,
        "BTN": 7.0,
        "SB": -1.0,
        "BB": 4.0,
    }[position]


def posted_blind(position: str) -> float:
    """Retourne la blinde déjà postée par la position."""
    if position == "SB":
        return SMALL_BLIND_BB
    if position == "BB":
        return BIG_BLIND_BB
    return 0.0


def simulate_preflop_context(position: str, rng: np.random.Generator) -> dict:
    """
    Simule un contexte préflop minimal : action précédente, pot, mise à payer,
    stack effectif et pot odds.
    """
    table_size = 6
    stack_bb = float(np.clip(rng.normal(loc=100, scale=28), 20, 220))
    effective_stack_bb = float(np.clip(stack_bb + rng.normal(loc=0, scale=12), 15, 220))

    if position in {"UTG", "MP"}:
        previous_action = str(rng.choice(["unopened", "limped"], p=[0.86, 0.14]))
    elif position in {"CO", "BTN"}:
        previous_action = str(rng.choice(["unopened", "limped", "open_raise"], p=[0.62, 0.13, 0.25]))
    elif position == "SB":
        previous_action = str(rng.choice(["unopened", "limped", "open_raise"], p=[0.58, 0.12, 0.30]))
    else:  # BB
        previous_action = str(rng.choice(["unopened", "limped", "open_raise", "three_bet"], p=[0.24, 0.16, 0.52, 0.08]))

    facing_raise = previous_action in {"open_raise", "three_bet"}
    facing_3bet = previous_action == "three_bet"

    base_pot = SMALL_BLIND_BB + BIG_BLIND_BB
    current_bet_bb = 0.0

    if previous_action == "limped":
        current_bet_bb = BIG_BLIND_BB
        pot_size_bb = base_pot + rng.integers(1, 4) * BIG_BLIND_BB
    elif previous_action == "open_raise":
        current_bet_bb = float(np.clip(rng.normal(loc=2.5, scale=0.35), 2.0, 3.5))
        pot_size_bb = base_pot + current_bet_bb + rng.uniform(0.0, 1.0)
    elif previous_action == "three_bet":
        current_bet_bb = float(np.clip(rng.normal(loc=8.5, scale=1.2), 6.5, 11.5))
        pot_size_bb = base_pot + current_bet_bb + rng.uniform(2.0, 5.0)
    else:
        current_bet_bb = BIG_BLIND_BB if position in {"SB", "BB"} else 0.0
        pot_size_bb = base_pot

    amount_to_call_bb = max(current_bet_bb - posted_blind(position), 0.0)
    pot_after_call = pot_size_bb + amount_to_call_bb
    pot_odds = amount_to_call_bb / pot_after_call if pot_after_call > 0 else 0.0

    return {
        "table_size": table_size,
        "stack_bb": stack_bb,
        "effective_stack_bb": effective_stack_bb,
        "small_blind_bb": SMALL_BLIND_BB,
        "big_blind_bb": BIG_BLIND_BB,
        "posted_blind_bb": posted_blind(position),
        "previous_action": previous_action,
        "facing_raise": int(facing_raise),
        "facing_3bet": int(facing_3bet),
        "pot_size_bb": float(pot_size_bb),
        "current_bet_bb": float(current_bet_bb),
        "amount_to_call_bb": float(amount_to_call_bb),
        "pot_odds": float(np.clip(pot_odds, 0.0, 0.8)),
    }


def normalize_probabilities(fold: float, call: float, raise_: float) -> dict:
    """Force des fréquences fold/call/raise valides."""
    values = np.array([fold, call, raise_], dtype=float)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total == 0:
        values = np.array([1.0, 0.0, 0.0])
    else:
        values = values / total
    return {"fold": float(values[0]), "call": float(values[1]), "raise": float(values[2])}

## On utilise ici une sigmoide pour transformer un score (relatif à la force de la main et à la position) en une valeur comprise entre 0 et 1. Plus le sore de jeu est élevé par rapport au seil défini, plus la sigmoide se rapproche de 1, qui est la probabilité de faire l'action
def get_gto_frequencies(hand_strength: float, position: str, context: dict) -> dict:
    """
    Retourne une stratégie préflop simplifiée sous forme de fréquences.

    Le modèle prend en compte :
    - la force de la main ;
    - la position ;
    - l'action précédente ;
    - la grosse blinde / petite blinde ;
    - les pot odds ;
    - la profondeur de stack.
    """
    playability = hand_strength + position_bonus(position)
    pot_odds = context["pot_odds"]
    effective_stack = context["effective_stack_bb"]
    previous_action = context["previous_action"]

    # Les mains à potentiel gagnent en jouabilité quand les stacks sont profonds.
    if effective_stack > 120 and hand_strength >= 45:
        playability += 2.0
    if effective_stack < 35:
        playability += 2.0 if hand_strength >= 72 else -2.0

    # BB/SB : la blinde déjà postée rend certains calls plus défendables.
    if position == "BB":
        playability += 3.0 + 10.0 * pot_odds
    elif position == "SB":
        playability -= 1.0

    if previous_action in {"unopened", "limped"}:
        # Spot d'ouverture ou d'isolation : on raise plus souvent les mains fortes.
        raise_score = (playability - 56) / 11
        call_score = (playability - 43) / 12
        raise_prob = 1 / (1 + np.exp(-raise_score))
        call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.55 * raise_prob)
        fold_prob = 1 - raise_prob - call_prob
        return normalize_probabilities(fold_prob, call_prob, raise_prob)

    if previous_action == "open_raise":
        # Face à une relance : call/3-bet/fold selon force, pot odds et position.
        threebet_score = (playability - 73) / 10
        call_score = (playability + 20 * pot_odds - 53) / 10
        raise_prob = 1 / (1 + np.exp(-threebet_score))
        call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.45 * raise_prob)
        fold_prob = 1 - raise_prob - call_prob
        return normalize_probabilities(fold_prob, call_prob, raise_prob)

    # Face à un 3-bet : ranges très resserrées.
    raise_score = (playability - 84) / 8
    call_score = (playability + 12 * pot_odds - 68) / 9
    raise_prob = 1 / (1 + np.exp(-raise_score))
    call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.5 * raise_prob)
    fold_prob = 1 - raise_prob - call_prob
    return normalize_probabilities(fold_prob, call_prob, raise_prob)


def recommended_action(gto_freqs: dict) -> str:
    """Retourne l'action ayant la fréquence théorique la plus forte."""
    return max(gto_freqs, key=gto_freqs.get)


def sample_action_from_frequencies(freqs: dict, rng: np.random.Generator) -> str:
    """Échantillonne fold/call/raise à partir d'une distribution."""
    return str(rng.choice(ACTIONS, p=[freqs["fold"], freqs["call"], freqs["raise"]]))


def choose_human_action(gto_freqs: dict, hand_strength: float, context: dict, hand_index: int, total_hands: int, profile: str, rng: np.random.Generator) -> str:
    """
    Simule l'action d'un humain.

    L'humain ne suit pas une stratégie optimale pure. Il peut être loose, tight,
    agressif, débutant ou régulier. La fatigue augmente aussi légèrement les erreurs.
    """
    progress = hand_index / max(total_hands - 1, 1)
    freqs = gto_freqs.copy()

    # Plus la session avance, plus les fréquences humaines se bruitent.
    noise_level = 0.10 + 0.10 * progress

    if profile == "recreational_loose":
        freqs["fold"] *= 0.72
        freqs["call"] *= 1.28
        freqs["raise"] *= 1.08
        noise_level += 0.05
    elif profile == "tight_regular":
        freqs["fold"] *= 1.22
        freqs["call"] *= 0.86
        freqs["raise"] *= 0.92
    elif profile == "aggressive_regular":
        freqs["fold"] *= 0.92
        freqs["call"] *= 0.78
        freqs["raise"] *= 1.35
    elif profile == "beginner":
        freqs["fold"] *= 0.88
        freqs["call"] *= 1.40
        freqs["raise"] *= 0.82
        noise_level += 0.10
    elif profile == "solid_regular":
        noise_level -= 0.03

    # Les mains marginales sont plus difficiles pour les humains.
    if 42 <= hand_strength <= 62:
        noise_level += 0.07

    freqs = normalize_probabilities(freqs["fold"], freqs["call"], freqs["raise"])

    # Mélange avec une distribution plus bruitée.
    random_style = rng.dirichlet([1.4, 1.3, 1.2])
    final_probs = np.array([freqs["fold"], freqs["call"], freqs["raise"]])
    final_probs = (1 - noise_level) * final_probs + noise_level * random_style
    final_probs = final_probs / final_probs.sum()

    return str(rng.choice(ACTIONS, p=final_probs))


def choose_bot_action(gto_freqs: dict, hand_strength: float, context: dict, profile: str, rng: np.random.Generator) -> str:
    """
    Simule l'action d'un bot.

    Les bots suivent très fortement les fréquences théoriques. Certains sont stricts,
    d'autres ajoutent du bruit pour paraître plus humains.
    """
    freqs = gto_freqs.copy()

    if profile == "gto_strict":
        noise_level = 0.015
    elif profile == "humanized_gto":
        noise_level = 0.07
    else:  # tight_grinder_bot
        noise_level = 0.04
        freqs["fold"] *= 1.08
        freqs["call"] *= 0.92
        freqs["raise"] *= 1.03

    if 42 <= hand_strength <= 62:
        noise_level += 0.02

    freqs = normalize_probabilities(freqs["fold"], freqs["call"], freqs["raise"])
    base_probs = np.array([freqs["fold"], freqs["call"], freqs["raise"]])
    random_style = rng.dirichlet([1.1, 1.1, 1.1])
    final_probs = (1 - noise_level) * base_probs + noise_level * random_style
    final_probs = final_probs / final_probs.sum()

    return str(rng.choice(ACTIONS, p=final_probs))


def compute_l1_distance(action: str, gto_freqs: dict) -> float:
    """Distance L1 entre l'action jouée en one-hot et la distribution théorique."""
    player_distribution = {candidate: 1.0 if candidate == action else 0.0 for candidate in ACTIONS}
    return float(sum(abs(player_distribution[candidate] - gto_freqs[candidate]) for candidate in ACTIONS))


def simulate_decision_time(is_bot: int, action: str, hand_strength: float, context: dict, rng: np.random.Generator) -> float:
    """Simule un temps de décision en secondes."""
    if is_bot:
        base_time = rng.normal(loc=2.25, scale=0.38)
    else:
        base_time = rng.normal(loc=5.35, scale=1.55)

    if action == "raise":
        base_time += rng.normal(loc=0.38, scale=0.18)

    if 42 <= hand_strength <= 62 or context["facing_raise"]:
        base_time += rng.normal(loc=0.65 if not is_bot else 0.12, scale=0.20)

    return float(np.clip(base_time, 0.5, 22.0))


def simulate_bet_size_bb(is_bot: int, action: str, context: dict, rng: np.random.Generator) -> float:
    """Simule un sizing en grosses blindes."""
    if action == "fold":
        return 0.0

    amount_to_call = context["amount_to_call_bb"]
    previous_action = context["previous_action"]

    if action == "call":
        return float(max(amount_to_call, BIG_BLIND_BB if previous_action in {"unopened", "limped"} else amount_to_call))

    # Raise / open raise / 3-bet / 4-bet simplifié.
    if previous_action in {"unopened", "limped"}:
        loc = 2.45 if is_bot else 2.55. # il s'agit du premier raise, le bot relance de 2.45 en moyenne sinon 2.55 pour l'humain
        scale = 0.15 if is_bot else 0.38 # c'est ici que la différence est flagrante
        return float(np.clip(rng.normal(loc=loc, scale=scale), 2.0, 4.5))

    if previous_action == "open_raise":
        loc = 8.2 if is_bot else 8.5
        scale = 0.45 if is_bot else 1.25 # pour un trois bet, l'écart type est plus grand ce qui coincide avec la difficulté de sizer ce coup
        return float(np.clip(rng.normal(loc=loc, scale=scale), 5.5, 13.0))
    # ici on est dans la situation du 4 bet, plus rare et bcp plus imprécis ( dans la réalité, ça part souvent à tapis)
    loc = 19.0 if is_bot else 20.0
    scale = 1.0 if is_bot else 3.0
    return float(np.clip(rng.normal(loc=loc, scale=scale), 12.0, 35.0))


def action_entropy(actions: pd.Series) -> float:
    """Calcule une entropie normalisée entre 0 et 1 sur fold/call/raise."""
    probabilities = actions.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return float(entropy / np.log2(len(ACTIONS)))


def simulate_player_hands(player_id: str, is_bot: int, n_hands: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simule toutes les décisions préflop d'un joueur."""
    rows = []
    profile = str(rng.choice(BOT_PROFILES if is_bot else HUMAN_PROFILES))

    for hand_index in range(n_hands):
        card_1, card_2 = draw_two_cards(rng)
        hand_class = get_hand_class(card_1, card_2)
        hand_strength = compute_hand_strength(hand_class)
        hand_family = classify_hand_family(hand_class, hand_strength)
        position = str(rng.choice(POSITIONS))
        context = simulate_preflop_context(position, rng)
        gto_freqs = get_gto_frequencies(hand_strength, position, context)
        gto_action = recommended_action(gto_freqs)

        if is_bot:
            player_action = choose_bot_action(gto_freqs, hand_strength, context, profile, rng)
        else:
            player_action = choose_human_action(gto_freqs, hand_strength, context, hand_index, n_hands, profile, rng)

        chosen_action_probability = gto_freqs[player_action]
        gto_l1_distance = compute_l1_distance(player_action, gto_freqs)
        decision_time = simulate_decision_time(is_bot, player_action, hand_strength, context, rng)
        bet_size_bb = simulate_bet_size_bb(is_bot, player_action, context, rng)

        rows.append(
            {
                "player_id": player_id,
                "hand_id": hand_index + 1,
                "is_bot": is_bot,
                "player_profile": profile,
                "table_size": context["table_size"],
                "position": position,
                "small_blind_bb": context["small_blind_bb"],
                "big_blind_bb": context["big_blind_bb"],
                "posted_blind_bb": context["posted_blind_bb"],
                "stack_bb": context["stack_bb"],
                "effective_stack_bb": context["effective_stack_bb"],
                "previous_action": context["previous_action"],
                "facing_raise": context["facing_raise"],
                "facing_3bet": context["facing_3bet"],
                "pot_size_bb": context["pot_size_bb"],
                "current_bet_bb": context["current_bet_bb"],
                "amount_to_call_bb": context["amount_to_call_bb"],
                "pot_odds": context["pot_odds"],
                "card_1": card_1,
                "card_2": card_2,
                "hand_class": hand_class,
                "hand_family": hand_family,
                "hand_strength": hand_strength,
                "gto_fold_probability": gto_freqs["fold"],
                "gto_call_probability": gto_freqs["call"],
                "gto_raise_probability": gto_freqs["raise"],
                "gto_action": gto_action,
                "player_action": player_action,
                "chosen_action_probability": chosen_action_probability,
                "gto_l1_distance": gto_l1_distance,
                "is_gto_correct": int(player_action == gto_action),
                "decision_time": decision_time,
                "bet_size_bb": bet_size_bb,
                "bet_size": bet_size_bb,
            }
        )

    return pd.DataFrame(rows)


def _safe_mean(series: pd.Series) -> float:
    """Retourne une moyenne sûre, égale à 0 si la série est vide."""
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def _safe_rate(df: pd.DataFrame, condition_column: str = "is_gto_correct") -> float:
    """Retourne une moyenne sûre, égale à 0 si le DataFrame est vide."""
    if len(df) == 0:
        return 0.0
    return float(df[condition_column].mean())


def aggregate_player_features(player_hands: pd.DataFrame) -> dict:
    """Agrège les décisions préflop d'un joueur en variables de modèle."""
    actions = player_hands["player_action"]
    hands_played = len(player_hands)

    vpip = float((actions != "fold").mean())
    pfr = float((actions == "raise").mean())

    calls = int((actions == "call").sum())
    raises = int((actions == "raise").sum())
    af = float(raises / max(calls, 1))

    played_hands = player_hands[player_hands["player_action"] != "fold"]
    bet_size_mean = _safe_mean(played_hands["bet_size_bb"])
    bet_size_std = float(played_hands["bet_size_bb"].std(ddof=0)) if len(played_hands) else 0.0

    weak_hands = player_hands[player_hands["hand_strength"] < 45]
    trash_hands = player_hands[player_hands["hand_family"] == "trash"]
    premium_hands = player_hands[player_hands["hand_strength"] >= 78]
    strong_hands = player_hands[player_hands["hand_strength"] >= 70]
    marginal_hands = player_hands[(player_hands["hand_strength"] >= 42) & (player_hands["hand_strength"] <= 62)]

    weak_hand_play_rate = float((weak_hands["player_action"] != "fold").mean()) if len(weak_hands) else 0.0
    trash_hand_vpip = float((trash_hands["player_action"] != "fold").mean()) if len(trash_hands) else 0.0
    premium_hand_play_rate = float((premium_hands["player_action"] != "fold").mean()) if len(premium_hands) else 0.0
    strong_hand_aggression_rate = float((strong_hands["player_action"] == "raise").mean()) if len(strong_hands) else 0.0
    marginal_hand_error_rate = float(1 - marginal_hands["is_gto_correct"].mean()) if len(marginal_hands) else 0.0

    fold_spots = player_hands[player_hands["gto_action"] == "fold"]
    call_spots = player_hands[player_hands["gto_action"] == "call"]
    raise_spots = player_hands[player_hands["gto_action"] == "raise"]
    open_spots = player_hands[player_hands["previous_action"].isin(["unopened", "limped"])]
    defense_spots = player_hands[player_hands["previous_action"] == "open_raise"]
    threebet_spots = player_hands[player_hands["previous_action"] == "three_bet"]
    blind_defense_spots = player_hands[(player_hands["position"].isin(["SB", "BB"])) & (player_hands["facing_raise"] == 1)]
    sb_steal_spots = player_hands[(player_hands["position"] == "SB") & (player_hands["previous_action"] == "unopened")]
    pot_odds_call_spots = player_hands[(player_hands["amount_to_call_bb"] > 0) & (player_hands["pot_odds"] <= 0.30)]

    first_half = player_hands.iloc[: hands_played // 2]
    second_half = player_hands.iloc[hands_played // 2 :]
    fatigue_slope = float(second_half["is_gto_correct"].mean() - first_half["is_gto_correct"].mean())

    gto_similarity = float(player_hands["is_gto_correct"].mean())

    return {
        "vpip": vpip,
        "pfr": pfr,
        "af": af,
        "action_entropy": action_entropy(actions),
        "decision_time_mean": float(player_hands["decision_time"].mean()),
        "decision_time_std": float(player_hands["decision_time"].std(ddof=0)),
        "bet_size_mean": bet_size_mean,
        "bet_size_std": bet_size_std,
        "gto_similarity": gto_similarity,
        "gto_deviation_rate": float(1 - gto_similarity),
        "mean_gto_action_probability": float(player_hands["chosen_action_probability"].mean()),
        "mean_l1_gto_distance": float(player_hands["gto_l1_distance"].mean()),
        "std_l1_gto_distance": float(player_hands["gto_l1_distance"].std(ddof=0)),
        "avg_hand_strength": float(player_hands["hand_strength"].mean()),
        "weak_hand_play_rate": weak_hand_play_rate,
        "trash_hand_vpip": trash_hand_vpip,
        "premium_hand_play_rate": premium_hand_play_rate,
        "strong_hand_aggression_rate": strong_hand_aggression_rate,
        "marginal_hand_error_rate": marginal_hand_error_rate,
        "gto_fold_follow_rate": _safe_rate(fold_spots),
        "gto_call_follow_rate": _safe_rate(call_spots),
        "gto_raise_follow_rate": _safe_rate(raise_spots),
        "open_raise_accuracy": _safe_rate(open_spots),
        "defense_accuracy": _safe_rate(defense_spots),
        "threebet_accuracy": _safe_rate(threebet_spots),
        "blind_defense_rate": float((blind_defense_spots["player_action"] != "fold").mean()) if len(blind_defense_spots) else 0.0,
        "blind_defense_accuracy": _safe_rate(blind_defense_spots),
        "sb_steal_attempt_rate": float((sb_steal_spots["player_action"] == "raise").mean()) if len(sb_steal_spots) else 0.0,
        "pot_odds_call_accuracy": _safe_rate(pot_odds_call_spots),
        "hands_played": int(hands_played),
        "sessions_played": int(np.ceil(hands_played / 60)),
        "fatigue_slope": fatigue_slope,
    }


def generate_dataset(n_players: int, bot_ratio: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère le dataset joueur agrégé et le dataset détaillé main par main."""
    rng = np.random.default_rng(random_state)

    n_bots = int(n_players * bot_ratio)
    n_humans = n_players - n_bots

    labels = np.array([0] * n_humans + [1] * n_bots)
    rng.shuffle(labels)

    player_rows = []
    hand_rows = []

    for player_index, is_bot in enumerate(labels):
        player_id = f"P{str(player_index + 1).zfill(5)}"

        # Même ordre de grandeur de volume entre humains et bots : le modèle doit
        # surtout apprendre les décisions, pas seulement le nombre de mains.
        n_hands = int(rng.integers(low=50, high=91))

        player_hands = simulate_player_hands(player_id, int(is_bot), n_hands, rng)
        features = aggregate_player_features(player_hands)

        player_rows.append({"player_id": player_id, "is_bot": int(is_bot), **features})
        hand_rows.extend(player_hands.to_dict("records"))

    players_dataset = pd.DataFrame(player_rows)
    hands_dataset = pd.DataFrame(hand_rows)

    return players_dataset, hands_dataset


def save_dataset(players_dataset: pd.DataFrame, output_path) -> None:
    """Sauvegarde le dataset agrégé par joueur."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    players_dataset.to_csv(output_path, index=False)


def save_hands_dataset(hands_dataset: pd.DataFrame, output_path) -> None:
    """Sauvegarde le dataset détaillé main par main."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hands_dataset.to_csv(output_path, index=False)
