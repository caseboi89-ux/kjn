"""Market vocabulary, score parsing and bet settlement.

Canonical markets (``line`` is always from the *selection's* point of view for
handicaps, e.g. ``B +1.5``):

==================  ======================  ===================================
market              selections              meaning
==================  ======================  ===================================
ML                  A, B                    match winner
GAME_HCP            A, B (+line)            game ("set") handicap
GAMES_TOTAL         OVER, UNDER (+line)     total games played
CORRECT_SCORE       "3-1", "0-3", ...       exact game score, A's games first
POINTS_TOTAL        OVER, UNDER (+line)     total points in the match
POINTS_HCP          A, B (+line)            point handicap over the match
G1_ML               A, B                    first-game winner
G1_POINTS_TOTAL     OVER, UNDER (+line)     total points in game 1
==================  ======================  ===================================
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ML = "ML"
GAME_HCP = "GAME_HCP"
GAMES_TOTAL = "GAMES_TOTAL"
CORRECT_SCORE = "CORRECT_SCORE"
POINTS_TOTAL = "POINTS_TOTAL"
POINTS_HCP = "POINTS_HCP"
G1_ML = "G1_ML"
G1_POINTS_TOTAL = "G1_POINTS_TOTAL"

MARKETS = (ML, GAME_HCP, GAMES_TOTAL, CORRECT_SCORE, POINTS_TOTAL, POINTS_HCP, G1_ML, G1_POINTS_TOTAL)
SIDE_MARKETS = {ML, G1_ML}
HCP_MARKETS = {GAME_HCP, POINTS_HCP}
TOTAL_MARKETS = {GAMES_TOTAL, POINTS_TOTAL, G1_POINTS_TOTAL}
LINE_MARKETS = HCP_MARKETS | TOTAL_MARKETS
POINT_MARKETS = {POINTS_TOTAL, POINTS_HCP, G1_POINTS_TOTAL}

MARKET_LABELS = {
    ML: "Match winner",
    GAME_HCP: "Game handicap",
    GAMES_TOTAL: "Total games",
    CORRECT_SCORE: "Correct score",
    POINTS_TOTAL: "Total points",
    POINTS_HCP: "Point handicap",
    G1_ML: "Game 1 winner",
    G1_POINTS_TOTAL: "Game 1 total points",
}

_MARKET_ALIASES = {
    "ml": ML, "moneyline": ML, "match_winner": ML, "winner": ML, "h2h": ML, "match": ML, "12": ML,
    "game_handicap": GAME_HCP, "set_handicap": GAME_HCP, "sets_handicap": GAME_HCP,
    "games_handicap": GAME_HCP, "handicap": GAME_HCP, "game_hcp": GAME_HCP, "spread": GAME_HCP,
    "games_total": GAMES_TOTAL, "total_games": GAMES_TOTAL, "total_sets": GAMES_TOTAL,
    "sets_total": GAMES_TOTAL, "totals": GAMES_TOTAL,
    "correct_score": CORRECT_SCORE, "exact_score": CORRECT_SCORE, "cs": CORRECT_SCORE,
    "points_total": POINTS_TOTAL, "total_points": POINTS_TOTAL,
    "points_hcp": POINTS_HCP, "points_handicap": POINTS_HCP, "point_handicap": POINTS_HCP,
    "g1_ml": G1_ML, "first_game_winner": G1_ML, "game1_winner": G1_ML, "set1_winner": G1_ML,
    "g1_points_total": G1_POINTS_TOTAL, "first_game_total_points": G1_POINTS_TOTAL,
    "game1_total_points": G1_POINTS_TOTAL,
}

_SELECTION_ALIASES = {
    "a": "A", "home": "A", "1": "A", "p1": "A", "player_a": "A", "player1": "A",
    "b": "B", "away": "B", "2": "B", "p2": "B", "player_b": "B", "player2": "B",
    "over": "OVER", "o": "OVER", "under": "UNDER", "u": "UNDER",
}


def normalize_market(value: str) -> str:
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if key.upper() in MARKETS:
        return key.upper()
    if key in _MARKET_ALIASES:
        return _MARKET_ALIASES[key]
    raise ValueError(f"unknown market {value!r}")


def normalize_selection(value: str, player_a: str = "", player_b: str = "") -> str:
    text = value.strip()
    key = text.lower()
    if key in _SELECTION_ALIASES:
        return _SELECTION_ALIASES[key]
    if player_a and key == player_a.strip().lower():
        return "A"
    if player_b and key == player_b.strip().lower():
        return "B"
    m = re.fullmatch(r"(\d)\s*[-:]\s*(\d)", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    raise ValueError(f"unknown selection {value!r}")


def swap_selection(market: str, selection: str) -> str:
    """Selection name after swapping which player is A and which is B."""
    if selection == "A":
        return "B"
    if selection == "B":
        return "A"
    if market == CORRECT_SCORE:
        a, b = selection.split("-")
        return f"{b}-{a}"
    return selection


def games_to_win(best_of: int) -> int:
    if best_of % 2 == 0 or best_of < 1:
        raise ValueError(f"best_of must be odd, got {best_of}")
    return best_of // 2 + 1


def correct_score_selections(best_of: int) -> list[str]:
    w = games_to_win(best_of)
    return [f"{w}-{k}" for k in range(w)] + [f"{k}-{w}" for k in range(w)]


def group_key(market: str, selection: str, line: float | None) -> tuple:
    """Key identifying the set of selections that make up one complete market.

    Handicap lines are converted to player A's perspective so that ``A -1.5``
    and ``B +1.5`` land in the same group.
    """
    if market in HCP_MARKETS:
        a_line = line if selection == "A" else -line  # type: ignore[operator]
        return (market, round(a_line, 3))
    if market in TOTAL_MARKETS:
        return (market, round(line, 3))  # type: ignore[arg-type]
    return (market, None)


def group_selections(market: str, best_of: int) -> list[str]:
    if market in SIDE_MARKETS or market in HCP_MARKETS:
        return ["A", "B"]
    if market in TOTAL_MARKETS:
        return ["OVER", "UNDER"]
    if market == CORRECT_SCORE:
        return correct_score_selections(best_of)
    raise ValueError(market)


def selection_line(market: str, selection: str, group_line: float | None) -> float | None:
    """Line for ``selection`` given a group's (A-perspective) line."""
    if market in HCP_MARKETS:
        return group_line if selection == "A" else -group_line  # type: ignore[operator]
    return group_line


def describe(market: str, selection: str, line: float | None, player_a: str = "A", player_b: str = "B") -> str:
    who = {"A": player_a, "B": player_b}
    if market == ML:
        return f"{who[selection]} to win"
    if market == G1_ML:
        return f"{who[selection]} to win game 1"
    if market in HCP_MARKETS:
        unit = "games" if market == GAME_HCP else "points"
        return f"{who[selection]} {line:+g} {unit}"
    if market in TOTAL_MARKETS:
        unit = {GAMES_TOTAL: "games", POINTS_TOTAL: "points", G1_POINTS_TOTAL: "points in game 1"}[market]
        return f"{selection.title()} {line:g} {unit}"
    if market == CORRECT_SCORE:
        return f"{player_a} {selection} {player_b}"
    return f"{market} {selection} {line}"


# --------------------------------------------------------------------------- scores

_PAIR = re.compile(r"(\d+)\s*[-:]\s*(\d+)")


@dataclass(frozen=True)
class MatchScore:
    """Final score from player A's perspective."""

    games_a: int
    games_b: int
    game_points: tuple[tuple[int, int], ...] = ()

    @property
    def has_points(self) -> bool:
        return bool(self.game_points)

    @property
    def points_a(self) -> int:
        return sum(a for a, _ in self.game_points)

    @property
    def points_b(self) -> int:
        return sum(b for _, b in self.game_points)

    def swapped(self) -> "MatchScore":
        return MatchScore(self.games_b, self.games_a, tuple((b, a) for a, b in self.game_points))

    def __str__(self) -> str:
        if self.game_points:
            return " ".join(f"{a}-{b}" for a, b in self.game_points)
        return f"{self.games_a}-{self.games_b}"


def _valid_game(a: int, b: int) -> bool:
    hi, lo = max(a, b), min(a, b)
    if hi < 11:
        return False
    if hi == 11:
        return lo <= 9
    return hi - lo == 2


def parse_score(text: str) -> MatchScore:
    """Parse ``"11-7 9-11 11-5 11-8"``, ``"3-1"`` or ``"3-1 (11-7, 9-11, 11-5, 11-8)"``."""
    pairs = [(int(a), int(b)) for a, b in _PAIR.findall(text or "")]
    if not pairs:
        raise ValueError(f"no score found in {text!r}")
    summary = None
    if max(pairs[0]) < 11:
        summary, pairs = pairs[0], pairs[1:]
    for a, b in pairs:
        if not _valid_game(a, b):
            raise ValueError(f"invalid game score {a}-{b} in {text!r}")
    if pairs:
        ga = sum(1 for a, b in pairs if a > b)
        gb = len(pairs) - ga
        if summary and summary != (ga, gb):
            raise ValueError(f"summary {summary} disagrees with game scores in {text!r}")
        return MatchScore(ga, gb, tuple(pairs))
    assert summary is not None
    if summary[0] == summary[1] or max(summary) > 4:
        raise ValueError(f"invalid match score {text!r}")
    return MatchScore(summary[0], summary[1])


def _cmp(value: float, line: float) -> str:
    if abs(value - line) < 1e-9:
        return "push"
    return "win" if value > line else "lose"


def settle(market: str, selection: str, line: float | None, score: MatchScore) -> str | None:
    """Return ``"win"``, ``"push"`` or ``"lose"``; ``None`` if the score lacks the detail."""
    if market in POINT_MARKETS and not score.has_points:
        return None
    if market == G1_ML and not score.has_points:
        return None
    if market == ML:
        a_won = score.games_a > score.games_b
        return "win" if a_won == (selection == "A") else "lose"
    if market == G1_ML:
        a, b = score.game_points[0]
        return "win" if (a > b) == (selection == "A") else "lose"
    if market == CORRECT_SCORE:
        return "win" if selection == f"{score.games_a}-{score.games_b}" else "lose"
    if market in HCP_MARKETS:
        if market == GAME_HCP:
            diff = score.games_a - score.games_b
        else:
            diff = score.points_a - score.points_b
        if selection == "B":
            diff = -diff
        return _cmp(diff + line, 0.0)  # type: ignore[operator]
    if market in TOTAL_MARKETS:
        if market == GAMES_TOTAL:
            total = score.games_a + score.games_b
        elif market == POINTS_TOTAL:
            total = score.points_a + score.points_b
        else:
            total = sum(score.game_points[0])
        res = _cmp(total, line)  # type: ignore[arg-type]
        if res == "push" or selection == "OVER":
            return res
        return "win" if res == "lose" else "lose"
    raise ValueError(f"cannot settle {market}/{selection}")
