"""Synthetic demo data. **Not real players, books or prices.**

Simulates a Liga-Pro-style league (small groups playing round robins every
day) and three fictional bookmakers so the whole pipeline can be tried
without a paid odds feed:

* ``DemoSharp`` – low margin, accurate, sharpens further by the close.
* ``DemoSoftA`` – higher margin, noisier strength estimates, prices
  derivatives from a point model that ignores game-to-game variation.
* ``DemoSoftB`` – decent match-winner line but hangs derivatives with crude
  rules of thumb (the kind of inconsistency the scanner is built to catch).

Results from the simulated games feed the ratings; the scanner never sees the
hidden true strengths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from . import markets as mk
from .data import Quote, Result, write_odds, write_results
from .markov import _N_LOSER, _WINNER_PTS, game_outcomes, match_model
from .odds import apply_margin, sigmoid

TRUE_SIGMA = 0.10


@dataclass
class BookSpec:
    name: str
    open_noise: float   # error in the book's strength estimate (logit point units; 0.01 ~ 1.8pp on the match price)
    close_noise: float
    margin: float
    style: str  # "model", "model_naive", "lazy"
    markets: tuple[str, ...]


BOOKS = (
    BookSpec("DemoSharp", 0.010, 0.005, 0.035, "model",
             (mk.ML, mk.GAME_HCP, mk.GAMES_TOTAL, mk.POINTS_TOTAL)),
    BookSpec("DemoSoftA", 0.015, 0.012, 0.065, "model_naive",
             (mk.ML, mk.GAME_HCP, mk.GAMES_TOTAL, mk.CORRECT_SCORE)),
    BookSpec("DemoSoftB", 0.012, 0.010, 0.07, "lazy",
             (mk.ML, mk.GAME_HCP, mk.GAMES_TOTAL, mk.G1_ML)),
)


def _simulate_match(rng: np.random.Generator, d: float, best_of: int) -> mk.MatchScore:
    w = mk.games_to_win(best_of)
    ga = gb = 0
    games = []
    while ga < w and gb < w:
        p = sigmoid(d + rng.normal(0, TRUE_SIGMA))
        win_a, win_b = game_outcomes(p)
        probs = np.concatenate([win_a, win_b])
        k = int(rng.choice(len(probs), p=probs / probs.sum()))
        a_wins = k < _N_LOSER
        loser = k % _N_LOSER
        wp = int(_WINNER_PTS[loser])
        games.append((wp, loser) if a_wins else (loser, wp))
        ga += a_wins
        gb += not a_wins
    return mk.MatchScore(ga, gb, tuple(games))


def _book_groups(book: BookSpec, d_est: float, best_of: int) -> list[tuple[str, float | None, list[str], list[float]]]:
    """Fair probabilities the book believes in, per market group (A-perspective lines)."""
    p = sigmoid(d_est)
    sigma = TRUE_SIGMA if book.style == "model" else 0.0
    model = match_model(p, best_of, sigma)
    m = model.p_match
    fav = "A" if m >= 0.5 else "B"
    fav_m = max(m, 1 - m)
    out = []
    for market in book.markets:
        if market == mk.ML:
            out.append((market, None, ["A", "B"], [m, 1 - m]))
        elif market == mk.G1_ML:
            g = model.p_game if book.style != "lazy" else 0.5 + (m - 0.5) * 0.6
            out.append((market, None, ["A", "B"], [g, 1 - g]))
        elif market == mk.GAME_HCP:
            a_line = -1.5 if fav == "A" else 1.5
            if book.style == "lazy":
                pf = max(0.05, fav_m - 0.2)
                pa = pf if fav == "A" else 1 - pf
            else:
                pa = model.prob(market, "A", a_line)
            out.append((market, a_line, ["A", "B"], [pa, 1 - pa]))
        elif market == mk.GAMES_TOTAL:
            line = 3.5
            po = 0.74 - 0.4 * abs(m - 0.5) if book.style == "lazy" else model.prob(market, "OVER", line)
            out.append((market, line, ["OVER", "UNDER"], [po, 1 - po]))
        elif market == mk.POINTS_TOTAL:
            line = model.median_line(market)
            po = model.prob(market, "OVER", line)
            out.append((market, line, ["OVER", "UNDER"], [po, 1 - po]))
        elif market == mk.CORRECT_SCORE:
            sels = mk.correct_score_selections(best_of)
            out.append((market, None, sels, [max(model.prob(market, s), 1e-3) for s in sels]))
    return out


def _quotes_for(book: BookSpec, d_est: float, base: dict, snapshot: str) -> list[Quote]:
    quotes = []
    for market, a_line, sels, probs in _book_groups(book, d_est, base["best_of"]):
        margin = book.margin * (1.6 if market == mk.CORRECT_SCORE else 1.0)
        total = sum(probs)
        prices = apply_margin([pr / total for pr in probs], margin)
        for sel, price in zip(sels, prices):
            line = mk.selection_line(market, sel, a_line)
            quotes.append(Quote(book=book.name, market=market, selection=sel, line=line,
                                price=max(1.01, math.floor(price * 100) / 100), snapshot=snapshot, **base))
    return quotes


def generate(seed: int = 7, n_players: int = 40, days: int = 30, groups_per_day: int = 8, group_size: int = 5,
             upcoming_groups: int = 6, start: datetime | None = None) -> tuple[list[Result], list[Quote]]:
    rng = np.random.default_rng(seed)
    if start is None:  # history ends yesterday; the upcoming fixtures are today
        today = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)
        start = today - timedelta(days=days)
    names = [f"Demo Player {i:02d}" for i in range(1, n_players + 1)]
    theta = rng.normal(0.0, 0.10, n_players)
    results: list[Result] = []
    quotes: list[Quote] = []
    match_no = 0

    for day in range(days + 1):
        upcoming = day == days
        theta = theta + rng.normal(0.0, 0.006, n_players)
        for grp in range(upcoming_groups if upcoming else groups_per_day):
            players = rng.choice(n_players, size=group_size, replace=False)
            league = "Demo League A" if grp % 2 == 0 else "Demo League B"
            t0 = start + timedelta(days=day, hours=grp)
            for k, (i, j) in enumerate(combinations(players, 2)):
                match_no += 1
                when = t0 + timedelta(minutes=20 * k)
                base = dict(match_id=f"demo-{match_no:05d}", start_time=when, tournament=league,
                            player_a=names[i], player_b=names[j], best_of=5)
                d_true = theta[i] - theta[j]
                form = rng.normal(0.0, 0.02)  # match-day form: unknown at the open, partly priced by the close
                for book in BOOKS:
                    noise_open = rng.normal(0.0, book.open_noise)
                    if upcoming:
                        quotes += _quotes_for(book, d_true + noise_open, base, "current")
                    else:
                        quotes += _quotes_for(book, d_true + noise_open, base, "open")
                        close_est = d_true + 0.7 * form + rng.normal(0.0, book.close_noise)
                        quotes += _quotes_for(book, close_est, base, "close")
                if not upcoming:
                    score = _simulate_match(rng, d_true + form, 5)
                    results.append(Result(base["match_id"], when, league, names[i], names[j], score, 5))
    return results, quotes


def write_demo(directory: str | Path, **kwargs) -> tuple[Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    results, quotes = generate(**kwargs)
    rp, op = out / "results.csv", out / "odds.csv"
    write_results(rp, results)
    write_odds(op, quotes)
    return rp, op
