"""Exact point -> game -> match probability model for table tennis.

A game is first to 11 points, win by two. A match is best of 5 (first to 3
games) or best of 7. Given ``p`` – the probability that player A wins any one
point – every market (winner, game handicap, totals, correct score, points) has
a closed-form fair price.

``sigma`` adds game-to-game variation: each game draws its own point
probability ``sigmoid(logit(p) + N(0, sigma))``. That widens the spread of
game scores the way real form swings do; :func:`ttedge.backtest.calibrate_sigma`
fits it from historical point scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import markets as mk
from .odds import logit, sigmoid

DEUCE_CAP = 25  # deuce pairs modelled after 10-10; the remaining tail mass is < 1e-7
_N_LOSER = 11 + DEUCE_CAP  # loser point counts 0..9 plus 10..10+DEUCE_CAP
_LOSER_PTS = np.arange(_N_LOSER)
_WINNER_PTS = np.where(_LOSER_PTS <= 9, 11, _LOSER_PTS + 2)
_GAME_TOTAL = _WINNER_PTS + _LOSER_PTS
_GAME_MARGIN = _WINNER_PTS - _LOSER_PTS
_MAX_MARGIN = 11
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(11)
_GH_W = _GH_W / math.sqrt(math.pi)
_COMB = np.array([math.comb(10 + k, k) for k in range(10)], dtype=float)
_C2010 = float(math.comb(20, 10))
_K10 = np.arange(10)


def _win_by_loser_points(p: float) -> np.ndarray:
    """P(A wins the game and B scores exactly l points), for l = 0 .. 10+DEUCE_CAP."""
    q = 1.0 - p
    out = np.empty(_N_LOSER)
    k = np.arange(10)
    out[:10] = _COMB * p ** 11 * q ** k
    base = _C2010 * p ** 10 * q ** 10 * p * p
    out[10:] = base * (2.0 * p * q) ** np.arange(DEUCE_CAP + 1)
    return out


def _point_probs(p: float, sigma: float) -> list[tuple[float, float]]:
    if sigma <= 0:
        return [(p, 1.0)]
    lp = logit(p)
    return [(sigmoid(lp + math.sqrt(2.0) * sigma * x), w) for x, w in zip(_GH_X, _GH_W)]


def game_outcomes(p: float, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (win_a, win_b): probability of each loser-point count when A / B wins a game."""
    win_a = np.zeros(_N_LOSER)
    win_b = np.zeros(_N_LOSER)
    for pp, w in _point_probs(p, sigma):
        win_a += w * _win_by_loser_points(pp)
        win_b += w * _win_by_loser_points(1.0 - pp)
    total = win_a.sum() + win_b.sum()
    return win_a / total, win_b / total


def game_win_prob(p: float, sigma: float = 0.0) -> float:
    """Probability A wins a single game. Closed form, no truncation."""
    nodes = _point_probs(p, sigma)
    pp = np.array([n[0] for n in nodes])
    w = np.array([n[1] for n in nodes])
    q = 1.0 - pp
    reg = (pp[:, None] ** 11 * q[:, None] ** _K10) @ _COMB
    deuce = _C2010 * pp ** 10 * q ** 10 * pp * pp / (pp * pp + q * q)
    return float(np.dot(w, reg + deuce))


def match_win_prob_from_game(g: float, best_of: int = 5) -> float:
    w = mk.games_to_win(best_of)
    return sum(math.comb(w - 1 + k, k) * g ** w * (1.0 - g) ** k for k in range(w))


def match_win_prob(p: float, best_of: int = 5, sigma: float = 0.0) -> float:
    return match_win_prob_from_game(game_win_prob(p, sigma), best_of)


def _game_win_prob_vec(p: np.ndarray, sigma: float) -> np.ndarray:
    if sigma > 0:
        lp = np.log(p / (1 - p))[:, None] + math.sqrt(2.0) * sigma * _GH_X[None, :]
        pp = 1.0 / (1.0 + np.exp(-lp))
        w = _GH_W[None, :]
    else:
        pp = p[:, None]
        w = np.ones((1, 1))
    q = 1.0 - pp
    reg = np.sum(_COMB * pp[..., None] ** 11 * q[..., None] ** _K10, axis=-1)
    deuce = _C2010 * pp ** 12 * q ** 10 / (pp * pp + q * q)
    return np.sum(w * (reg + deuce), axis=1)


@lru_cache(maxsize=64)
def _inverse_table(best_of: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    lp = np.linspace(-2.5, 2.5, 5001)  # logit(p) grid: p in [0.076, 0.924] covers m in [1e-6, 1 - 1e-6]
    p = 1.0 / (1.0 + np.exp(-lp))
    g = _game_win_prob_vec(p, sigma)
    w = mk.games_to_win(best_of)
    m = sum(math.comb(w - 1 + k, k) * g ** w * (1.0 - g) ** k for k in range(w))
    m = np.clip(m, 1e-15, 1 - 1e-15)
    return np.log(m / (1 - m)), lp


def point_prob_from_match_prob(m: float, best_of: int = 5, sigma: float = 0.0) -> float:
    """Invert :func:`match_win_prob`: point-win probability implied by a match price."""
    m = min(max(m, 1e-6), 1 - 1e-6)
    lm_grid, lp_grid = _inverse_table(best_of, round(sigma, 4))
    lp = float(np.interp(math.log(m / (1 - m)), lm_grid, lp_grid))
    p = 1.0 / (1.0 + math.exp(-lp))
    # one secant-free Newton polish on the exact function
    eps = 1e-6
    f0 = match_win_prob(p, best_of, sigma) - m
    slope = (match_win_prob(p + eps, best_of, sigma) - match_win_prob(p - eps, best_of, sigma)) / (2 * eps)
    if slope > 0:
        p = min(max(p - f0 / slope, 1e-4), 1 - 1e-4)
    return p


def _threshold(values: np.ndarray, probs: np.ndarray, line: float) -> tuple[float, float, float]:
    """(P(X > line), P(X == line), P(X < line))."""
    over = float(probs[values > line + 1e-9].sum())
    push = float(probs[np.abs(values - line) <= 1e-9].sum())
    under = float(probs[values < line - 1e-9].sum())
    return over, push, under


@dataclass
class MatchModel:
    """All market probabilities for one match given A's point-win probability."""

    p_point: float
    best_of: int = 5
    sigma: float = 0.0
    p_game: float = field(init=False)
    p_match: float = field(init=False)
    scores: dict[tuple[int, int], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.p_point < 1.0:
            raise ValueError("p_point must be in (0, 1)")
        w = mk.games_to_win(self.best_of)
        win_a, win_b = game_outcomes(self.p_point, self.sigma)
        self.p_game = float(win_a.sum())

        n_tot = int(_GAME_TOTAL.max()) + 1
        tot_a = np.bincount(_GAME_TOTAL, weights=win_a, minlength=n_tot)
        tot_b = np.bincount(_GAME_TOTAL, weights=win_b, minlength=n_tot)
        dif_a = np.zeros(2 * _MAX_MARGIN + 1)
        dif_b = np.zeros(2 * _MAX_MARGIN + 1)
        np.add.at(dif_a, _MAX_MARGIN + _GAME_MARGIN, win_a)
        np.add.at(dif_b, _MAX_MARGIN - _GAME_MARGIN, win_b)
        self._g1_total = tot_a + tot_b

        # dynamic programme over game scores; each state carries the joint
        # (unnormalised) distribution of total points and of point difference
        layer = {(0, 0): (1.0, np.array([1.0]), np.array([1.0]))}
        terminal: dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]] = {}
        g, h = self.p_game, 1.0 - self.p_game
        while layer:
            nxt: dict[tuple[int, int], list] = {}
            for (a, b), (pr, tv, dv) in layer.items():
                for (na, nb), step, tvec, dvec in (
                    ((a + 1, b), g, tot_a, dif_a),
                    ((a, b + 1), h, tot_b, dif_b),
                ):
                    t_new = np.convolve(tv, tvec)
                    d_new = np.convolve(dv, dvec)
                    if (na, nb) in nxt:
                        cur = nxt[(na, nb)]
                        cur[0] += pr * step
                        cur[1] += t_new
                        cur[2] += d_new
                    else:
                        nxt[(na, nb)] = [pr * step, t_new, d_new]
            layer = {}
            for key, (pr, tv, dv) in nxt.items():
                if key[0] == w or key[1] == w:
                    terminal[key] = (pr, tv, dv)
                else:
                    layer[key] = (pr, tv, dv)

        self.scores = {k: v[0] for k, v in sorted(terminal.items())}
        self.p_match = sum(pr for (a, _), pr in self.scores.items() if a == w)

        max_t = max(len(v[1]) for v in terminal.values())
        totals = np.zeros(max_t)
        span = _MAX_MARGIN * self.best_of
        diffs = np.zeros(2 * span + 1)
        for (a, b), (_, tv, dv) in terminal.items():
            totals[: len(tv)] += tv
            off = _MAX_MARGIN * (a + b)  # index of zero difference in dv
            diffs[span - off: span - off + len(dv)] += dv
        self._totals = totals
        self._total_values = np.arange(max_t)
        self._diffs = diffs
        self._diff_values = np.arange(-span, span + 1)

    # ------------------------------------------------------------------ queries
    def game_diff_dist(self) -> tuple[np.ndarray, np.ndarray]:
        vals = np.array([a - b for a, b in self.scores])
        return vals, np.array(list(self.scores.values()))

    def games_total_dist(self) -> tuple[np.ndarray, np.ndarray]:
        vals = np.array([a + b for a, b in self.scores])
        return vals, np.array(list(self.scores.values()))

    def points_total_dist(self) -> tuple[np.ndarray, np.ndarray]:
        return self._total_values, self._totals

    def points_diff_dist(self) -> tuple[np.ndarray, np.ndarray]:
        return self._diff_values, self._diffs

    def g1_points_dist(self) -> tuple[np.ndarray, np.ndarray]:
        return np.arange(len(self._g1_total)), self._g1_total

    def expected_points(self) -> float:
        return float(np.dot(self._total_values, self._totals) / self._totals.sum())

    def median_line(self, market: str) -> float:
        """Half-point line closest to an even split (how books usually hang totals)."""
        vals, probs = {
            mk.POINTS_TOTAL: self.points_total_dist,
            mk.G1_POINTS_TOTAL: self.g1_points_dist,
            mk.GAMES_TOTAL: self.games_total_dist,
        }[market]()
        cdf = np.cumsum(probs) / probs.sum()
        idx = int(np.searchsorted(cdf, 0.5))
        lo = float(vals[idx]) - 0.5
        hi = lo + 1.0
        # pick whichever half-point line is closer to 50/50
        p_lo = float(probs[vals > lo].sum())
        p_hi = float(probs[vals > hi].sum())
        return lo if abs(p_lo - 0.5) <= abs(p_hi - 0.5) else hi

    def outcome(self, market: str, selection: str, line: float | None = None) -> tuple[float, float, float]:
        """Return (p_win, p_push, p_lose) for one selection."""
        if market == mk.ML:
            pa = self.p_match
            return (pa, 0.0, 1 - pa) if selection == "A" else (1 - pa, 0.0, pa)
        if market == mk.G1_ML:
            pa = self.p_game
            return (pa, 0.0, 1 - pa) if selection == "A" else (1 - pa, 0.0, pa)
        if market == mk.CORRECT_SCORE:
            a, b = (int(x) for x in selection.split("-"))
            pr = self.scores.get((a, b), 0.0)
            return pr, 0.0, 1 - pr
        if line is None:
            raise ValueError(f"{market} needs a line")
        if market in mk.HCP_MARKETS:
            vals, probs = self.game_diff_dist() if market == mk.GAME_HCP else self.points_diff_dist()
            probs = probs / probs.sum()
            if selection == "B":
                vals = -vals
            win, push, lose = _threshold(vals + line, probs, 0.0)
            return win, push, lose
        if market in mk.TOTAL_MARKETS:
            vals, probs = {
                mk.GAMES_TOTAL: self.games_total_dist,
                mk.POINTS_TOTAL: self.points_total_dist,
                mk.G1_POINTS_TOTAL: self.g1_points_dist,
            }[market]()
            probs = probs / probs.sum()
            over, push, under = _threshold(vals, probs, line)
            return (over, push, under) if selection == "OVER" else (under, push, over)
        raise ValueError(f"unknown market {market}")

    def prob(self, market: str, selection: str, line: float | None = None) -> float:
        """Win probability, with pushes removed (i.e. conditional on the bet being settled)."""
        win, _push, lose = self.outcome(market, selection, line)
        return win / (win + lose) if win + lose > 0 else 0.0

    def standard_lines(self) -> list[tuple[str, float | None]]:
        """The market/line pairs books typically offer for this format."""
        w = mk.games_to_win(self.best_of)
        lines: list[tuple[str, float | None]] = [(mk.ML, None), (mk.G1_ML, None)]
        lines += [(mk.GAME_HCP, -(k + 0.5)) for k in range(1, w)]
        lines += [(mk.GAMES_TOTAL, w + k + 0.5) for k in range(w - 1)]
        lines += [(mk.POINTS_TOTAL, self.median_line(mk.POINTS_TOTAL))]
        lines += [(mk.G1_POINTS_TOTAL, self.median_line(mk.G1_POINTS_TOTAL))]
        lines += [(mk.CORRECT_SCORE, None)]
        return lines

    def fair_table(self) -> list[dict]:
        """Fair probability and price for every selection of the standard markets."""
        rows = []
        for market, line in self.standard_lines():
            if market in mk.HCP_MARKETS:
                sels = [("A", line), ("B", -line)]  # type: ignore[operator]
            else:
                sels = [(s, line) for s in mk.group_selections(market, self.best_of)]
            for sel, sel_line in sels:
                win, push, lose = self.outcome(market, sel, sel_line)
                rows.append({
                    "market": market,
                    "selection": sel,
                    "line": sel_line,
                    "p_win": win,
                    "p_push": push,
                    "fair_price": (1 + lose / win) if win > 0 else None,
                })
        return rows


@lru_cache(maxsize=20000)
def _cached(p_round: float, best_of: int, sigma_round: float) -> MatchModel:
    return MatchModel(p_round, best_of, sigma_round)


def match_model(p_point: float, best_of: int = 5, sigma: float = 0.0) -> MatchModel:
    """Cached :class:`MatchModel` (p rounded to 1e-6: match prices move by < 0.001 points)."""
    return _cached(round(min(max(p_point, 1e-4), 1 - 1e-4), 6), best_of, round(sigma, 4))


def fit_point_prob(
    targets: list[tuple[str, str, float | None, float, float]],
    best_of: int = 5,
    sigma: float = 0.0,
) -> float | None:
    """Point probability that best explains a set of market probabilities.

    ``targets`` holds ``(market, selection, line, prob, weight)``, where ``prob`` is
    the push-free win probability. Minimises weighted squared error in logit space.
    """
    if not targets:
        return None

    def loss(p: float) -> float:
        model = match_model(p, best_of, sigma)
        err = 0.0
        for market, sel, line, prob, weight in targets:
            err += weight * (logit(model.prob(market, sel, line)) - logit(prob)) ** 2
        return err

    grid = np.linspace(0.30, 0.70, 41)
    losses = [loss(float(p)) for p in grid]
    i = int(np.argmin(losses))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, len(grid) - 1)])
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(25):
        m1 = hi - phi * (hi - lo)
        m2 = lo + phi * (hi - lo)
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)
