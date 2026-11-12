"""Player strength ratings.

:class:`PointRating` is the main model. Each player has a strength on the logit
point scale with an uncertainty, so ``P(A wins a point) = sigmoid(r_A - r_B)``.
After each match the ratings take a Kalman-filter step using the points won (or
the games won when only the game score is known). Learning from points rather
than wins uses ~75 observations per match instead of one, which matters in the
high-frequency leagues (Czech Liga Pro, TT Cup, Setka Cup, TT Elite Series)
where the same players meet again and again.

:class:`EloRating` is a plain win/loss Elo kept as a baseline for comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .markets import MatchScore
from .markov import game_win_prob, match_win_prob, point_prob_from_match_prob
from .odds import sigmoid


def player_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def _days(t: datetime | None) -> float | None:
    return None if t is None else t.timestamp() / 86400.0


@dataclass
class PlayerState:
    name: str
    mu: float = 0.0
    var: float = 0.0
    matches: int = 0
    wins: int = 0
    points_won: int = 0
    points_played: int = 0
    last_seen: float | None = None


@dataclass
class Prediction:
    p_point: float
    p_match: float
    diff_mean: float
    diff_sd: float
    matches_a: int
    matches_b: int

    @property
    def min_matches(self) -> int:
        return min(self.matches_a, self.matches_b)


@dataclass
class PointRatingConfig:
    prior_sd: float = 0.12         # spread of player strengths (logit point units)
    drift_sd_per_day: float = 0.01  # how fast true strength moves between matches
    min_sd: float = 0.015
    overdispersion: float = 1.3     # >1: points are less informative than i.i.d. coin flips
    max_step: float = 1.0           # clip extreme innovations (walkovers, injuries)


@dataclass
class PointRating:
    config: PointRatingConfig = field(default_factory=PointRatingConfig)
    players: dict[str, PlayerState] = field(default_factory=dict)

    def _get(self, name: str) -> PlayerState:
        key = player_key(name)
        st = self.players.get(key)
        if st is None:
            st = PlayerState(name=name.strip(), var=self.config.prior_sd ** 2)
            self.players[key] = st
        return st

    def _drifted_var(self, st: PlayerState, now: float | None) -> float:
        var = st.var
        if now is not None and st.last_seen is not None and now > st.last_seen:
            var += self.config.drift_sd_per_day ** 2 * (now - st.last_seen)
        return min(var, self.config.prior_sd ** 2)

    def predict(self, a: str, b: str, best_of: int = 5, when: datetime | None = None) -> Prediction:
        now = _days(when)
        sa = self.players.get(player_key(a)) or PlayerState(a, var=self.config.prior_sd ** 2)
        sb = self.players.get(player_key(b)) or PlayerState(b, var=self.config.prior_sd ** 2)
        d = sa.mu - sb.mu
        var = self._drifted_var(sa, now) + self._drifted_var(sb, now)
        # expected sigmoid under rating uncertainty (probit approximation)
        p = sigmoid(d / math.sqrt(1.0 + math.pi * var / 8.0))
        return Prediction(p, match_win_prob(p, best_of), d, math.sqrt(var), sa.matches, sb.matches)

    def update(self, a: str, b: str, score: MatchScore, when: datetime | None = None) -> None:
        cfg = self.config
        now = _days(when)
        sa, sb = self._get(a), self._get(b)
        va, vb = self._drifted_var(sa, now), self._drifted_var(sb, now)
        d = sa.mu - sb.mu
        p = sigmoid(d)
        if score.has_points:
            n = score.points_a + score.points_b
            frac = score.points_a / n
            slope = p * (1 - p)  # d p / d d
            innov = (frac - p) / slope
            obs_var = cfg.overdispersion * p * (1 - p) / (n * slope ** 2)
        else:
            n = score.games_a + score.games_b
            g = game_win_prob(p)
            eps = 1e-4
            slope = (game_win_prob(sigmoid(d + eps)) - game_win_prob(sigmoid(d - eps))) / (2 * eps)
            innov = (score.games_a / n - g) / slope
            obs_var = cfg.overdispersion * g * (1 - g) / (n * slope ** 2)
        innov = max(-cfg.max_step, min(cfg.max_step, innov))
        s = va + vb + obs_var
        ka, kb = va / s, vb / s
        sa.mu += ka * innov
        sb.mu -= kb * innov
        sa.var = max(va * (1 - ka), cfg.min_sd ** 2)
        sb.var = max(vb * (1 - kb), cfg.min_sd ** 2)
        for st, won, pts_w, pts_l in (
            (sa, score.games_a > score.games_b, score.points_a, score.points_b),
            (sb, score.games_b > score.games_a, score.points_b, score.points_a),
        ):
            st.matches += 1
            st.wins += int(won)
            st.points_won += pts_w
            st.points_played += pts_w + pts_l
            if now is not None:
                st.last_seen = now

    def leaderboard(self, min_matches: int = 1) -> list[dict]:
        rows = [
            {
                "player": st.name,
                "rating": round(st.mu, 4),
                "sd": round(math.sqrt(st.var), 4),
                "matches": st.matches,
                "wins": st.wins,
                "win_pct": round(st.wins / st.matches, 3) if st.matches else None,
                "point_pct": round(st.points_won / st.points_played, 4) if st.points_played else None,
                # point-win probability against an average (rating 0) opponent
                "p_vs_avg": round(sigmoid(st.mu), 4),
            }
            for st in self.players.values()
            if st.matches >= min_matches
        ]
        rows.sort(key=lambda r: r["rating"], reverse=True)
        return rows


@dataclass
class EloRating:
    """Classic win/loss Elo, for comparison against the point model."""

    k: float = 32.0
    players: dict[str, PlayerState] = field(default_factory=dict)

    def _get(self, name: str) -> PlayerState:
        key = player_key(name)
        if key not in self.players:
            self.players[key] = PlayerState(name=name.strip(), mu=1500.0)
        return self.players[key]

    def predict(self, a: str, b: str, best_of: int = 5, when: datetime | None = None) -> Prediction:
        sa = self.players.get(player_key(a)) or PlayerState(a, mu=1500.0)
        sb = self.players.get(player_key(b)) or PlayerState(b, mu=1500.0)
        pm = 1.0 / (1.0 + 10 ** ((sb.mu - sa.mu) / 400.0))
        return Prediction(point_prob_from_match_prob(pm, best_of), pm, sa.mu - sb.mu, 0.0, sa.matches, sb.matches)

    def update(self, a: str, b: str, score: MatchScore, when: datetime | None = None) -> None:
        sa, sb = self._get(a), self._get(b)
        pm = 1.0 / (1.0 + 10 ** ((sb.mu - sa.mu) / 400.0))
        won = 1.0 if score.games_a > score.games_b else 0.0
        # larger K while a player is new
        ka = self.k * (2.0 if sa.matches < 15 else 1.0)
        kb = self.k * (2.0 if sb.matches < 15 else 1.0)
        sa.mu += ka * (won - pm)
        sb.mu -= kb * (won - pm)
        for st, w in ((sa, won), (sb, 1 - won)):
            st.matches += 1
            st.wins += int(w)
