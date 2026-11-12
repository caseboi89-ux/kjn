"""Walk-forward backtest and model calibration.

Matches are replayed in time order. Before each match the scanner sees only
the ratings learned from earlier results and the pre-match prices; bets are
settled against the real score, and only then do the ratings learn from it.
Nothing from the future leaks into a decision.

Two numbers matter most:

* **ROI** – noisy; needs thousands of bets before it means much.
* **CLV** (closing-line value) – expected value of each bet measured against
  the closing consensus price. Positive average CLV is the most reliable early
  sign that a method really finds mispriced lines.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from . import markets as mk
from .data import Quote, Result
from .markov import _GH_W, _GH_X, match_model
from .odds import devig, logit, sigmoid
from .ratings import EloRating, PointRating, PointRatingConfig, player_key
from .scanner import ScanConfig, _Event, scan_event


@dataclass
class BacktestConfig:
    bet_snapshot: str = "open"      # prices you would have bet; falls back to close/current
    close_snapshot: str = "close"   # prices used to measure CLV
    stake_mode: str = "kelly"       # "kelly" (scanner stake, capped) or "flat"
    flat_stake: float = 1.0         # units per bet in flat mode
    bankroll: float = 100.0         # units; Kelly stakes are a fraction of this (no compounding)
    best_price_only: bool = True    # one bet per selection, at the best-edge book


def _logloss(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _choose(by_snap: dict[str, list[Quote]], *order: str) -> list[Quote]:
    for s in order:
        if by_snap.get(s):
            return by_snap[s]
    return []


def run_backtest(results: list[Result], quotes: list[Quote], scan_cfg: ScanConfig | None = None,
                 rating_cfg: PointRatingConfig | None = None, cfg: BacktestConfig | None = None) -> dict:
    scan_cfg = scan_cfg or ScanConfig()
    cfg = cfg or BacktestConfig()
    ratings = PointRating(rating_cfg or PointRatingConfig())
    elo = EloRating()

    odds_by_match: dict[str, dict[str, list[Quote]]] = defaultdict(lambda: defaultdict(list))
    for q in quotes:
        odds_by_match[q.match_id][q.snapshot].append(q)

    preds = []          # (rating, elo, market, blend, outcome) for matches with odds and history
    calib = []          # (market prob of side, multiplicative prob of side, outcome) both sides
    bets = []
    matches_with_odds = 0

    for res in results:
        snaps = odds_by_match.get(res.match_id, {})
        pred = ratings.predict(res.player_a, res.player_b, res.best_of, res.start_time)
        elo_pred = elo.predict(res.player_a, res.player_b, res.best_of, res.start_time)
        a_won = int(res.score.games_a > res.score.games_b)

        close_q = _choose(snaps, cfg.close_snapshot, "current")
        close_ev = _Event(close_q, scan_cfg) if close_q else None
        if close_ev is not None:
            matches_with_odds += 1
            ml, _ = close_ev.consensus((mk.ML, None))
            # results are keyed to the odds by match_id; make sure A/B agree
            flipped = player_key(close_ev.player_a) != player_key(res.player_a)
            if ml is not None:
                m_close = ml["B"] if flipped else ml["A"]
                mult = []
                for book in close_ev.prices:
                    sels = close_ev.prices[book].get((mk.ML, None), {})
                    if "A" in sels and "B" in sels:
                        pa = devig([sels["A"], sels["B"]], "multiplicative")[0]
                        mult.append(1 - pa if flipped else pa)
                mult_p = float(np.mean(mult)) if mult else m_close
                calib.append((m_close, mult_p, a_won))
                calib.append((1 - m_close, 1 - mult_p, 1 - a_won))
                if pred.min_matches >= scan_cfg.min_model_matches:
                    pm = match_model(pred.p_point, res.best_of, scan_cfg.game_sigma).p_match
                    blend_p = sigmoid((1 - scan_cfg.model_weight) * logit(m_close) + scan_cfg.model_weight * logit(pm))
                    preds.append((pm, elo_pred.p_match, m_close, blend_p, a_won))

        bet_q = _choose(snaps, cfg.bet_snapshot, cfg.close_snapshot, "current")
        if bet_q:
            evals, _, _ = scan_event(bet_q, scan_cfg, ratings)
            opps = [e for e in evals if e.is_opportunity]
            if cfg.best_price_only:
                best: dict[tuple, object] = {}
                for e in opps:
                    key = mk.group_key(e.market, e.selection, e.line) + (e.selection,)
                    if key not in best or e.edge > best[key].edge:  # type: ignore[attr-defined]
                        best[key] = e
                opps = list(best.values())
            score = res.score
            if opps and player_flip(opps[0].player_a, res.player_a):
                score = score.swapped()
            for e in opps:
                outcome = mk.settle(e.market, e.selection, e.line, score)
                if outcome is None:
                    continue
                stake = cfg.flat_stake if cfg.stake_mode == "flat" else e.stake * cfg.bankroll
                if stake <= 0:
                    continue
                profit = stake * (e.price - 1) if outcome == "win" else (0.0 if outcome == "push" else -stake)
                clv = None
                if close_ev is not None and close_q is not bet_q:
                    clv = _clv(close_ev, scan_cfg, e)
                bets.append({
                    "match_id": e.match_id, "start_time": e.start_time, "book": e.book, "market": e.market,
                    "description": e.description, "price": e.price, "p_win": e.p_win, "edge": e.edge,
                    "stake": round(stake, 4), "result": outcome, "profit": round(profit, 4), "clv": clv,
                })

        ratings.update(res.player_a, res.player_b, res.score, res.start_time)
        elo.update(res.player_a, res.player_b, res.score, res.start_time)

    return {
        "matches": len(results),
        "matches_with_odds": matches_with_odds,
        "prediction": _prediction_metrics(preds),
        "calibration": _calibration(calib),
        "betting": _betting_metrics(bets),
        "bets": bets[-500:],
        "config": {"scan": scan_cfg.to_dict(), "backtest": asdict(cfg)},
    }


def player_flip(odds_a: str, result_a: str) -> bool:
    return player_key(odds_a) != player_key(result_a)


def _clv(close_ev: _Event, scan_cfg: ScanConfig, e) -> float | None:
    """EV of the bet price measured against the closing consensus (all books)."""
    flipped = player_key(close_ev.player_a) != player_key(e.player_a)
    sel = mk.swap_selection(e.market, e.selection) if flipped else e.selection
    gk = mk.group_key(e.market, sel, e.line)
    cons, _ = close_ev.consensus(gk)
    if cons is not None:
        p_win, p_push = cons[sel], 0.0
        p_point = close_ev.market_point_prob(None)
        if p_point is not None and e.line is not None:
            _, p_push, _ = match_model(p_point, close_ev.best_of, scan_cfg.game_sigma).outcome(e.market, sel, e.line)
            p_win *= 1 - p_push
    else:
        p_point = close_ev.market_point_prob(None)
        if p_point is None:
            return None
        p_win, p_push, _ = match_model(p_point, close_ev.best_of, scan_cfg.game_sigma).outcome(e.market, sel, e.line)
    p_lose = 1 - p_win - p_push
    return round(p_win * (e.price - 1) - p_lose, 5)


def _prediction_metrics(preds: list[tuple]) -> dict:
    if not preds:
        return {"n": 0}
    arr = np.array(preds)
    y = arr[:, 4].astype(int)
    out = {"n": int(len(arr))}
    for i, name in enumerate(("ratings", "elo", "market", "blend")):
        p = arr[:, i]
        out[name] = {
            "log_loss": round(float(np.mean([_logloss(pp, yy) for pp, yy in zip(p, y)])), 5),
            "brier": round(float(np.mean((p - y) ** 2)), 5),
            "accuracy": round(float(np.mean((p > 0.5) == (y == 1))), 4),
        }
    return out


def _calibration(rows: list[tuple]) -> dict:
    """Actual win rate by implied-probability bucket (favourite-longshot bias check)."""
    if not rows:
        return {"buckets": []}
    arr = np.array(rows)
    edges = np.linspace(0, 1, 11)
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr[:, 1] >= lo) & (arr[:, 1] < hi if hi < 1 else arr[:, 1] <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        actual = float(arr[mask, 2].mean())
        buckets.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": n,
            "implied_multiplicative": round(float(arr[mask, 1].mean()), 4),
            "implied_devigged": round(float(arr[mask, 0].mean()), 4),
            "actual": round(actual, 4),
            "se": round(math.sqrt(max(actual * (1 - actual), 1e-9) / n), 4),
        })
    return {"buckets": buckets}


def _betting_metrics(bets: list[dict]) -> dict:
    if not bets:
        return {"n": 0}
    stakes = np.array([b["stake"] for b in bets])
    profits = np.array([b["profit"] for b in bets])
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    drawdown = float(np.max(peak - cum)) if len(cum) else 0.0
    ret = profits / stakes
    se = float(np.std(ret, ddof=1) / math.sqrt(len(ret))) if len(ret) > 1 else float("nan")
    clvs = [b["clv"] for b in bets if b["clv"] is not None]
    step = max(1, len(cum) // 300)
    by_market: dict[str, dict] = defaultdict(lambda: {"n": 0, "staked": 0.0, "profit": 0.0, "clv": []})
    by_book: dict[str, dict] = defaultdict(lambda: {"n": 0, "staked": 0.0, "profit": 0.0, "clv": []})
    for b in bets:
        for table, key in ((by_market, b["market"]), (by_book, b["book"])):
            t = table[key]
            t["n"] += 1
            t["staked"] += b["stake"]
            t["profit"] += b["profit"]
            if b["clv"] is not None:
                t["clv"].append(b["clv"])

    def finish(table):
        return {k: {"n": v["n"], "staked": round(v["staked"], 2), "profit": round(v["profit"], 2),
                    "roi": round(v["profit"] / v["staked"], 4) if v["staked"] else None,
                    "avg_clv": round(float(np.mean(v["clv"])), 4) if v["clv"] else None}
                for k, v in sorted(table.items())}

    return {
        "n": len(bets),
        "staked": round(float(stakes.sum()), 2),
        "profit": round(float(profits.sum()), 2),
        "roi": round(float(profits.sum() / stakes.sum()), 4),
        "roi_se": round(se, 4),
        "hit_rate": round(float(np.mean([b["result"] == "win" for b in bets])), 4),
        "avg_price": round(float(np.mean([b["price"] for b in bets])), 3),
        "avg_edge": round(float(np.mean([b["edge"] for b in bets])), 4),
        "avg_clv": round(float(np.mean(clvs)), 4) if clvs else None,
        "pct_positive_clv": round(float(np.mean([c > 0 for c in clvs])), 4) if clvs else None,
        "max_drawdown": round(drawdown, 2),
        "curve": [{"i": int(i), "t": bets[i]["start_time"], "profit": round(float(cum[i]), 2)}
                  for i in range(0, len(cum), step)] + [{"i": len(cum) - 1, "t": bets[-1]["start_time"],
                                                          "profit": round(float(cum[-1]), 2)}],
        "by_market": finish(by_market),
        "by_book": finish(by_book),
    }


def _game_like(pp: np.ndarray, loser: int) -> np.ndarray:
    """Vectorised P(the game's winner wins it with the loser on ``loser`` points)."""
    q = 1.0 - pp
    if loser <= 9:
        return math.comb(10 + loser, loser) * pp ** 11 * q ** loser
    return math.comb(20, 10) * pp ** 12 * q ** 10 * (2.0 * pp * q) ** (loser - 10)


def calibrate_sigma(results: list[Result], rating_cfg: PointRatingConfig | None = None,
                    grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4),
                    min_matches: int = 15) -> dict:
    """Fit the game-to-game variation ``sigma`` from point scores.

    Uses a walk-forward rating as the pre-match estimate and integrates over
    that estimate's own uncertainty, so rating error is not mistaken for
    game-to-game variation. ``sigma`` is weakly identified: expect a flat
    likelihood near the optimum and treat the result as a rough setting.
    """
    ratings = PointRating(rating_cfg or PointRatingConfig())
    sig = np.asarray(grid, dtype=float)[:, None, None]          # (S, 1, 1)
    outer = (np.sqrt(2.0) * _GH_X)[None, :, None]                 # (1, O, 1)
    inner = (np.sqrt(2.0) * _GH_X)[None, None, :]                 # (1, 1, I)
    w_inner = _GH_W[None, None, :]
    w_outer = _GH_W[None, :]
    loglik = np.zeros(len(grid))
    n_games = n_matches = 0
    for res in results:
        pred = ratings.predict(res.player_a, res.player_b, res.best_of, res.start_time)
        if res.score.has_points and pred.min_matches >= min_matches:
            n_matches += 1
            n_games += len(res.score.game_points)
            d = pred.diff_mean + pred.diff_sd * outer + sig * inner   # (S, O, I)
            pp = 1.0 / (1.0 + np.exp(-d))
            match_like = np.ones((len(grid), len(_GH_X)))
            for a_pts, b_pts in res.score.game_points:
                loser = min(min(a_pts, b_pts), 10 + 25)
                side = pp if a_pts > b_pts else 1.0 - pp
                match_like *= np.sum(w_inner * _game_like(side, loser), axis=2)
            like = np.sum(w_outer * match_like, axis=1)
            loglik += np.log(np.maximum(like, 1e-300))
        ratings.update(res.player_a, res.player_b, res.score, res.start_time)
    if n_matches == 0:
        return {"sigma": 0.0, "matches": 0, "games": 0, "grid": []}
    best = int(np.argmax(loglik))
    return {
        "sigma": grid[best],
        "matches": n_matches,
        "games": n_games,
        "grid": [{"sigma": s, "log_lik": round(float(v), 2)} for s, v in zip(grid, loglik)],
    }
