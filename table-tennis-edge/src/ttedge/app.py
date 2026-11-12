"""Workspace: loaded data + fitted models, shared by the CLI and the web server."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import markets as mk
from .backtest import BacktestConfig, calibrate_sigma, run_backtest
from .data import Quote, Result, load_odds, load_results
from .markov import match_model, point_prob_from_match_prob
from .odds import devig
from .ratings import PointRating, PointRatingConfig
from .scanner import ScanConfig, ScanResult, scan
from .synthetic import generate


@dataclass
class Workspace:
    config: ScanConfig = field(default_factory=ScanConfig)
    rating_config: PointRatingConfig = field(default_factory=PointRatingConfig)
    results: list[Result] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source: str = "empty"
    auto_sigma: bool = True
    sigma_fit: dict | None = None
    ratings: PointRating | None = None
    last_scan: ScanResult | None = None
    last_backtest: dict | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ---------------------------------------------------------------- loading
    def load(self, results=None, odds=None, source: str = "files") -> None:
        with self.lock:
            if results is not None:
                self.results, errs = load_results(results)
                self.errors = [e for e in self.errors if not e.startswith("results")] + errs
            if odds is not None:
                self.quotes, errs = load_odds(odds)
                self.errors = [e for e in self.errors if not e.startswith("odds")] + errs
            self.source = source
            self.refresh(refit_sigma=results is not None)

    def load_demo(self, seed: int = 7) -> None:
        with self.lock:
            self.results, self.quotes = generate(seed=seed)
            self.errors = []
            self.source = "synthetic demo data (fictional players, books and prices)"
            self.refresh(refit_sigma=True)

    def refresh(self, refit_sigma: bool = False) -> None:
        with self.lock:
            if self.auto_sigma:
                if refit_sigma or self.sigma_fit is None:
                    has_points = any(r.score.has_points for r in self.results)
                    self.sigma_fit = calibrate_sigma(self.results, self.rating_config) if has_points else None
                fit = self.sigma_fit
                # too few matches and the fit is noise: fall back to i.i.d. points
                self.config.game_sigma = float(fit["sigma"]) if fit and fit["matches"] >= 200 else 0.0
            self.ratings = PointRating(self.rating_config)
            for r in self.results:
                self.ratings.update(r.player_a, r.player_b, r.score, r.start_time)
            self.last_scan = scan(self.quotes, self.config, self.ratings, snapshot=self._scan_snapshot())
            self.last_backtest = None

    def _scan_snapshot(self) -> str | None:
        snaps = {q.snapshot for q in self.quotes}
        if "current" in snaps:
            return "current"
        return "close" if "close" in snaps else None

    def update_config(self, values: dict) -> None:
        with self.lock:
            if "auto_sigma" in values:
                self.auto_sigma = bool(values.pop("auto_sigma"))
            merged = {**self.config.to_dict(), **values}
            self.config = ScanConfig.from_dict(merged)
            if "game_sigma" in values:
                self.auto_sigma = False
            self.refresh()

    # ---------------------------------------------------------------- queries
    def state(self) -> dict:
        snaps: dict[str, int] = {}
        for q in self.quotes:
            snaps[q.snapshot] = snaps.get(q.snapshot, 0) + 1
        scan_res = self.last_scan
        return {
            "source": self.source,
            "results": len(self.results),
            "quotes": len(self.quotes),
            "snapshots": snaps,
            "events": len(scan_res.events) if scan_res else 0,
            "opportunities": len(scan_res.opportunities) if scan_res else 0,
            "arbitrages": len(scan_res.arbitrages) if scan_res else 0,
            "books": sorted({q.book for q in self.quotes}),
            "errors": self.errors[:50],
            "error_count": len(self.errors),
            "config": self.config.to_dict(),
            "auto_sigma": self.auto_sigma,
            "sigma_fit": self.sigma_fit,
            "rating_config": asdict(self.rating_config),
        }

    def scan_dict(self) -> dict:
        if self.last_scan is None:
            return {"opportunities": [], "arbitrages": [], "events": []}
        data = self.last_scan.to_dict()
        # event list without the per-event fair tables keeps the payload small
        data["events"] = [{k: v for k, v in e.items() if k != "fair"} for e in data["events"]]
        data.pop("evaluations")
        return data

    def event_detail(self, match_id: str) -> dict | None:
        if self.last_scan is None:
            return None
        events = [e for e in self.last_scan.events if e.match_id == match_id]
        if not events:
            return None
        evals = [asdict(e) for e in self.last_scan.evaluations if e.match_id == match_id]
        evals.sort(key=lambda e: (mk.MARKETS.index(e["market"]), e["line"] or 0, e["selection"], e["book"]))
        return {"event": events[0].to_dict(), "evaluations": evals}

    def leaderboard(self, min_matches: int = 1) -> list[dict]:
        return self.ratings.leaderboard(min_matches) if self.ratings else []

    def backtest(self, **kwargs) -> dict:
        with self.lock:
            if self.last_backtest is None or kwargs:
                bt_cfg = BacktestConfig(**kwargs) if kwargs else BacktestConfig()
                self.last_backtest = run_backtest(self.results, self.quotes, self.config, self.rating_config, bt_cfg)
            return self.last_backtest


def price_match(price_a: float | None = None, price_b: float | None = None, p_point: float | None = None,
                best_of: int = 5, sigma: float = 0.0, devig_method: str = "power") -> dict:
    """Fair prices for every standard market from a match-winner price pair or a point probability."""
    info: dict = {"best_of": best_of, "sigma": sigma}
    if p_point is None:
        if price_a is None or price_b is None:
            raise ValueError("give both match-winner prices or p_point")
        pa, _ = devig([price_a, price_b], devig_method)
        info.update(price_a=price_a, price_b=price_b, margin=round(1 / price_a + 1 / price_b - 1, 5),
                    fair_match_prob=round(pa, 5))
        p_point = point_prob_from_match_prob(pa, best_of, sigma)
    if not 0 < p_point < 1:
        raise ValueError("p_point must be between 0 and 1")
    model = match_model(p_point, best_of, sigma)
    rows = model.fair_table()
    for r in rows:
        r["description"] = mk.describe(r["market"], r["selection"], r["line"], "Player A", "Player B")
    info.update(p_point=round(p_point, 6), p_game=round(model.p_game, 5), p_match=round(model.p_match, 5),
                expected_points=round(model.expected_points(), 2), fair=rows)
    return info


def default_paths(directory: str | Path) -> tuple[Path, Path]:
    d = Path(directory)
    return d / "results.csv", d / "odds.csv"
