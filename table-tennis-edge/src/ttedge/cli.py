"""Command-line interface: ``ttedge <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import markets as mk
from .app import Workspace, price_match
from .backtest import calibrate_sigma
from .data import load_results
from .odds import parse_price
from .scanner import ScanConfig
from .synthetic import write_demo


def _table(rows: list[dict], cols: list[tuple[str, str]], limit: int | None = None) -> str:
    rows = rows[:limit] if limit else rows
    if not rows:
        return "  (none)"
    cells = [[h for _, h in cols]] + [["" if r.get(k) is None else str(r.get(k)) for k, _ in cols] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _pct(x: float | None, digits: int = 1) -> str | None:
    return None if x is None else f"{x * 100:.{digits}f}%"


def _workspace(args) -> Workspace:
    cfg = ScanConfig()
    overrides = {}
    for name in ("min_edge", "model_weight", "max_odds", "min_odds", "kelly_multiplier", "max_stake"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    if getattr(args, "devig", None):
        overrides["devig_method"] = args.devig
    if getattr(args, "sharp", None):
        overrides["sharp_books"] = args.sharp
    if overrides:
        cfg = ScanConfig.from_dict({**cfg.to_dict(), **overrides})
    ws = Workspace(config=cfg)
    sigma = getattr(args, "sigma", "auto")
    if sigma not in (None, "auto"):
        ws.auto_sigma = False
        ws.config.game_sigma = float(sigma)
    if getattr(args, "demo", False):
        ws.load_demo()
    else:
        results = getattr(args, "results", None)
        odds = getattr(args, "odds", None)
        names = ", ".join(Path(x).name for x in (results, odds) if x) or "no data"
        ws.load(results=Path(results) if results else None, odds=Path(odds) if odds else None, source=names)
    for err in ws.errors[:10]:
        print(f"warning: {err}", file=sys.stderr)
    if len(ws.errors) > 10:
        print(f"warning: ... {len(ws.errors) - 10} more row errors", file=sys.stderr)
    return ws


def cmd_demo(args) -> int:
    rp, op = write_demo(args.out, seed=args.seed)
    print(f"wrote {rp} and {op}  (synthetic: fictional players, books and prices)")
    print(f"next: ttedge scan --results {rp} --odds {op}")
    return 0


def cmd_scan(args) -> int:
    ws = _workspace(args)
    res = ws.last_scan
    assert res is not None
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=1, default=str))
        print(f"wrote {args.json}")
    print(f"data: {ws.source} | {len(ws.quotes)} prices, {len(res.events)} matches, "
          f"{len(ws.results)} past results | sigma={ws.config.game_sigma:g}")
    rows = [{
        "start": o.start_time[5:16].replace("T", " "),
        "match": f"{o.player_a} v {o.player_b}",
        "bet": o.description, "book": o.book, "price": f"{o.price:.2f}", "fair": f"{o.fair_price:.2f}",
        "edge": _pct(o.edge), "stake": _pct(o.stake, 2),
        "why": ",".join(f for f in o.flags if f != "opportunity"),
    } for o in res.opportunities]
    print(f"\nMispriced lines (edge >= {ws.config.min_edge:.0%}): {len(rows)}")
    print(_table(rows, [("start", "start"), ("match", "match"), ("bet", "bet"), ("book", "book"),
                        ("price", "price"), ("fair", "fair"), ("edge", "edge"), ("stake", "stake"),
                        ("why", "flags")], args.top))
    arbs = [{"match": f"{a.player_a} v {a.player_b}", "market": mk.MARKET_LABELS[a.market],
             "legs": " | ".join(f"{leg['description']} @{leg['price']} {leg['book']} ({leg['stake_share']:.0%})"
                                for leg in a.legs),
             "return": _pct(a.margin, 2)} for a in res.arbitrages]
    print(f"\nArbitrage: {len(arbs)}")
    print(_table(arbs, [("match", "match"), ("market", "market"), ("legs", "legs"), ("return", "return")], args.top))
    return 0


def cmd_price(args) -> int:
    if args.p_point is not None:
        out = price_match(p_point=args.p_point, best_of=args.best_of, sigma=args.sigma)
    else:
        if args.price_a is None or args.price_b is None:
            print("give two match-winner prices (e.g. ttedge price 1.65 2.25) or --p-point", file=sys.stderr)
            return 2
        out = price_match(parse_price(args.price_a), parse_price(args.price_b), best_of=args.best_of,
                          sigma=args.sigma, devig_method=args.devig or "power")
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    if "margin" in out:
        print(f"input {out['price_a']} / {out['price_b']}  margin {_pct(out['margin'], 2)}  "
              f"fair P(A) {_pct(out['fair_match_prob'])}")
    print(f"point-win P(A) {out['p_point']:.4f}  game-win {_pct(out['p_game'])}  match-win {_pct(out['p_match'])}  "
          f"expected points {out['expected_points']}")
    rows = [{"market": mk.MARKET_LABELS[r["market"]], "selection": r["description"],
             "prob": _pct(r["p_win"]), "push": _pct(r["p_push"]) if r["p_push"] else "",
             "fair": f"{r['fair_price']:.3f}" if r["fair_price"] else "-"} for r in out["fair"]]
    print(_table(rows, [("market", "market"), ("selection", "selection"), ("prob", "win prob"),
                        ("push", "push"), ("fair", "fair odds")]))
    return 0


def cmd_backtest(args) -> int:
    ws = _workspace(args)
    bt = ws.backtest(**{k: v for k, v in {"bet_snapshot": args.bet_snapshot, "stake_mode": args.stake}.items() if v})
    if args.json:
        Path(args.json).write_text(json.dumps(bt, indent=1, default=str))
        print(f"wrote {args.json}")
    pred, bet = bt["prediction"], bt["betting"]
    print(f"data: {ws.source} | {bt['matches']} matches, {bt['matches_with_odds']} with odds | "
          f"sigma={ws.config.game_sigma:g}")
    if pred.get("n"):
        print(f"\nPrediction quality on {pred['n']} matches (lower log loss is better):")
        print(_table([{"model": k, **pred[k]} for k in ("market", "blend", "ratings", "elo")],
                     [("model", "model"), ("log_loss", "log loss"), ("brier", "brier"), ("accuracy", "accuracy")]))
    if bet.get("n"):
        print(f"\nBetting: {bet['n']} bets, staked {bet['staked']}u, profit {bet['profit']}u, "
              f"ROI {_pct(bet['roi'], 2)} (+/- {_pct(bet['roi_se'], 2)} s.e.)")
        print(f"  claimed edge {_pct(bet['avg_edge'], 2)} vs closing-line value {_pct(bet['avg_clv'], 2)} "
              f"({_pct(bet['pct_positive_clv'])} of bets beat the close); max drawdown {bet['max_drawdown']}u")
        rows = [{"market": k, **v, "roi": _pct(v["roi"], 2), "avg_clv": _pct(v["avg_clv"], 2)}
                for k, v in bet["by_market"].items()]
        print(_table(rows, [("market", "market"), ("n", "bets"), ("staked", "staked"), ("profit", "profit"),
                            ("roi", "roi"), ("avg_clv", "clv")]))
        rows = [{"book": k, **v, "roi": _pct(v["roi"], 2), "avg_clv": _pct(v["avg_clv"], 2)}
                for k, v in bet["by_book"].items()]
        print(_table(rows, [("book", "book"), ("n", "bets"), ("staked", "staked"), ("profit", "profit"),
                            ("roi", "roi"), ("avg_clv", "clv")]))
    else:
        print("\nNo bets placed (need odds with match_ids that match the results file).")
    buckets = bt["calibration"]["buckets"]
    if buckets:
        print("\nMarket calibration (favourite-longshot check):")
        print(_table(buckets, [("bucket", "implied"), ("n", "n"), ("implied_multiplicative", "proportional"),
                               ("implied_devigged", "devigged"), ("actual", "actual"), ("se", "s.e.")]))
    return 0


def cmd_ratings(args) -> int:
    ws = _workspace(args)
    rows = ws.leaderboard(args.min_matches)
    print(_table(rows, [("player", "player"), ("rating", "rating"), ("sd", "sd"), ("matches", "matches"),
                        ("win_pct", "win%"), ("point_pct", "point%"), ("p_vs_avg", "P(point) v avg")], args.top))
    return 0


def cmd_calibrate(args) -> int:
    results, errors = load_results(Path(args.results))
    for e in errors[:10]:
        print(f"warning: {e}", file=sys.stderr)
    fit = calibrate_sigma(results)
    print(f"best sigma {fit['sigma']} from {fit['matches']} matches / {fit['games']} games")
    print(_table(fit["grid"], [("sigma", "sigma"), ("log_lik", "log likelihood")]))
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    ws = _workspace(args) if (args.demo or args.odds or args.results) else Workspace()
    if not (args.demo or args.odds or args.results):
        ws.load_demo()
    serve(ws, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ttedge", description="Find mispriced table tennis betting lines.")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp):
        sp.add_argument("--results", help="results CSV (past matches, used to train ratings)")
        sp.add_argument("--odds", help="odds CSV")
        sp.add_argument("--demo", action="store_true", help="use built-in synthetic demo data")
        sp.add_argument("--sigma", default="auto", help="game-to-game variation; 'auto' fits it from results")
        sp.add_argument("--min-edge", dest="min_edge", type=float)
        sp.add_argument("--model-weight", dest="model_weight", type=float)
        sp.add_argument("--min-odds", dest="min_odds", type=float)
        sp.add_argument("--max-odds", dest="max_odds", type=float)
        sp.add_argument("--kelly", dest="kelly_multiplier", type=float, help="fraction of full Kelly (default 0.25)")
        sp.add_argument("--max-stake", dest="max_stake", type=float)
        sp.add_argument("--devig", choices=("power", "shin", "multiplicative", "additive"))
        sp.add_argument("--sharp", help="comma-separated sharp books (default pinnacle,sbobet,betfair,...)")

    sp = sub.add_parser("demo", help="write synthetic demo CSVs")
    sp.add_argument("--out", default="demo-data")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("scan", help="list mispriced lines and arbitrage")
    data_args(sp)
    sp.add_argument("--top", type=int, default=40)
    sp.add_argument("--json", help="also write full results to this JSON file")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("price", help="fair prices for every market from a match price")
    sp.add_argument("price_a", nargs="?")
    sp.add_argument("price_b", nargs="?")
    sp.add_argument("--p-point", dest="p_point", type=float, help="point-win probability instead of prices")
    sp.add_argument("--best-of", dest="best_of", type=int, default=5)
    sp.add_argument("--sigma", type=float, default=0.0)
    sp.add_argument("--devig", choices=("power", "shin", "multiplicative", "additive"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_price)

    sp = sub.add_parser("backtest", help="walk-forward backtest on past results + odds")
    data_args(sp)
    sp.add_argument("--bet-snapshot", dest="bet_snapshot", choices=("open", "close", "current"))
    sp.add_argument("--stake", choices=("kelly", "flat"))
    sp.add_argument("--json")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("ratings", help="player rating table")
    data_args(sp)
    sp.add_argument("--min-matches", dest="min_matches", type=int, default=5)
    sp.add_argument("--top", type=int, default=50)
    sp.set_defaults(func=cmd_ratings)

    sp = sub.add_parser("calibrate", help="fit game-to-game variation from point scores")
    sp.add_argument("--results", required=True)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("serve", help="run the web dashboard")
    data_args(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "sharp", None):
        args.sharp = [s.strip().lower() for s in args.sharp.split(",") if s.strip()]
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
