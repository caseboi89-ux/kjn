import json
import threading
import urllib.request

import pytest

from ttedge.app import Workspace, price_match
from ttedge.backtest import calibrate_sigma, run_backtest
from ttedge.data import load_odds, load_results, write_odds, write_results
from ttedge.scanner import ScanConfig
from ttedge.server import make_server
from ttedge.synthetic import generate


@pytest.fixture(scope="module")
def demo():
    return generate(seed=11, n_players=16, days=8, groups_per_day=4, upcoming_groups=2)


def test_synthetic_round_trips_through_csv(tmp_path, demo):
    results, quotes = demo
    write_results(tmp_path / "r.csv", results)
    write_odds(tmp_path / "o.csv", quotes)
    r2, e1 = load_results(tmp_path / "r.csv")
    q2, e2 = load_odds(tmp_path / "o.csv")
    assert not e1 and not e2
    assert len(r2) == len(results) and len(q2) == len(quotes)
    assert str(r2[0].score) == str(results[0].score)


def test_backtest_runs_and_reports(demo):
    results, quotes = demo
    bt = run_backtest(results, quotes, ScanConfig(min_model_matches=5, game_sigma=0.1))
    assert bt["matches"] == len(results)
    assert bt["matches_with_odds"] == len(results)
    assert bt["prediction"]["n"] > 0
    for k in ("market", "ratings", "elo", "blend"):
        assert 0.4 < bt["prediction"][k]["log_loss"] < 0.8
    bet = bt["betting"]
    assert bet["n"] > 0 and bet["avg_clv"] is not None
    assert abs(sum(v["profit"] for v in bet["by_market"].values()) - bet["profit"]) < 0.05
    assert sum(b["n"] for b in bt["calibration"]["buckets"]) == 2 * len(results)


def test_backtest_does_not_peek_at_the_result(demo):
    """Bets on a match must not change if that match's score changes."""
    results, quotes = demo
    import copy
    from ttedge import markets as mk
    last = results[-1]
    flipped = copy.deepcopy(results)
    flipped[-1].score = last.score.swapped()
    a = run_backtest(results, quotes, ScanConfig(min_model_matches=5))
    b = run_backtest(flipped, quotes, ScanConfig(min_model_matches=5))
    bets_a = [(x["description"], x["price"], x["stake"]) for x in a["bets"] if x["match_id"] == last.match_id]
    bets_b = [(x["description"], x["price"], x["stake"]) for x in b["bets"] if x["match_id"] == last.match_id]
    assert bets_a == bets_b
    assert mk.ML  # keep import used


def test_calibrate_sigma(demo):
    results, _ = demo
    fit = calibrate_sigma(results, min_matches=5)
    assert fit["matches"] > 0 and fit["sigma"] in [g["sigma"] for g in fit["grid"]]


def test_price_match():
    out = price_match(1.62, 2.35)
    assert out["margin"] == pytest.approx(1 / 1.62 + 1 / 2.35 - 1, abs=1e-5)
    assert out["p_match"] == pytest.approx(out["fair_match_prob"], abs=2e-5)
    assert price_match(p_point=0.5)["p_match"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        price_match(1.62)


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, r.read()


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_endpoints(demo):
    results, quotes = demo
    ws = Workspace()
    ws.results, ws.quotes, ws.source = results, quotes, "test"
    ws.refresh(refit_sigma=True)
    httpd = make_server(ws, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, html = _get(base, "/")
        assert status == 200 and b"TT Edge" in html
        state = json.loads(_get(base, "/api/state")[1])
        assert state["results"] == len(results)
        scan = json.loads(_get(base, "/api/scan")[1])
        assert scan["events"]
        mid = scan["events"][0]["match_id"]
        detail = json.loads(_get(base, "/api/event?id=" + mid)[1])
        assert detail["event"]["fair"] and detail["evaluations"]
        assert json.loads(_get(base, "/api/ratings")[1])
        status, priced = _post(base, "/api/price", {"price_a": "1.5", "price_b": "2.6"})
        assert status == 200 and priced["fair"]
        status, err = _post(base, "/api/price", {"price_a": "0.5", "price_b": "2.6"})
        assert status == 400 and "error" in err
        status, st = _post(base, "/api/config", {"min_edge": 0.05})
        assert status == 200 and st["config"]["min_edge"] == 0.05
        csv_text = "match_id,player_a,player_b,book,market,selection,line,price\nx,A,B,K,ML,A,,1.9\nx,A,B,K,ML,B,,1.9\n"
        status, st = _post(base, "/api/upload", {"kind": "odds", "csv": csv_text})
        assert status == 200 and st["quotes"] == 2
        status, st = _post(base, "/api/clear", {})
        assert status == 200 and st["quotes"] == 0 and st["results"] == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
