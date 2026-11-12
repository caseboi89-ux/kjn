from datetime import datetime, timedelta, timezone

import numpy as np

from ttedge import markets as mk
from ttedge.ratings import EloRating, PointRating


def test_point_rating_learns_the_stronger_player():
    r = PointRating()
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(30):
        r.update("Strong", "Weak", mk.parse_score("11-8 11-9 9-11 11-7"), t + timedelta(hours=i))
    pred = r.predict("Strong", "Weak")
    assert pred.p_point > 0.53
    assert pred.p_match > 0.75
    back = r.predict("Weak", "Strong")
    assert back.p_point == 1 - pred.p_point or abs(back.p_point - (1 - pred.p_point)) < 1e-12
    board = r.leaderboard()
    assert board[0]["player"] == "Strong"
    assert board[0]["sd"] < 0.12


def test_unknown_players_are_even_and_names_normalised():
    r = PointRating()
    assert abs(r.predict("A", "B").p_match - 0.5) < 1e-12
    r.update("Jan  Novak", "Petr", mk.parse_score("3-0"))
    assert "jan novak" in r.players
    assert r.predict("jan novak", "Petr").p_point > 0.5  # games-only update moves ratings


def test_ratings_track_simulated_strengths():
    rng = np.random.default_rng(3)
    true = {f"P{i}": rng.normal(0, 0.1) for i in range(12)}
    r = PointRating()
    names = list(true)
    from ttedge.synthetic import _simulate_match
    for _ in range(1500):
        a, b = rng.choice(names, 2, replace=False)
        r.update(a, b, _simulate_match(rng, true[a] - true[b], 5))
    est = np.array([r.players[n.lower()].mu for n in names])
    tru = np.array([true[n] for n in names])
    assert np.corrcoef(est, tru)[0, 1] > 0.9


def test_elo_baseline():
    e = EloRating()
    for _ in range(10):
        e.update("X", "Y", mk.parse_score("3-1"))
    assert e.predict("X", "Y").p_match > 0.7
