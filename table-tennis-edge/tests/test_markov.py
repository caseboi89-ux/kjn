import numpy as np
import pytest

from ttedge import markets as mk
from ttedge.markov import (MatchModel, fit_point_prob, game_win_prob, match_model, match_win_prob,
                           point_prob_from_match_prob)


def test_even_match_is_symmetric():
    m = MatchModel(0.5)
    assert m.p_game == pytest.approx(0.5)
    assert m.p_match == pytest.approx(0.5)
    assert m.scores[(3, 0)] == pytest.approx(0.125)
    assert m.scores[(3, 1)] == pytest.approx(0.1875)
    assert m.outcome(mk.GAMES_TOTAL, "OVER", 3.5)[0] == pytest.approx(0.75)


@pytest.mark.parametrize("p", [0.42, 0.5, 0.53, 0.6])
@pytest.mark.parametrize("best_of", [5, 7])
@pytest.mark.parametrize("sigma", [0.0, 0.15])
def test_distributions_are_consistent(p, best_of, sigma):
    m = MatchModel(p, best_of, sigma)
    assert sum(m.scores.values()) == pytest.approx(1.0)
    assert m.p_game == pytest.approx(game_win_prob(p, sigma), abs=1e-7)
    assert m.p_match == pytest.approx(match_win_prob(p, best_of, sigma), abs=1e-7)
    assert m.points_total_dist()[1].sum() == pytest.approx(1.0, abs=1e-6)
    assert m.points_diff_dist()[1].sum() == pytest.approx(1.0, abs=1e-6)
    # handicap sides and totals sides are complementary
    for line in (-1.5, 1.5, -2.5):
        a = m.outcome(mk.GAME_HCP, "A", line)[0]
        b = m.outcome(mk.GAME_HCP, "B", -line)[0]
        assert a + b == pytest.approx(1.0)
    line = m.median_line(mk.POINTS_TOTAL)
    over, push, under = m.outcome(mk.POINTS_TOTAL, "OVER", line)
    assert push == 0 and over + under == pytest.approx(1.0, abs=1e-6)
    assert abs(over - 0.5) < 0.1
    # point handicap: A -0.5 points == A wins more points
    vals, probs = m.points_diff_dist()
    assert m.outcome(mk.POINTS_HCP, "A", -0.5)[0] == pytest.approx(probs[vals > 0].sum() / probs.sum(), abs=1e-9)


def test_integer_lines_push():
    m = MatchModel(0.55)
    win, push, lose = m.outcome(mk.GAMES_TOTAL, "OVER", 4.0)
    assert push == pytest.approx(m.scores[(3, 1)] + m.scores[(1, 3)])
    assert win + push + lose == pytest.approx(1.0)


@pytest.mark.parametrize("m", [0.05, 0.3, 0.5, 0.72, 0.95])
@pytest.mark.parametrize("sigma", [0.0, 0.1])
def test_inversion_round_trip(m, sigma):
    p = point_prob_from_match_prob(m, 5, sigma)
    assert match_win_prob(p, 5, sigma) == pytest.approx(m, abs=1e-9)


def test_fit_point_prob_recovers_truth():
    truth = match_model(0.535, 5, 0.0)
    targets = [(mk.GAME_HCP, "A", -1.5, truth.prob(mk.GAME_HCP, "A", -1.5), 1.0),
               (mk.GAMES_TOTAL, "OVER", 3.5, truth.prob(mk.GAMES_TOTAL, "OVER", 3.5), 1.0)]
    assert fit_point_prob(targets) == pytest.approx(0.535, abs=1e-3)


def _simulate(p, n, rng):
    """Independent point-by-point simulation of best-of-5 matches."""
    out = []
    for _ in range(n):
        ga = gb = 0
        pts = 0
        first = None
        while ga < 3 and gb < 3:
            a = b = 0
            while not ((a >= 11 or b >= 11) and abs(a - b) >= 2):
                if rng.random() < p:
                    a += 1
                else:
                    b += 1
            pts += a + b
            if first is None:
                first = a + b
            ga += a > b
            gb += b > a
        out.append((ga, gb, pts, first))
    return np.array(out)


def test_matches_monte_carlo():
    rng = np.random.default_rng(1)
    p = 0.53
    sims = _simulate(p, 12000, rng)
    m = MatchModel(p)
    se = lambda q: 4 * np.sqrt(q * (1 - q) / len(sims))  # noqa: E731
    p_a = float(np.mean(sims[:, 0] == 3))
    assert abs(p_a - m.p_match) < se(m.p_match)
    p30 = float(np.mean((sims[:, 0] == 3) & (sims[:, 1] == 0)))
    assert abs(p30 - m.scores[(3, 0)]) < se(m.scores[(3, 0)])
    assert abs(float(np.mean(sims[:, 2])) - m.expected_points()) < 0.6
    over = float(np.mean(sims[:, 3] > 18.5))
    assert abs(over - m.prob(mk.G1_POINTS_TOTAL, "OVER", 18.5)) < se(0.5)
