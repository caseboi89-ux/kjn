import pytest

from ttedge import markets as mk


def test_parse_score_formats():
    s = mk.parse_score("11-7 9-11 11-5 13-11")
    assert (s.games_a, s.games_b) == (3, 1)
    assert s.points_a == 44 and s.points_b == 34
    assert mk.parse_score("3:1").game_points == ()
    s2 = mk.parse_score("3-1 (11-7, 9-11, 11-5, 13-11)")
    assert s2 == s
    assert mk.parse_score("7-11, 11-9, 5-11, 11-13").games_b == 3


@pytest.mark.parametrize("bad", ["", "11-10", "12-9 11-3 11-2", "3-1 (11-7, 11-9, 11-5)", "2-2", "9-8"])
def test_parse_score_rejects_bad(bad):
    with pytest.raises(ValueError):
        mk.parse_score(bad)


def test_settlement():
    s = mk.parse_score("11-7 9-11 11-5 11-8")  # A wins 3-1, points 42-31
    assert mk.settle(mk.ML, "A", None, s) == "win"
    assert mk.settle(mk.ML, "B", None, s) == "lose"
    assert mk.settle(mk.GAME_HCP, "A", -1.5, s) == "win"
    assert mk.settle(mk.GAME_HCP, "A", -2.5, s) == "lose"
    assert mk.settle(mk.GAME_HCP, "B", 1.5, s) == "lose"
    assert mk.settle(mk.GAME_HCP, "A", -2.0, s) == "push"
    assert mk.settle(mk.GAMES_TOTAL, "OVER", 3.5, s) == "win"
    assert mk.settle(mk.GAMES_TOTAL, "UNDER", 3.5, s) == "lose"
    assert mk.settle(mk.GAMES_TOTAL, "UNDER", 4.0, s) == "push"
    assert mk.settle(mk.CORRECT_SCORE, "3-1", None, s) == "win"
    assert mk.settle(mk.POINTS_TOTAL, "OVER", 72.5, s) == "win"
    assert mk.settle(mk.POINTS_HCP, "B", 10.5, s) == "lose"
    assert mk.settle(mk.POINTS_HCP, "B", 11.5, s) == "win"
    assert mk.settle(mk.G1_ML, "A", None, s) == "win"
    assert mk.settle(mk.G1_POINTS_TOTAL, "UNDER", 18.5, s) == "win"
    games_only = mk.parse_score("3-1")
    assert mk.settle(mk.POINTS_TOTAL, "OVER", 72.5, games_only) is None


def test_normalisation_and_groups():
    assert mk.normalize_market("Set Handicap") == mk.GAME_HCP
    assert mk.normalize_market("moneyline") == mk.ML
    assert mk.normalize_selection("Home") == "A"
    assert mk.normalize_selection("Jan Novak", "Jan Novak", "Petr Svoboda") == "A"
    assert mk.normalize_selection("3:1") == "3-1"
    assert mk.group_key(mk.GAME_HCP, "A", -1.5) == mk.group_key(mk.GAME_HCP, "B", 1.5)
    assert mk.swap_selection(mk.CORRECT_SCORE, "3-1") == "1-3"
    assert mk.correct_score_selections(7)[:2] == ["4-0", "4-1"]
