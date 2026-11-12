from datetime import datetime, timezone

import pytest

from ttedge import markets as mk
from ttedge.data import Quote, load_odds
from ttedge.markov import match_model, point_prob_from_match_prob
from ttedge.odds import apply_margin
from ttedge.scanner import ScanConfig, scan

T = datetime(2026, 11, 13, 10, 0, tzinfo=timezone.utc)


def q(book, market, sel, price, line=None, a="Ann", b="Bea", mid="m1"):
    return Quote(mid, T, "Test Cup", a, b, book, market, sel, line, price)


def board(p_a=0.6, margin=0.05, books=("B1", "B2", "B3")):
    out = []
    for book in books:
        pa, pb = apply_margin([p_a, 1 - p_a], margin)
        out += [q(book, mk.ML, "A", round(pa, 3)), q(book, mk.ML, "B", round(pb, 3))]
    return out


def test_fair_board_has_no_opportunities():
    res = scan(board(), ScanConfig(min_edge=0.01))
    assert res.opportunities == []
    assert res.arbitrages == []
    assert res.events[0].p_match == pytest.approx(0.6, abs=1e-3)


def test_outlier_price_is_flagged_and_book_not_compared_with_itself():
    quotes = board() + [q("Soft", mk.ML, "A", 1.40), q("Soft", mk.ML, "B", 2.90)]
    res = scan(quotes, ScanConfig())
    opp = res.opportunities
    assert len(opp) == 1
    o = opp[0]
    assert (o.book, o.selection) == ("Soft", "B")
    assert o.p_win == pytest.approx(0.4, abs=2e-3)
    assert o.edge == pytest.approx(0.4 * 2.90 - 1, abs=0.01)
    assert o.consensus_books == 3
    assert "beats_consensus" in o.flags
    # the same price compared with a board of only itself has no reference
    alone = scan([q("Solo", mk.ML, "A", 1.40), q("Solo", mk.ML, "B", 2.90)], ScanConfig())
    assert alone.opportunities == []


def test_reversed_player_order_is_aligned():
    quotes = board() + [q("Flip", mk.ML, "A", 2.90, a="Bea", b="Ann"), q("Flip", mk.ML, "B", 1.40, a="Bea", b="Ann")]
    res = scan(quotes, ScanConfig())
    assert len(res.opportunities) == 1
    o = res.opportunities[0]
    assert o.book == "Flip" and o.selection == "B" and o.player_b == "Bea"


def test_derivative_inconsistent_with_own_match_price():
    sigma = 0.0
    p = point_prob_from_match_prob(0.6, 5, sigma)
    model = match_model(p, 5, sigma)
    over = model.prob(mk.GAMES_TOTAL, "OVER", 3.5)
    quotes = board()
    for book in ("B1", "B2"):
        po, pu = apply_margin([over, 1 - over], 0.05)
        quotes += [q(book, mk.GAMES_TOTAL, "OVER", round(po, 3), 3.5), q(book, mk.GAMES_TOTAL, "UNDER", round(pu, 3), 3.5)]
    # B3 hangs a lazy total (~61% over) although its own 60/40 ML implies ~74%
    quotes += [q("B3", mk.GAMES_TOTAL, "OVER", 1.55, 3.5), q("B3", mk.GAMES_TOTAL, "UNDER", 2.45, 3.5)]
    res = scan(quotes, ScanConfig(game_sigma=sigma))
    hits = [o for o in res.opportunities if o.market == mk.GAMES_TOTAL]
    assert len(hits) == 1
    h = hits[0]
    assert h.book == "B3" and h.selection == "OVER"
    assert "inconsistent_with_own_ml" in h.flags
    assert h.p_win == pytest.approx(over, abs=0.01)


def test_model_prices_derivatives_with_no_other_quotes():
    # fair P(A -1.5) is ~39.5% when A is a 60% favourite
    quotes = board() + [q("B3", mk.GAME_HCP, "A", 2.85, -1.5), q("B3", mk.GAME_HCP, "B", 1.45, 1.5)]
    res = scan(quotes, ScanConfig())
    hit = [o for o in res.opportunities if o.market == mk.GAME_HCP]
    assert hit and hit[0].selection == "A" and hit[0].consensus_prob is None


def test_arbitrage_detected():
    quotes = [q("X", mk.ML, "A", 2.15), q("X", mk.ML, "B", 1.70), q("Y", mk.ML, "A", 1.80), q("Y", mk.ML, "B", 2.05)]
    res = scan(quotes, ScanConfig())
    assert len(res.arbitrages) == 1
    arb = res.arbitrages[0]
    assert arb.margin == pytest.approx(1 / (1 / 2.15 + 1 / 2.05) - 1, abs=1e-4)
    assert {leg["book"] for leg in arb.legs} == {"X", "Y"}
    assert sum(leg["stake_share"] for leg in arb.legs) == pytest.approx(1.0, abs=1e-3)


def test_sharp_book_not_flagged_against_soft_books_by_default():
    quotes = board(books=("Soft1", "Soft2")) + [q("Pinnacle", mk.ML, "A", 1.50), q("Pinnacle", mk.ML, "B", 2.80)]
    res = scan(quotes, ScanConfig())
    assert not [o for o in res.opportunities if o.book == "Pinnacle"]
    flagged = [e for e in res.evaluations if e.book == "Pinnacle" and e.selection == "B"][0]
    assert "sharp_book_vs_soft_consensus" in flagged.flags
    res2 = scan(quotes, ScanConfig(bet_into_sharp=True))
    assert [o for o in res2.opportunities if o.book == "Pinnacle"]


def test_suspicious_edges_are_not_opportunities():
    quotes = board() + [q("Typo", mk.ML, "A", 1.10), q("Typo", mk.ML, "B", 9.00)]
    res = scan(quotes, ScanConfig(max_odds=20))
    ev = [e for e in res.evaluations if e.book == "Typo" and e.selection == "B"][0]
    assert "suspicious_edge" in ev.flags and not ev.is_opportunity


def test_csv_loading_with_aliases():
    text = """match_id,start_time,tournament,player_a,player_b,book,market,selection,line,price
m9,2026-11-13 10:00,Liga Pro,Jan Novak,Petr Svoboda,BookA,moneyline,Jan Novak,,-150
m9,2026-11-13 10:00,Liga Pro,Jan Novak,Petr Svoboda,BookA,moneyline,away,,+125
m9,2026-11-13 10:00,Liga Pro,Jan Novak,Petr Svoboda,BookA,set handicap,Petr Svoboda,1.5,1.62
m9,2026-11-13 10:00,Liga Pro,Jan Novak,Petr Svoboda,BookA,total games,over,,1.5
m9,2026-11-13 10:00,Liga Pro,Jan Novak,Petr Svoboda,BookA,weird,over,3.5,1.5
"""
    quotes, errors = load_odds(text)
    assert [x.selection for x in quotes] == ["A", "B", "B"]
    assert quotes[0].price == pytest.approx(1 + 100 / 150)
    assert quotes[2].line == 1.5 and quotes[2].market == mk.GAME_HCP
    assert len(errors) == 2
