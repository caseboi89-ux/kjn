import math

import pytest

from ttedge import odds


def test_price_parsing():
    assert odds.parse_price("1.85") == 1.85
    assert odds.parse_price("+150") == pytest.approx(2.5)
    assert odds.parse_price("-200") == pytest.approx(1.5)
    assert odds.parse_price("5/2") == pytest.approx(3.5)
    assert odds.parse_price(2) == 2.0
    for bad in ("", "1.0", "0.5", "abc", "+50"):
        with pytest.raises(ValueError):
            odds.parse_price(bad)


def test_american_round_trip():
    for dec in (1.2, 1.91, 2.0, 3.75):
        assert odds.american_to_decimal(odds.decimal_to_american(dec)) == pytest.approx(dec)


@pytest.mark.parametrize("method", odds.DEVIG_METHODS)
def test_devig_sums_to_one_and_keeps_order(method):
    prices = [1.55, 2.45]
    probs = odds.devig(prices, method)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1]
    multi = odds.devig([4.5, 3.2, 6.0, 8.0, 2.9], method)
    assert sum(multi) == pytest.approx(1.0)


def test_devig_without_margin_is_identity():
    fair = [0.62, 0.38]
    prices = [1 / p for p in fair]
    for method in odds.DEVIG_METHODS:
        assert odds.devig(prices, method) == pytest.approx(fair, abs=1e-9)


def test_power_and_shin_put_more_margin_on_longshot():
    prices = [1.25, 3.80]
    mult = odds.devig(prices, "multiplicative")
    for method in ("power", "shin"):
        p = odds.devig(prices, method)
        assert p[1] < mult[1]


def test_apply_margin_inverts_power_devig():
    fair = [0.7, 0.3]
    prices = odds.apply_margin(fair, 0.06)
    assert odds.overround(prices) == pytest.approx(0.06, abs=1e-9)
    assert odds.devig(prices, "power") == pytest.approx(fair, abs=1e-9)


def test_ev_kelly_and_pushes():
    assert odds.expected_value(2.2, 0.5) == pytest.approx(0.1)
    assert odds.kelly_fraction(2.2, 0.5) == pytest.approx((1.2 * 0.5 - 0.5) / 1.2)
    assert odds.kelly_fraction(1.8, 0.5) == 0.0
    # a push returns the stake: EV counts only settled outcomes
    assert odds.expected_value(2.0, 0.45, 0.10) == pytest.approx(0.45 - 0.45)
    assert odds.fair_price(0.45, 0.10) == pytest.approx(2.0)
    assert odds.fair_price(0.5) == 2.0
    assert math.isinf(odds.fair_price(0.0))
