"""Odds conversion, margin removal ("de-vigging"), expected value and Kelly sizing."""
from __future__ import annotations

import math
from typing import Sequence

DEVIG_METHODS = ("power", "shin", "multiplicative", "additive")


def american_to_decimal(american: float) -> float:
    if -100 < american < 100:
        raise ValueError(f"invalid American odds: {american}")
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"invalid decimal odds: {decimal}")
    return (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def fractional_to_decimal(text: str) -> float:
    num, den = text.split("/")
    return 1.0 + float(num) / float(den)


def parse_price(value: object) -> float:
    """Parse a price into decimal odds.

    Accepts decimal (``1.85``), American with an explicit sign (``+150`` / ``-120``)
    and fractional (``5/2``).
    """
    if isinstance(value, (int, float)):
        price = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty price")
        if "/" in text:
            price = fractional_to_decimal(text)
        elif text[0] in "+-":
            price = american_to_decimal(float(text))
        else:
            price = float(text)
    if not math.isfinite(price) or price <= 1.0:
        raise ValueError(f"price must be decimal odds > 1.0, got {value!r}")
    return price


def implied(prices: Sequence[float]) -> list[float]:
    return [1.0 / p for p in prices]


def overround(prices: Sequence[float]) -> float:
    """Bookmaker margin: sum of implied probabilities minus one."""
    return sum(implied(prices)) - 1.0


def _bisect(f, lo: float, hi: float, iters: int = 56) -> float:
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def devig(prices: Sequence[float], method: str = "power") -> list[float]:
    """Convert a complete market's prices into fair probabilities that sum to 1.

    * ``multiplicative`` – scale implied probabilities proportionally.
    * ``additive`` – subtract an equal share of the margin from each outcome.
    * ``power`` – find k with sum(pi_i ** k) = 1. Puts more of the margin on
      longshots, which matches the favourite-longshot bias usually seen in books.
    * ``shin`` – Shin (1993) model of insider trading; also longshot-weighted.
    """
    if len(prices) < 2:
        raise ValueError("need at least two outcomes to de-vig")
    pi = implied(prices)
    total = sum(pi)
    n = len(pi)
    if method == "multiplicative":
        return [x / total for x in pi]
    if method == "additive":
        out = [max(x - (total - 1.0) / n, 1e-9) for x in pi]
        s = sum(out)
        return [x / s for x in out]
    if method == "power":
        if abs(total - 1.0) < 1e-12:
            return list(pi)
        k = _bisect(lambda k: sum(x ** k for x in pi) - 1.0, 1e-3, 50.0)
        out = [x ** k for x in pi]
        s = sum(out)
        return [x / s for x in out]
    if method == "shin":
        if total <= 1.0:
            return [x / total for x in pi]

        def probs(z: float) -> list[float]:
            return [
                (math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / total) - z) / (2.0 * (1.0 - z))
                for x in pi
            ]

        z = _bisect(lambda z: sum(probs(z)) - 1.0, 0.0, 0.999)
        out = probs(z)
        s = sum(out)
        return [x / s for x in out]
    raise ValueError(f"unknown devig method {method!r}; choose from {DEVIG_METHODS}")


def apply_margin(probs: Sequence[float], margin: float) -> list[float]:
    """Inverse of power de-vig: turn fair probabilities into prices with ``margin``."""
    if margin <= 0:
        return [1.0 / p for p in probs]
    target = 1.0 + margin
    # find e in (0, 1] with sum(p ** e) = target
    e = _bisect(lambda e: sum(p ** e for p in probs) - target, 1e-3, 1.0)
    return [1.0 / (p ** e) for p in probs]


def expected_value(price: float, p_win: float, p_push: float = 0.0) -> float:
    """Expected profit per unit staked."""
    p_lose = max(0.0, 1.0 - p_win - p_push)
    return p_win * (price - 1.0) - p_lose


def fair_price(p_win: float, p_push: float = 0.0) -> float:
    """Break-even decimal odds (pushes return the stake)."""
    if p_win <= 0:
        return float("inf")
    p_lose = max(0.0, 1.0 - p_win - p_push)
    return 1.0 + p_lose / p_win


def kelly_fraction(price: float, p_win: float, p_push: float = 0.0) -> float:
    """Full-Kelly bankroll fraction (0 if the bet has no edge)."""
    b = price - 1.0
    p_lose = max(0.0, 1.0 - p_win - p_push)
    if b <= 0 or p_win + p_lose <= 0:
        return 0.0
    f = (b * p_win - p_lose) / (b * (p_win + p_lose))
    return max(0.0, f)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
