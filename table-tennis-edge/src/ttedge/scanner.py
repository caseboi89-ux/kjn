"""Find mispriced lines.

For every price on the board the scanner builds a fair probability from up to
three independent sources and reports the price if it beats that fair value:

1. **Leave-one-out market consensus** – every other book's margin-free price
   for the same selection (sharp books weighted up). A book is never compared
   against itself.
2. **Point model** – the consensus match price is inverted to a point-win
   probability, which prices every derivative (handicaps, totals, correct
   score, points) consistently. Books often hang derivatives with cruder
   rules than their match-winner line, so this catches inconsistent derivatives
   even when only a couple of books are quoting.
3. **Ratings** (optional) – a :class:`~ttedge.ratings.PointRating` trained on
   past results, blended in when both players have enough history.

It also reports cross-book arbitrage (best prices summing to < 100%).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import markets as mk
from .data import Quote
from .markov import fit_point_prob, match_model, point_prob_from_match_prob
from .odds import devig, expected_value, fair_price, kelly_fraction, logit, sigmoid
from .ratings import player_key


@dataclass
class ScanConfig:
    devig_method: str = "power"
    min_edge: float = 0.03            # minimum expected value per unit staked
    max_edge: float = 0.30            # above this the line is probably stale or a data error
    min_books: int = 1                # other books needed for a consensus price
    sharp_books: tuple[str, ...] = ("pinnacle", "sbobet", "betfair", "matchbook", "demosharp")
    sharp_weight: float = 3.0
    model_weight: float = 0.3         # weight of ratings vs market in the point probability
    min_model_matches: int = 15       # history each player needs before ratings are used
    derivative_consensus_weight: float = 0.5  # direct consensus vs point model for derivatives
    game_sigma: float = 0.0
    kelly_multiplier: float = 0.25
    max_stake: float = 0.02           # cap per bet, fraction of bankroll
    min_odds: float = 1.20
    max_odds: float = 6.00
    inconsistency_threshold: float = 0.04  # flag derivatives this far from the book's own ML
    bet_into_sharp: bool = False      # allow "opportunities" at a sharp book judged only by softer books

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sharp_books"] = list(self.sharp_books)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScanConfig":
        base = asdict(cls())
        clean = {}
        for k, v in data.items():
            if k not in base:
                continue
            if k == "sharp_books":
                if isinstance(v, str):
                    v = [s.strip() for s in v.split(",") if s.strip()]
                clean[k] = tuple(str(s).lower() for s in v)
            elif k in ("min_books", "min_model_matches"):
                clean[k] = int(v)
            elif k == "devig_method":
                clean[k] = str(v)
            elif k == "bet_into_sharp":
                clean[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            else:
                clean[k] = float(v)
        return cls(**{**base, **clean, "sharp_books": clean.get("sharp_books", cls().sharp_books)})


@dataclass
class Evaluation:
    match_id: str
    start_time: str
    tournament: str
    player_a: str
    player_b: str
    best_of: int
    book: str
    market: str
    selection: str
    line: float | None
    price: float
    description: str
    p_win: float
    p_push: float
    fair_price: float
    edge: float
    kelly: float
    stake: float
    consensus_prob: float | None
    consensus_books: int
    model_prob: float | None
    rating_match_prob: float | None
    market_match_prob: float | None
    book_prob: float | None
    book_margin: float | None
    own_ml_prob: float | None
    flags: list[str] = field(default_factory=list)

    @property
    def is_opportunity(self) -> bool:
        return "opportunity" in self.flags


@dataclass
class Arbitrage:
    match_id: str
    player_a: str
    player_b: str
    market: str
    line: float | None
    legs: list[dict]
    margin: float  # guaranteed return on total stake

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventSummary:
    match_id: str
    start_time: str
    tournament: str
    player_a: str
    player_b: str
    best_of: int
    books: list[str]
    p_point: float | None
    p_point_market: float | None
    p_point_rating: float | None
    p_match: float | None
    rating_matches: tuple[int, int] | None
    fair: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    evaluations: list[Evaluation]
    arbitrages: list[Arbitrage]
    events: list[EventSummary]

    @property
    def opportunities(self) -> list[Evaluation]:
        opps = [e for e in self.evaluations if e.is_opportunity]
        opps.sort(key=lambda e: e.edge, reverse=True)
        return opps

    def to_dict(self) -> dict:
        return {
            "opportunities": [asdict(e) for e in self.opportunities],
            "evaluations": [asdict(e) for e in self.evaluations],
            "arbitrages": [a.to_dict() for a in self.arbitrages],
            "events": [e.to_dict() for e in self.events],
        }


def _blend(p1: float | None, p2: float | None, w2: float) -> float | None:
    if p1 is None:
        return p2
    if p2 is None or w2 <= 0:
        return p1
    return sigmoid((1 - w2) * logit(p1) + w2 * logit(p2))


def _iso(t: datetime | None) -> str:
    return t.isoformat(timespec="minutes") if t else ""


class _Event:
    """All quotes for one match, oriented to a single A/B convention."""

    def __init__(self, quotes: list[Quote], cfg: ScanConfig):
        first = quotes[0]
        self.cfg = cfg
        self.match_id = first.match_id
        self.start_time = first.start_time
        self.tournament = first.tournament
        self.player_a, self.player_b = first.player_a, first.player_b
        self.best_of = first.best_of
        ka, kb = player_key(self.player_a), player_key(self.player_b)
        # prices[book][group_key][selection] = price
        self.prices: dict[str, dict[tuple, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for q in quotes:
            qa, qb = player_key(q.player_a), player_key(q.player_b)
            sel = q.selection
            if (qa, qb) == (kb, ka):
                sel = mk.swap_selection(q.market, sel)
            elif (qa, qb) != (ka, kb):
                continue
            gk = mk.group_key(q.market, sel, q.line)
            self.prices[q.book][gk][sel] = q.price
        self.books = sorted(self.prices)
        # margin-free probabilities for every complete market of every book
        self.fair: dict[str, dict[tuple, dict[str, float]]] = defaultdict(dict)
        self.margin: dict[str, dict[tuple, float]] = defaultdict(dict)
        for book, groups in self.prices.items():
            for gk, sels in groups.items():
                needed = mk.group_selections(gk[0], self.best_of)
                if all(s in sels for s in needed):
                    prices = [sels[s] for s in needed]
                    probs = devig(prices, cfg.devig_method)
                    self.fair[book][gk] = dict(zip(needed, probs))
                    self.margin[book][gk] = sum(1 / p for p in prices) - 1
        self._pm_cache: dict[str | None, float | None] = {}
        # Derivative prices are trusted in proportion to how well they agree with
        # the same book's own match-winner price: a book that hangs totals with a
        # rule of thumb should not drag the consensus for everyone else.
        self.trust: dict[str, dict[tuple, float]] = defaultdict(dict)
        for book, groups in self.fair.items():
            p_own = self.own_point_prob(book)
            model = match_model(p_own, self.best_of, cfg.game_sigma) if p_own is not None else None
            for gk, probs in groups.items():
                if gk[0] == mk.ML:
                    self.trust[book][gk] = 1.0
                    continue
                if model is None:
                    self.trust[book][gk] = 0.5
                    continue
                dev = max(abs(probs[s] - model.prob(gk[0], s, mk.selection_line(gk[0], s, gk[1]))) for s in probs)
                self.trust[book][gk] = 1.0 / (1.0 + (dev / cfg.inconsistency_threshold) ** 2)

    def _weight(self, book: str) -> float:
        return self.cfg.sharp_weight if self.is_sharp(book) else 1.0

    def is_sharp(self, book: str) -> bool:
        return book.lower() in self.cfg.sharp_books

    def sharp_quoting(self, gk: tuple, exclude: str | None = None) -> int:
        return sum(1 for b in self.fair if gk in self.fair[b] and b != exclude and self.is_sharp(b))

    def consensus(self, gk: tuple, exclude: str | None = None) -> tuple[dict[str, float] | None, int]:
        books = [b for b in self.fair if gk in self.fair[b] and b != exclude]
        if len(books) < max(1, self.cfg.min_books):
            return None, len(books)
        sels = mk.group_selections(gk[0], self.best_of)
        weights = {b: self._weight(b) * self.trust[b].get(gk, 1.0) for b in books}
        total_w = sum(weights.values())
        raw = {s: sigmoid(sum(weights[b] * logit(self.fair[b][gk][s]) for b in books) / total_w) for s in sels}
        norm = sum(raw.values())
        return {s: v / norm for s, v in raw.items()}, len(books)

    def market_point_prob(self, exclude: str | None = None) -> float | None:
        if exclude in self._pm_cache:
            return self._pm_cache[exclude]
        sigma, bo = self.cfg.game_sigma, self.best_of
        ml, _ = self.consensus((mk.ML, None), exclude)
        if ml is not None:
            p = point_prob_from_match_prob(ml["A"], bo, sigma)
        else:
            groups = {gk for b in self.fair if b != exclude for gk in self.fair[b]}
            targets = []
            for gk in groups:
                cons, _ = self.consensus(gk, exclude)
                if cons is None:
                    continue
                for sel, prob in cons.items():
                    line = mk.selection_line(gk[0], sel, gk[1])
                    targets.append((gk[0], sel, line, prob, 1.0))
            p = fit_point_prob(targets, bo, sigma) if targets else None
        self._pm_cache[exclude] = p
        return p

    def own_point_prob(self, book: str) -> float | None:
        ml = self.fair.get(book, {}).get((mk.ML, None))
        if ml is None:
            return None
        return point_prob_from_match_prob(ml["A"], self.best_of, self.cfg.game_sigma)


def scan(quotes: list[Quote], cfg: ScanConfig | None = None, ratings=None,
         snapshot: str | None = "current") -> ScanResult:
    """Evaluate every quote. ``ratings`` is an optional fitted rating model."""
    cfg = cfg or ScanConfig()
    by_match: dict[str, list[Quote]] = defaultdict(list)
    for q in quotes:
        if snapshot is None or q.snapshot == snapshot:
            by_match[q.match_id].append(q)
    evaluations: list[Evaluation] = []
    arbs: list[Arbitrage] = []
    events: list[EventSummary] = []
    for match_quotes in by_match.values():
        ev_list, arb_list, summary = scan_event(match_quotes, cfg, ratings)
        evaluations.extend(ev_list)
        arbs.extend(arb_list)
        events.append(summary)
    events.sort(key=lambda e: (e.start_time, e.match_id))
    arbs.sort(key=lambda a: a.margin, reverse=True)
    return ScanResult(evaluations, arbs, events)


def scan_event(quotes: list[Quote], cfg: ScanConfig, ratings=None):
    ev = _Event(quotes, cfg)
    sigma, bo = cfg.game_sigma, ev.best_of

    p_rating = rating_mp = None
    rating_matches = None
    if ratings is not None:
        pred = ratings.predict(ev.player_a, ev.player_b, bo, ev.start_time)
        rating_matches = (pred.matches_a, pred.matches_b)
        if pred.min_matches >= cfg.min_model_matches:
            p_rating, rating_mp = pred.p_point, pred.p_match

    def blended(exclude: str | None) -> float | None:
        return _blend(ev.market_point_prob(exclude), p_rating, cfg.model_weight)

    evaluations: list[Evaluation] = []
    for book in ev.books:
        p_mkt = ev.market_point_prob(book)
        p_fair = _blend(p_mkt, p_rating, cfg.model_weight)
        model = match_model(p_fair, bo, sigma) if p_fair is not None else None
        p_own = ev.own_point_prob(book)
        own_model = match_model(p_own, bo, sigma) if p_own is not None else None
        market_mp = match_model(p_mkt, bo, sigma).p_match if p_mkt is not None else None
        for gk, sels in ev.prices[book].items():
            market = gk[0]
            cons, n_cons = ev.consensus(gk, exclude=book)
            for sel, price in sels.items():
                line = mk.selection_line(market, sel, gk[1])
                flags: list[str] = []
                cons_p = cons[sel] if cons else None
                model_p = None
                if model is not None:
                    win, push, lose = model.outcome(market, sel, line)
                    model_p = win / (win + lose)
                    settled = 1.0 - push
                    if market == mk.ML or cons_p is None:
                        f = model_p
                    else:
                        f = sigmoid((1 - cfg.derivative_consensus_weight) * logit(model_p)
                                    + cfg.derivative_consensus_weight * logit(cons_p))
                    p_win, p_push = f * settled, push
                    if p_mkt is None:
                        flags.append("ratings_only")
                elif cons_p is not None:
                    p_win, p_push = cons_p, 0.0
                    flags.append("consensus_only")
                else:
                    continue
                edge = expected_value(price, p_win, p_push)
                kelly = kelly_fraction(price, p_win, p_push)
                stake = min(cfg.max_stake, cfg.kelly_multiplier * kelly)
                book_p = ev.fair.get(book, {}).get(gk, {}).get(sel)
                own_p = None
                if own_model is not None and market != mk.ML:
                    own_p = own_model.prob(market, sel, line)
                    if book_p is not None and abs(book_p - own_p) >= cfg.inconsistency_threshold:
                        flags.append("inconsistent_with_own_ml")
                if cons_p is not None and price > 1.0 / cons_p:
                    flags.append("beats_consensus")
                if n_cons and n_cons < 2:
                    flags.append("thin_consensus")
                if rating_mp is not None and market_mp is not None and abs(rating_mp - market_mp) > 0.15:
                    flags.append("ratings_disagree_with_market")
                if edge > cfg.max_edge:
                    flags.append("suspicious_edge")
                blocked = False
                if ev.is_sharp(book) and not ev.sharp_quoting((mk.ML, None), exclude=book):
                    # a sharp book judged only by softer books: usually the soft books are wrong
                    flags.append("sharp_book_vs_soft_consensus")
                    blocked = not cfg.bet_into_sharp
                if (edge >= cfg.min_edge and cfg.min_odds <= price <= cfg.max_odds
                        and edge <= cfg.max_edge and not blocked):
                    flags.insert(0, "opportunity")
                evaluations.append(Evaluation(
                    match_id=ev.match_id, start_time=_iso(ev.start_time), tournament=ev.tournament,
                    player_a=ev.player_a, player_b=ev.player_b, best_of=bo, book=book, market=market,
                    selection=sel, line=line, price=price,
                    description=mk.describe(market, sel, line, ev.player_a, ev.player_b),
                    p_win=round(p_win, 5), p_push=round(p_push, 5),
                    fair_price=round(fair_price(p_win, p_push), 4), edge=round(edge, 5),
                    kelly=round(kelly, 5), stake=round(stake, 5),
                    consensus_prob=None if cons_p is None else round(cons_p, 5), consensus_books=n_cons,
                    model_prob=None if model_p is None else round(model_p, 5),
                    rating_match_prob=None if rating_mp is None else round(rating_mp, 4),
                    market_match_prob=None if market_mp is None else round(market_mp, 4),
                    book_prob=None if book_p is None else round(book_p, 5),
                    book_margin=None if gk not in ev.margin.get(book, {}) else round(ev.margin[book][gk], 4),
                    own_ml_prob=None if own_p is None else round(own_p, 5),
                    flags=flags,
                ))

    # cross-book arbitrage
    arbs: list[Arbitrage] = []
    all_groups = {gk for b in ev.prices for gk in ev.prices[b]}
    for gk in all_groups:
        sels = mk.group_selections(gk[0], bo)
        best: dict[str, tuple[float, str]] = {}
        for book in ev.books:
            for sel, price in ev.prices[book].get(gk, {}).items():
                if sel not in best or price > best[sel][0]:
                    best[sel] = (price, book)
        if not all(s in best for s in sels):
            continue
        inv = sum(1.0 / best[s][0] for s in sels)
        if inv < 1.0:
            legs = [{"selection": s,
                     "description": mk.describe(gk[0], s, mk.selection_line(gk[0], s, gk[1]), ev.player_a, ev.player_b),
                     "book": best[s][1], "price": best[s][0],
                     "stake_share": round((1.0 / best[s][0]) / inv, 4)} for s in sels]
            arbs.append(Arbitrage(ev.match_id, ev.player_a, ev.player_b, gk[0], gk[1], legs, round(1.0 / inv - 1.0, 5)))

    p_all = blended(None)
    fair_rows = []
    if p_all is not None:
        fair_rows = match_model(p_all, bo, sigma).fair_table()
        for r in fair_rows:
            r["description"] = mk.describe(r["market"], r["selection"], r["line"], ev.player_a, ev.player_b)
    summary = EventSummary(
        match_id=ev.match_id, start_time=_iso(ev.start_time), tournament=ev.tournament,
        player_a=ev.player_a, player_b=ev.player_b, best_of=bo, books=ev.books,
        p_point=None if p_all is None else round(p_all, 5),
        p_point_market=None if ev.market_point_prob(None) is None else round(ev.market_point_prob(None), 5),
        p_point_rating=None if p_rating is None else round(p_rating, 5),
        p_match=None if p_all is None else round(match_model(p_all, bo, sigma).p_match, 5),
        rating_matches=rating_matches, fair=fair_rows,
    )
    return evaluations, arbs, summary
