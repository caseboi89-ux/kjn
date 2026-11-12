"""CSV input/output.

``results.csv`` (one row per finished match)::

    match_id,start_time,tournament,player_a,player_b,score[,best_of]
    m1,2026-11-12T09:30,Czech Liga Pro,Player One,Player Two,11-7 9-11 11-5 11-8

``odds.csv`` (one row per price)::

    match_id,start_time,tournament,player_a,player_b,book,market,selection,line,price[,snapshot][,best_of]
    m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,ML,A,,1.72
    m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAME_HCP,B,1.5,1.66

``selection`` may also be a player's name, ``home``/``away``, ``over``/``under``;
``price`` may be decimal, American (``+150``) or fractional (``5/2``);
``snapshot`` is ``open``, ``close`` or ``current`` (the default).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import markets as mk
from .odds import parse_price


def parse_time(text: str) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        t = None
        for fmt in ("%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                t = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if t is None:
            raise ValueError(f"unrecognised time {text!r}")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


@dataclass
class Result:
    match_id: str
    start_time: datetime | None
    tournament: str
    player_a: str
    player_b: str
    score: mk.MatchScore
    best_of: int = 5


@dataclass
class Quote:
    match_id: str
    start_time: datetime | None
    tournament: str
    player_a: str
    player_b: str
    book: str
    market: str
    selection: str
    line: float | None
    price: float
    snapshot: str = "current"
    best_of: int = 5


def _read_rows(source: str | Path | io.TextIOBase) -> list[dict]:
    """A :class:`~pathlib.Path` is read from disk; a ``str`` is CSV text."""
    if isinstance(source, Path):
        with open(source, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    if isinstance(source, str):
        if not source.strip():
            return []
        return list(csv.DictReader(io.StringIO(source.lstrip("﻿"))))
    return list(csv.DictReader(source))


def _clean(row: dict) -> dict:
    return {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}


def _match_id(row: dict) -> str:
    if row.get("match_id"):
        return row["match_id"]
    return f"{row.get('start_time', '')[:10]}|{row['player_a']}|{row['player_b']}"


def _best_of(row: dict) -> int:
    value = row.get("best_of")
    if not value:
        return 5
    bo = int(value)
    mk.games_to_win(bo)
    return bo


def load_results(source) -> tuple[list[Result], list[str]]:
    """Load results; returns (results sorted by time, list of row errors)."""
    results, errors = [], []
    for i, raw in enumerate(_read_rows(source), start=2):
        row = _clean(raw)
        try:
            if not row.get("player_a") or not row.get("player_b"):
                raise ValueError("player_a and player_b are required")
            results.append(Result(
                match_id=_match_id(row),
                start_time=parse_time(row.get("start_time", "")),
                tournament=row.get("tournament", ""),
                player_a=row["player_a"],
                player_b=row["player_b"],
                score=mk.parse_score(row.get("score", "")),
                best_of=_best_of(row),
            ))
        except (ValueError, KeyError) as exc:
            errors.append(f"results row {i}: {exc}")
    results.sort(key=lambda r: (r.start_time is None, r.start_time or datetime.min.replace(tzinfo=timezone.utc)))
    return results, errors


def load_odds(source) -> tuple[list[Quote], list[str]]:
    quotes, errors = [], []
    for i, raw in enumerate(_read_rows(source), start=2):
        row = _clean(raw)
        try:
            if not row.get("player_a") or not row.get("player_b"):
                raise ValueError("player_a and player_b are required")
            market = mk.normalize_market(row.get("market", ""))
            selection = mk.normalize_selection(row.get("selection", ""), row["player_a"], row["player_b"])
            line_text = row.get("line", "")
            line = float(line_text) if line_text else None
            if market in mk.LINE_MARKETS and line is None:
                raise ValueError(f"{market} needs a line")
            if market not in mk.LINE_MARKETS:
                line = None
            snapshot = (row.get("snapshot") or "current").lower()
            if snapshot not in ("open", "close", "current"):
                raise ValueError(f"snapshot must be open/close/current, got {snapshot!r}")
            quotes.append(Quote(
                match_id=_match_id(row),
                start_time=parse_time(row.get("start_time", "")),
                tournament=row.get("tournament", ""),
                player_a=row["player_a"],
                player_b=row["player_b"],
                book=row.get("book") or "unknown",
                market=market,
                selection=selection,
                line=line,
                price=parse_price(row.get("price", "")),
                snapshot=snapshot,
                best_of=_best_of(row),
            ))
        except (ValueError, KeyError) as exc:
            errors.append(f"odds row {i}: {exc}")
    return quotes, errors


RESULT_FIELDS = ["match_id", "start_time", "tournament", "player_a", "player_b", "score", "best_of"]
ODDS_FIELDS = ["match_id", "start_time", "tournament", "player_a", "player_b", "book", "market",
               "selection", "line", "price", "snapshot", "best_of"]


def _iso(t: datetime | None) -> str:
    return t.isoformat(timespec="minutes") if t else ""


def write_results(path: str | Path, results: list[Result]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(RESULT_FIELDS)
        for r in results:
            w.writerow([r.match_id, _iso(r.start_time), r.tournament, r.player_a, r.player_b, str(r.score), r.best_of])


def write_odds(path: str | Path, quotes: list[Quote]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ODDS_FIELDS)
        for q in quotes:
            w.writerow([q.match_id, _iso(q.start_time), q.tournament, q.player_a, q.player_b, q.book, q.market,
                        q.selection, "" if q.line is None else f"{q.line:g}", f"{q.price:.2f}", q.snapshot, q.best_of])
