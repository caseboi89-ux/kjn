# TT Edge: find mispriced table tennis lines

A local app that prices every common table tennis market from first
principles, compares each bookmaker's price with fair value, and lists the
lines worth betting. It also backtests the method on your own history so
you can see whether it beats the closing line.

* **Scanner**: value bets and cross-book arbitrage across match winner, game
  handicap, total games, correct score, total points, point handicap, game-1
  winner and game-1 points.
* **Point model**: exact point → game → match maths (11 points, win by 2,
  best of 5/7). One point-win probability prices every market consistently.
* **Ratings**: point-level Kalman ratings learned from every point in your
  results, with a win/loss Elo baseline for comparison.
* **Backtest**: walk-forward, no look-ahead. Reports ROI with its standard
  error, closing-line value (CLV), prediction quality against the closing
  market, and a favourite-longshot calibration table.
* **Dashboard**: a local web page (no internet, no dependencies beyond numpy),
  plus a CLI.

Research behind the design (market structure, likely sources of edge,
integrity risks, data providers): [docs/RESEARCH.md](docs/RESEARCH.md).

> **No real prices are bundled.** The demo uses synthetic players and
> bookmakers so the pipeline can be tried end to end. To find real
> mispriced lines, load real odds (see *Getting real data*).

## Browser version (no install)

`web/tt-edge-standalone.html` is the whole app in one file: open it in any
browser, or use the published claude.ai artifact built from the same source.
Everything runs locally in the page (engine port in `web/engine.js`, UI in
`web/page.html`; rebuild with `python3 web/build.py`). It loads a fictional
demo board on start; load your own CSV files on the *Data & settings* tab.

## Quick start

```bash
cd table-tennis-edge
pip install -e .            # Python 3.10+, numpy

ttedge serve                # dashboard with demo data at http://127.0.0.1:8765
ttedge price 1.62 2.35      # fair prices for every market from one match price
ttedge scan --demo          # value bets + arbs in the terminal
ttedge backtest --demo      # does the method beat the close?

# your own data
ttedge scan --results results.csv --odds odds.csv
ttedge serve --results results.csv --odds odds.csv
```

## How a line is judged

For every price on the board:

1. **Remove the margin** from each book's complete market (power method by
   default; Shin, proportional and additive are available).
2. **Leave-one-out consensus**: average the other books' margin-free
   probabilities in log-odds, with sharp books (Pinnacle, SBOBet, Betfair…)
   weighted 3×. A book is never compared with itself.
3. **Point-win probability**: invert the consensus match price to the
   probability that player A wins a point. That one number prices every
   derivative exactly (`ttedge price` shows the full table).
4. **Ratings blend** (optional): once both players have 15+ matches, the
   rating model's point probability is blended in (30% by default).
5. **Derivatives** blend the point model with the direct consensus for that
   line. Books whose derivatives disagree with their *own* match price are
   down-weighted, so one lazy book cannot drag the consensus.
6. **Edge** = expected profit per unit staked at fair value (pushes on whole
   lines handled). **Stake** = ¼ Kelly, capped at 2% of bankroll.

Flags explain each pick:

| flag | meaning |
|---|---|
| `beats_consensus` | beats the other books' margin-free price on its own |
| `inconsistent_with_own_ml` | this derivative disagrees with the same book's match price by 4+ points |
| `thin_consensus` | only one other book quotes the market |
| `ratings_disagree_with_market` | ratings and market differ by 15+ points on the match; check for news, injuries or integrity issues |
| `suspicious_edge` | edge above the 30% sanity cap; likely stale or a data error (never listed as a bet) |
| `sharp_book_vs_soft_consensus` | a sharp book judged only by softer books (not listed as a bet unless `bet_into_sharp`) |

## Data formats

`results.csv`, one row per finished match (trains the ratings):

```csv
match_id,start_time,tournament,player_a,player_b,score,best_of
m1,2026-11-12T09:30,Czech Liga Pro,Player One,Player Two,11-7 9-11 11-5 11-8,5
```

`score` can be game scores (best, since it feeds the point model), `3-1`, or
`3-1 (11-7, 9-11, 11-5, 11-8)`.

`odds.csv`, one row per price:

```csv
match_id,start_time,tournament,player_a,player_b,book,market,selection,line,price,snapshot
m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,ML,A,,1.72,current
m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAME_HCP,B,1.5,1.55,current
m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAMES_TOTAL,OVER,3.5,1.40,current
```

* `market`: `ML`, `GAME_HCP`, `GAMES_TOTAL`, `CORRECT_SCORE`, `POINTS_TOTAL`,
  `POINTS_HCP`, `G1_ML`, `G1_POINTS_TOTAL` (aliases such as `moneyline`,
  `set handicap` and `total games` also work).
* `selection`: `A`/`B` (or `home`/`away`, or the player's name), `OVER`/`UNDER`,
  or a score such as `3-1` (player A's games first).
* `line`: from the selection's point of view (`B +1.5`).
* `price`: decimal, American (`+150`, `-120`) or fractional (`5/2`).
* `snapshot`: `current` (scanned), or `open`/`close` for history (the
  backtest bets at `open` and measures CLV at `close`).
* Books may list the players in either order; rows are aligned automatically.

## Getting real data

This repo ships no scraper and no API keys. Options from the research:
OddsPapi (table tennis `sportId 25`), SportsGameOdds and Sportradar. Write a
small script that pulls fixtures, odds and results and writes the two CSVs
above, then run `ttedge scan` or reload them in the dashboard. For a
meaningful backtest, collect several weeks of **open and close** snapshots
plus results with game scores.

## Reading the backtest

* **CLV first.** Average edge at the price you took, measured against the
  closing consensus. Consistently positive CLV is the best evidence that the
  method finds real mispricing. ROI needs thousands of bets to separate from
  luck (see its standard error).
* **Claimed edge vs CLV.** If the scanner claims 6% and captures 1%, the model
  is overconfident: raise `min_edge` or lower the ratings weight.
* **By market / by book** shows where the value is. In the demo, soft-book
  match-winner picks have *negative* CLV, while rule-of-thumb derivatives
  have positive CLV.
* **Prediction quality.** If ratings can't approach the closing market's log
  loss, keep the ratings weight low.

## Limitations

* Points are modelled as independent within a game, with optional
  game-to-game variation (`sigma`, fitted from your point scores). Serve
  order is not modelled.
* Ratings ignore fatigue (several matches per day), head-to-head style and
  injuries. The market often knows these things.
* A big edge is more often a stale line, a data error or informed money than
  a gift. Several high-volume leagues have had match-fixing investigations.

Bet only where it is legal for you and with money you can afford to lose.
Help: begambleaware.org · 1-800-GAMBLER (US).

## Development

```bash
pip install -e '.[dev]'
pytest                      # 68 tests, including a Monte Carlo check of the point model
```

Layout: `odds.py` (margins, EV, Kelly) · `markov.py` (pricing model) ·
`ratings.py` · `scanner.py` · `backtest.py` · `data.py` (CSV) ·
`synthetic.py` (demo data) · `app.py`, `server.py`, `static/index.html`
(dashboard) · `cli.py`.
