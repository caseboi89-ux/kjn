# Research notes: where table tennis lines get mispriced

Compiled 12 Nov 2026 from web search results. Direct page fetches and all
bookmaker/odds sites were blocked from the build environment, so the claims
below come from search-result summaries, and a lot of them are vendor
content. Each section says how much weight the evidence can bear.

## 1. The market

* **Volume is huge and concentrated in a few made-for-betting leagues.** One
  odds vendor (OddsPapi) logged **647–675 finished matches a day** across
  Czech Liga Pro, TT Cup and TT Elite Series on 23–25 Oct 2026, with 427
  upcoming fixtures carrying odds in one snapshot. [OddsPapi blog]
* **Depth differs by league.** In that census every board with 100+ books
  was Czech Liga Pro or TT Cup, and **TT Elite Series never passed 50
  books**. Same source: on TT Elite Series and WTT Champions boards, SBOBet's
  margin was **9–12%**, near the board median. Treat these as rough
  vendor figures. [OddsPapi blog]
* **ITTF/WTT events** (WTT Champions, Star Contender, etc.) are fewer, more
  public and probably priced more carefully. The high-frequency leagues are
  where the volume, and the most room for error, sit.

**What that means for finding edges.** With hundreds of matches a day,
bookmakers price from automated models and feeds. Match-winner lines get the
most attention and money. Derivative markets (game handicap, total games,
correct score, points totals) are often generated from simpler rules, and
margins are wide enough that books disagree with each other a lot.

## 2. Where mispricing plausibly comes from

These are the hypotheses the app is built to test. None is proven on real
data in this repo yet.

| Source of mispricing | Why it might exist | How `ttedge` tests it |
|---|---|---|
| **Book-to-book dispersion** | Thin boards, 6–12% margins and many soft books copying feeds at different speeds | Each price is compared with the margin-free consensus of the *other* books (leave-one-out, sharp books weighted 3×) |
| **Derivatives inconsistent with the same book's match price** | Totals and handicaps hung by rule of thumb instead of a consistent model | The book's own match price is inverted to a point-win probability, which prices every derivative exactly; the `inconsistent_with_own_ml` flag fires when the book disagrees with itself by 4+ points |
| **Information in the point scores** | Small player pools play each other constantly; win/loss Elo wastes ~75 point observations per match | A Kalman-filter point rating learns from every point won; the backtest compares it with win/loss Elo and the closing market |
| **Favourite-longshot bias** | Common in betting markets; tennis evidence is mixed (see §3) | Calibration table: implied vs actual win rate by probability bucket, for proportional and power de-vigging |
| **Under-modelled game-to-game swings** | An i.i.d.-points model under-predicts 3-2s and long matches | `sigma` (per-game strength noise) is fitted from point scores and fed into derivative pricing |

## 3. How much edge is realistic?

* **Little independent evidence.** The only table-tennis-specific edge claim
  found was self-published: a Czech Liga Pro model reporting **+2.71% ROI
  out-of-sample over 4,204 bets**. Its own calibration gap over the market is
  a Brier score of **0.2403 vs 0.2423**, a difference that could be noise.
  Not audited. [CPR Table Tennis site]
* **Tennis is the closest academic evidence.** In professional tennis,
  **betting odds out-predict Elo-type ratings for men**; for women a blend of
  Elo variants did slightly better on accuracy. Official rankings were a poor
  predictor. The favourite-longshot evidence is mixed: one study found none
  but found books underpriced much lower-ranked players; another (~45,000
  matches) found a stronger bias among lower-ranked players and in
  high-profile events. [NTU Elo study] [UCD WP24/03] [Tennis Abstract]
* **Takeaway:** expect the **match-winner line at the sharpest books to be
  hard to beat**. Look first for stale or soft books against a sharp
  consensus, and for derivatives that disagree with their own match price.
  Judge every method by **closing-line value**, not short-run ROI.

The synthetic demo shows this pattern on purpose. Its bets on soft-book
match winners show negative CLV (−1.0%), while its bets on rule-of-thumb
derivative prices show positive CLV (+3.3% on total games). The scanner's
claimed edge (~5.5%) is far above the CLV it captures (~1.1%), so raise
`min_edge` and trust the backtest over the headline number.

## 4. Integrity risk is real; build it into decisions

* **July 2020:** after a Sports Wagering Integrity Monitoring Association
  alert, **New Jersey suspended wagering on Ukrainian table tennis**,
  including Setka Cup, WIN Cup and TT Cup. [ESPN] [CBS Sports] [Gambling Insider]
* The Sports Integrity Initiative noted **no evidence of fixing** at Moscow
  Liga Pro, Setka Cup or Win Cup, but said their structure could attract
  fixers and that **none is ITTF-authorised**. [Sports Integrity Initiative]
* **2025:** Table Tennis England **suspended four members** after a match-fixing
  and unusual-betting investigation supported by the UK Gambling Commission's
  intelligence unit, linked to fixing in Ukraine in 2020. [Gambling Commission] [TTE decision]

**Implications:** a price far from consensus can be informed money, not a
mistake. The scanner caps believable edges (`max_edge`, default 30%), flags
`suspicious_edge`, and marks cases where ratings and market disagree by more
than 15 points. Big unexplained line moves deserve suspicion, not
excitement.

## 5. Practical constraints

* **Account limits:** soft books limit or close winning accounts quickly;
  arbitrage and stale-line betting get noticed first.
* **Settlement rules differ** (retirements, walkovers, games not completed):
  check each book's rules before treating two prices as the same bet.
* **Speed:** with hundreds of matches a day and frequent re-pricing, a
  snapshot goes stale in minutes. Poll close to start time, when the most
  books are posted.

## 6. Data sources for real use (not tested here: network blocked)

| Provider | What the search results say |
|---|---|
| OddsPapi | Table tennis = `sportId 25`; Czech Liga Pro, TT Cup, TT Elite Series, WTT; many books per match |
| SportsGameOdds | Dedicated table tennis odds API with moneylines and results |
| Sportradar Table Tennis API | Scores and stats for all major competitions; odds coverage unclear |
| Statscore | Table tennis coverage listed |

Any of these can feed `ttedge` by writing the odds and results to the CSV
formats in the README (a small adapter script per provider).

## Sources

* [OddsPapi – Table Tennis Odds API: Time Your Poll to the 125-Book Window](https://oddspapi.io/blog/table-tennis-odds-api/)
* [CPR Table Tennis – Systematic Edge in Undermodeled Markets](https://andydiaz122.github.io/cpr-table-tennis-website/)
* [How well do Elo-based ratings predict professional tennis matches? (NTU)](https://irep.ntu.ac.uk/id/eprint/42038)
* [Hegarty & Whelan, UCD Working Paper WP24/03](https://www.ucd.ie/economics/t4media/WP24_03.pdf)
* [Tennis Abstract – Why do my forecasts sometimes differ from betting markets?](https://www.tennisabstract.com/blog/?p=7612)
* [Supervised learning for table tennis match prediction (arXiv 2303.16776)](https://arxiv.org/pdf/2303.16776v1.pdf)
* [Ingram – A point-based Bayesian hierarchical model for tennis](https://martiningram.github.io/papers/bayes_point_based.pdf)
* [ESPN – New Jersey suspends betting on Ukrainian table tennis](https://africa.espn.com/chalk/story/_/id/29436278/new-jersey-suspends-betting-ukrainian-table-tennis-match-fixing-alert)
* [CBS Sports – NJ suspends Ukrainian table tennis betting](https://www.cbssports.com/general/news/new-jersey-suspends-ukrainian-table-tennis-betting-over-match-fixing-concerns)
* [Gambling Insider – NJ halts betting on Ukrainian table tennis](https://www.gamblinginsider.com/news/9522/nj-halts-betting-on-ukrainian-table-tennis-after-match-fixing-alert)
* [Sports Integrity Initiative](https://www.sportsintegrityinitiative.com/?p=21767)
* [ESPN – Gambling on table tennis is blowing up, but are the matches legit?](https://www.espn.com/chalk/story/_/id/29206521)
* [UK Gambling Commission – Four table tennis players suspended for match-fixing](https://www.gamblingcommission.gov.uk/news/article/four-table-tennis-players-suspended-for-match-fixing)
* [Table Tennis England – Appeals Panel decision, Oct 2025 (PDF)](https://www.tabletennisengland.co.uk/content/uploads/2025/12/Savill-Ors-v.-TTE-Appeals-Panel-Decision-dated-16-October-2025585297.1.pdf)
* [SportsGameOdds – Table Tennis odds API](https://sportsgameodds.com/sports/table-tennis-odds-api)
* [Sportradar – Table Tennis API](https://developer.sportradar.com/table-tennis/reference)
* [Statscore – table tennis coverage](https://www.statscore.com/coverage/tipsterapi/table-tennis/)
