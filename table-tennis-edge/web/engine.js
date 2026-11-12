/* TT Edge engine: browser port of the Python ttedge package (odds, point model,
   ratings, scanner, backtest, synthetic demo). No dependencies. */
(function (root) {
"use strict";

// ====================================================================== odds
const logit = (p) => { p = Math.min(Math.max(p, 1e-9), 1 - 1e-9); return Math.log(p / (1 - p)); };
const sigmoid = (x) => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));

function bisect(f, lo, hi, iters = 56) {
  let flo = f(lo);
  for (let i = 0; i < iters; i++) {
    const mid = 0.5 * (lo + hi), fm = f(mid);
    if ((fm > 0) === (flo > 0)) { lo = mid; flo = fm; } else hi = mid;
  }
  return 0.5 * (lo + hi);
}

function americanToDecimal(a) {
  if (a > -100 && a < 100) throw new Error(`invalid American odds: ${a}`);
  return 1 + (a > 0 ? a / 100 : 100 / -a);
}

function parsePrice(value) {
  let price;
  if (typeof value === "number") price = value;
  else {
    const t = String(value ?? "").trim();
    if (!t) throw new Error("empty price");
    if (t.includes("/")) { const [n, d] = t.split("/"); price = 1 + parseFloat(n) / parseFloat(d); }
    else if (t[0] === "+" || t[0] === "-") price = americanToDecimal(parseFloat(t));
    else price = Number(t);
  }
  if (!isFinite(price) || price <= 1) throw new Error(`price must be decimal odds above 1.0, got "${value}"`);
  return price;
}

const DEVIG_METHODS = ["power", "shin", "multiplicative", "additive"];

function devig(prices, method = "power") {
  const pi = prices.map((p) => 1 / p), total = pi.reduce((a, b) => a + b, 0), n = pi.length;
  const norm = (arr) => { const s = arr.reduce((a, b) => a + b, 0); return arr.map((x) => x / s); };
  if (method === "multiplicative") return pi.map((x) => x / total);
  if (method === "additive") return norm(pi.map((x) => Math.max(x - (total - 1) / n, 1e-9)));
  if (method === "power") {
    if (Math.abs(total - 1) < 1e-12) return pi.slice();
    const k = bisect((k) => pi.reduce((a, x) => a + x ** k, 0) - 1, 1e-3, 50);
    return norm(pi.map((x) => x ** k));
  }
  if (method === "shin") {
    if (total <= 1) return pi.map((x) => x / total);
    const probs = (z) => pi.map((x) => (Math.sqrt(z * z + 4 * (1 - z) * x * x / total) - z) / (2 * (1 - z)));
    const z = bisect((z) => probs(z).reduce((a, b) => a + b, 0) - 1, 0, 0.999);
    return norm(probs(z));
  }
  throw new Error(`unknown margin method ${method}`);
}

function applyMargin(probs, margin) {
  if (margin <= 0) return probs.map((p) => 1 / p);
  const e = bisect((e) => probs.reduce((a, p) => a + p ** e, 0) - (1 + margin), 1e-3, 1);
  return probs.map((p) => 1 / p ** e);
}
const expectedValue = (price, pWin, pPush = 0) => pWin * (price - 1) - Math.max(0, 1 - pWin - pPush);
const fairPrice = (pWin, pPush = 0) => pWin <= 0 ? Infinity : 1 + Math.max(0, 1 - pWin - pPush) / pWin;
function kellyFraction(price, pWin, pPush = 0) {
  const b = price - 1, pLose = Math.max(0, 1 - pWin - pPush);
  if (b <= 0 || pWin + pLose <= 0) return 0;
  return Math.max(0, (b * pWin - pLose) / (b * (pWin + pLose)));
}

// =================================================================== markets
const ML = "ML", GAME_HCP = "GAME_HCP", GAMES_TOTAL = "GAMES_TOTAL", CORRECT_SCORE = "CORRECT_SCORE",
  POINTS_TOTAL = "POINTS_TOTAL", POINTS_HCP = "POINTS_HCP", G1_ML = "G1_ML", G1_POINTS_TOTAL = "G1_POINTS_TOTAL";
const MARKETS = [ML, GAME_HCP, GAMES_TOTAL, CORRECT_SCORE, POINTS_TOTAL, POINTS_HCP, G1_ML, G1_POINTS_TOTAL];
const SIDE = new Set([ML, G1_ML]), HCP = new Set([GAME_HCP, POINTS_HCP]),
  TOTAL = new Set([GAMES_TOTAL, POINTS_TOTAL, G1_POINTS_TOTAL]), POINT_MKTS = new Set([POINTS_TOTAL, POINTS_HCP, G1_POINTS_TOTAL]);
const LINE_MKTS = new Set([...HCP, ...TOTAL]);
const MARKET_LABELS = { ML: "Match winner", GAME_HCP: "Game handicap", GAMES_TOTAL: "Total games", CORRECT_SCORE: "Correct score",
  POINTS_TOTAL: "Total points", POINTS_HCP: "Point handicap", G1_ML: "Game 1 winner", G1_POINTS_TOTAL: "Game 1 points" };
const MARKET_ALIASES = {
  ml: ML, moneyline: ML, match_winner: ML, winner: ML, h2h: ML, match: ML, "12": ML,
  game_handicap: GAME_HCP, set_handicap: GAME_HCP, sets_handicap: GAME_HCP, games_handicap: GAME_HCP, handicap: GAME_HCP, game_hcp: GAME_HCP, spread: GAME_HCP,
  games_total: GAMES_TOTAL, total_games: GAMES_TOTAL, total_sets: GAMES_TOTAL, sets_total: GAMES_TOTAL, totals: GAMES_TOTAL,
  correct_score: CORRECT_SCORE, exact_score: CORRECT_SCORE, cs: CORRECT_SCORE,
  points_total: POINTS_TOTAL, total_points: POINTS_TOTAL,
  points_hcp: POINTS_HCP, points_handicap: POINTS_HCP, point_handicap: POINTS_HCP,
  g1_ml: G1_ML, first_game_winner: G1_ML, game1_winner: G1_ML, set1_winner: G1_ML,
  g1_points_total: G1_POINTS_TOTAL, first_game_total_points: G1_POINTS_TOTAL, game1_total_points: G1_POINTS_TOTAL,
};
const SELECTION_ALIASES = { a: "A", home: "A", "1": "A", p1: "A", player_a: "A", player1: "A",
  b: "B", away: "B", "2": "B", p2: "B", player_b: "B", player2: "B", over: "OVER", o: "OVER", under: "UNDER", u: "UNDER" };

function normalizeMarket(v) {
  const key = String(v).trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (MARKETS.includes(key.toUpperCase())) return key.toUpperCase();
  if (key in MARKET_ALIASES) return MARKET_ALIASES[key];
  throw new Error(`unknown market "${v}"`);
}
function normalizeSelection(v, a = "", b = "") {
  const t = String(v).trim(), k = t.toLowerCase();
  if (k in SELECTION_ALIASES) return SELECTION_ALIASES[k];
  if (a && k === a.trim().toLowerCase()) return "A";
  if (b && k === b.trim().toLowerCase()) return "B";
  const m = t.match(/^(\d)\s*[-:]\s*(\d)$/);
  if (m) return `${m[1]}-${m[2]}`;
  throw new Error(`unknown selection "${v}"`);
}
function swapSelection(market, sel) {
  if (sel === "A") return "B";
  if (sel === "B") return "A";
  if (market === CORRECT_SCORE) { const [a, b] = sel.split("-"); return `${b}-${a}`; }
  return sel;
}
function gamesToWin(bo) { if (bo % 2 === 0 || bo < 1) throw new Error(`best_of must be odd, got ${bo}`); return Math.floor(bo / 2) + 1; }
function correctScoreSelections(bo) { const w = gamesToWin(bo), out = []; for (let k = 0; k < w; k++) out.push(`${w}-${k}`); for (let k = 0; k < w; k++) out.push(`${k}-${w}`); return out; }
const r3 = (x) => Math.round(x * 1000) / 1000;
function groupKey(market, sel, line) {
  if (HCP.has(market)) return `${market}|${r3(sel === "A" ? line : -line)}`;
  if (TOTAL.has(market)) return `${market}|${r3(line)}`;
  return `${market}|`;
}
function parseGroupKey(gk) { const [m, l] = gk.split("|"); return [m, l === "" ? null : parseFloat(l)]; }
function groupSelections(market, bo) {
  if (SIDE.has(market) || HCP.has(market)) return ["A", "B"];
  if (TOTAL.has(market)) return ["OVER", "UNDER"];
  if (market === CORRECT_SCORE) return correctScoreSelections(bo);
  throw new Error(market);
}
const selectionLine = (market, sel, gl) => HCP.has(market) ? (sel === "A" ? gl : -gl) : gl;
const fmtLine = (x) => (x > 0 ? "+" : "") + (+x.toFixed(3));
function describe(market, sel, line, a = "A", b = "B") {
  const who = { A: a, B: b };
  if (market === ML) return `${who[sel]} to win`;
  if (market === G1_ML) return `${who[sel]} to win game 1`;
  if (HCP.has(market)) return `${who[sel]} ${fmtLine(line)} ${market === GAME_HCP ? "games" : "points"}`;
  if (TOTAL.has(market)) {
    const unit = { GAMES_TOTAL: "games", POINTS_TOTAL: "points", G1_POINTS_TOTAL: "points in game 1" }[market];
    return `${sel === "OVER" ? "Over" : "Under"} ${+line.toFixed(3)} ${unit}`;
  }
  if (market === CORRECT_SCORE) return `${a} ${sel} ${b}`;
  return `${market} ${sel} ${line}`;
}

class MatchScore {
  constructor(ga, gb, games = []) { this.ga = ga; this.gb = gb; this.games = games; }
  get hasPoints() { return this.games.length > 0; }
  get pointsA() { return this.games.reduce((s, g) => s + g[0], 0); }
  get pointsB() { return this.games.reduce((s, g) => s + g[1], 0); }
  swapped() { return new MatchScore(this.gb, this.ga, this.games.map(([a, b]) => [b, a])); }
  toString() { return this.games.length ? this.games.map(([a, b]) => `${a}-${b}`).join(" ") : `${this.ga}-${this.gb}`; }
}
function validGame(a, b) { const hi = Math.max(a, b), lo = Math.min(a, b); if (hi < 11) return false; if (hi === 11) return lo <= 9; return hi - lo === 2; }
function parseScore(text) {
  let pairs = [...String(text || "").matchAll(/(\d+)\s*[-:]\s*(\d+)/g)].map((m) => [+m[1], +m[2]]);
  if (!pairs.length) throw new Error(`no score found in "${text}"`);
  let summary = null;
  if (Math.max(...pairs[0]) < 11) { summary = pairs[0]; pairs = pairs.slice(1); }
  for (const [a, b] of pairs) if (!validGame(a, b)) throw new Error(`invalid game score ${a}-${b} in "${text}"`);
  if (pairs.length) {
    const ga = pairs.filter(([a, b]) => a > b).length, gb = pairs.length - ga;
    if (summary && (summary[0] !== ga || summary[1] !== gb)) throw new Error(`summary disagrees with game scores in "${text}"`);
    return new MatchScore(ga, gb, pairs);
  }
  if (summary[0] === summary[1] || Math.max(...summary) > 4) throw new Error(`invalid match score "${text}"`);
  return new MatchScore(summary[0], summary[1]);
}
function cmp(v, line) { return Math.abs(v - line) < 1e-9 ? "push" : v > line ? "win" : "lose"; }
function settle(market, sel, line, s) {
  if ((POINT_MKTS.has(market) || market === G1_ML) && !s.hasPoints) return null;
  if (market === ML) return (s.ga > s.gb) === (sel === "A") ? "win" : "lose";
  if (market === G1_ML) { const [a, b] = s.games[0]; return (a > b) === (sel === "A") ? "win" : "lose"; }
  if (market === CORRECT_SCORE) return sel === `${s.ga}-${s.gb}` ? "win" : "lose";
  if (HCP.has(market)) {
    let d = market === GAME_HCP ? s.ga - s.gb : s.pointsA - s.pointsB;
    if (sel === "B") d = -d;
    return cmp(d + line, 0);
  }
  if (TOTAL.has(market)) {
    const t = market === GAMES_TOTAL ? s.ga + s.gb : market === POINTS_TOTAL ? s.pointsA + s.pointsB : s.games[0][0] + s.games[0][1];
    const r = cmp(t, line);
    if (r === "push" || sel === "OVER") return r;
    return r === "lose" ? "win" : "lose";
  }
  throw new Error(`cannot settle ${market}`);
}

// ==================================================================== markov
const DEUCE_CAP = 25, N_LOSER = 11 + DEUCE_CAP, MAX_MARGIN = 11;
const WINNER_PTS = [], GAME_TOTAL = [], GAME_MARGIN = [];
for (let l = 0; l < N_LOSER; l++) { const w = l <= 9 ? 11 : l + 2; WINNER_PTS.push(w); GAME_TOTAL.push(w + l); GAME_MARGIN.push(w - l); }
const MAX_GAME_TOTAL = Math.max(...GAME_TOTAL);
const GH_X = [-3.6684708465595826, -2.7832900997816514, -2.0259480158257555, -1.3265570844949328, -0.6568095668820998, 0,
  0.6568095668820998, 1.3265570844949328, 2.0259480158257555, 2.7832900997816514, 3.6684708465595826];
const GH_W = [1.4395603937142596e-06, 0.00034681946632334544, 0.011911395444911507, 0.11722787516770851, 0.42935975235612495,
  0.6547592869145916, 0.42935975235612495, 0.11722787516770851, 0.011911395444911507, 0.00034681946632334544, 1.4395603937142596e-06]
  .map((w) => w / Math.sqrt(Math.PI));
const COMB = []; { for (let k = 0; k < 10; k++) { let c = 1; for (let i = 1; i <= k; i++) c = c * (10 + i) / i; COMB.push(c); } }
const C2010 = 184756;

function winByLoserPoints(p) {
  const q = 1 - p, out = new Float64Array(N_LOSER), p11 = p ** 11;
  for (let k = 0; k < 10; k++) out[k] = COMB[k] * p11 * q ** k;
  const base = C2010 * p ** 10 * q ** 10 * p * p, r = 2 * p * q;
  let f = 1;
  for (let j = 0; j <= DEUCE_CAP; j++) { out[10 + j] = base * f; f *= r; }
  return out;
}
function pointProbs(p, sigma) {
  if (sigma <= 0) return [[p, 1]];
  const lp = logit(p);
  return GH_X.map((x, i) => [sigmoid(lp + Math.SQRT2 * sigma * x), GH_W[i]]);
}
function gameOutcomes(p, sigma = 0) {
  const winA = new Float64Array(N_LOSER), winB = new Float64Array(N_LOSER);
  for (const [pp, w] of pointProbs(p, sigma)) {
    const a = winByLoserPoints(pp), b = winByLoserPoints(1 - pp);
    for (let i = 0; i < N_LOSER; i++) { winA[i] += w * a[i]; winB[i] += w * b[i]; }
  }
  let t = 0; for (let i = 0; i < N_LOSER; i++) t += winA[i] + winB[i];
  for (let i = 0; i < N_LOSER; i++) { winA[i] /= t; winB[i] /= t; }
  return [winA, winB];
}
function gameWinProb(p, sigma = 0) {
  let total = 0;
  for (const [pp, w] of pointProbs(p, sigma)) {
    const q = 1 - pp, p11 = pp ** 11;
    let reg = 0; for (let k = 0; k < 10; k++) reg += COMB[k] * p11 * q ** k;
    const deuce = C2010 * pp ** 12 * q ** 10 / (pp * pp + q * q);
    total += w * (reg + deuce);
  }
  return total;
}
function matchWinProbFromGame(g, bo = 5) {
  const w = gamesToWin(bo); let s = 0, c = 1;
  for (let k = 0; k < w; k++) { if (k > 0) c = c * (w - 1 + k) / k; s += c * g ** w * (1 - g) ** k; }
  return s;
}
const matchWinProb = (p, bo = 5, sigma = 0) => matchWinProbFromGame(gameWinProb(p, sigma), bo);

const INV_TABLES = new Map();
function inverseTable(bo, sigma) {
  const key = `${bo}|${sigma}`;
  if (INV_TABLES.has(key)) return INV_TABLES.get(key);
  const n = 2001, lm = new Float64Array(n), lp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    lp[i] = -2.5 + 5 * i / (n - 1);
    let m = matchWinProb(sigmoid(lp[i]), bo, sigma);
    m = Math.min(Math.max(m, 1e-15), 1 - 1e-15);
    lm[i] = Math.log(m / (1 - m));
  }
  const t = [lm, lp]; INV_TABLES.set(key, t); return t;
}
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const n = xs.length; if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}
function pointProbFromMatchProb(m, bo = 5, sigma = 0) {
  m = Math.min(Math.max(m, 1e-6), 1 - 1e-6);
  const [lm, lpg] = inverseTable(bo, Math.round(sigma * 1e4) / 1e4);
  let p = sigmoid(interp(Math.log(m / (1 - m)), lm, lpg));
  for (let it = 0; it < 2; it++) {
    const eps = 1e-6, f0 = matchWinProb(p, bo, sigma) - m;
    const slope = (matchWinProb(p + eps, bo, sigma) - matchWinProb(p - eps, bo, sigma)) / (2 * eps);
    if (slope > 0) p = Math.min(Math.max(p - f0 / slope, 1e-4), 1 - 1e-4);
  }
  return p;
}

function conv(a, b) {
  const out = new Float64Array(a.length + b.length - 1);
  for (let i = 0; i < a.length; i++) { const ai = a[i]; if (ai === 0) continue; for (let j = 0; j < b.length; j++) out[i + j] += ai * b[j]; }
  return out;
}
function threshold(vals, probs, line) {
  let over = 0, push = 0, under = 0, tot = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i], p = probs[i]; tot += p;
    if (v > line + 1e-9) over += p; else if (Math.abs(v - line) <= 1e-9) push += p; else under += p;
  }
  return [over / tot, push / tot, under / tot];
}

class MatchModel {
  constructor(p, bo = 5, sigma = 0) {
    if (!(p > 0 && p < 1)) throw new Error("point probability must be between 0 and 1");
    this.pPoint = p; this.bestOf = bo; this.sigma = sigma;
    const w = gamesToWin(bo), [winA, winB] = gameOutcomes(p, sigma);
    let g = 0; for (let i = 0; i < N_LOSER; i++) g += winA[i];
    this.pGame = g;
    const totA = new Float64Array(MAX_GAME_TOTAL + 1), totB = new Float64Array(MAX_GAME_TOTAL + 1);
    const difA = new Float64Array(2 * MAX_MARGIN + 1), difB = new Float64Array(2 * MAX_MARGIN + 1);
    for (let l = 0; l < N_LOSER; l++) {
      totA[GAME_TOTAL[l]] += winA[l]; totB[GAME_TOTAL[l]] += winB[l];
      difA[MAX_MARGIN + GAME_MARGIN[l]] += winA[l]; difB[MAX_MARGIN - GAME_MARGIN[l]] += winB[l];
    }
    this.g1Total = totA.map((v, i) => v + totB[i]);
    let layer = new Map([["0,0", { a: 0, b: 0, pr: 1, tv: new Float64Array([1]), dv: new Float64Array([1]) }]]);
    const terminal = [];
    while (layer.size) {
      const next = new Map();
      for (const s of layer.values()) {
        for (const [na, nb, step, tvec, dvec] of [[s.a + 1, s.b, g, totA, difA], [s.a, s.b + 1, 1 - g, totB, difB]]) {
          // tvec/dvec already carry the game's win probability, so the
          // convolved vectors stay joint (they sum to the state's probability)
          const key = `${na},${nb}`, t = conv(s.tv, tvec), d = conv(s.dv, dvec);
          const cur = next.get(key);
          if (cur) { cur.pr += s.pr * step; addInto(cur.tv, t); addInto(cur.dv, d); }
          else next.set(key, { a: na, b: nb, pr: s.pr * step, tv: t, dv: d });
        }
      }
      layer = new Map();
      for (const [k, s] of next) { if (s.a === w || s.b === w) terminal.push(s); else layer.set(k, s); }
    }
    terminal.sort((x, y) => x.a - y.a || x.b - y.b);
    this.scores = {}; this.scoreList = [];
    let pm = 0, maxT = 0;
    for (const s of terminal) { this.scores[`${s.a}-${s.b}`] = s.pr; this.scoreList.push([s.a, s.b, s.pr]); if (s.a === w) pm += s.pr; maxT = Math.max(maxT, s.tv.length); }
    this.pMatch = pm;
    const span = MAX_MARGIN * bo, totals = new Float64Array(maxT), diffs = new Float64Array(2 * span + 1);
    for (const s of terminal) {
      for (let i = 0; i < s.tv.length; i++) totals[i] += s.tv[i];
      const off = MAX_MARGIN * (s.a + s.b), start = span - off;
      for (let i = 0; i < s.dv.length; i++) diffs[start + i] += s.dv[i];
    }
    this.totals = totals; this.totalVals = totals.map((_, i) => i);
    this.diffs = diffs; this.diffVals = diffs.map((_, i) => i - span);
  }
  dist(market) {
    if (market === GAME_HCP) return [this.scoreList.map(([a, b]) => a - b), this.scoreList.map((s) => s[2])];
    if (market === GAMES_TOTAL) return [this.scoreList.map(([a, b]) => a + b), this.scoreList.map((s) => s[2])];
    if (market === POINTS_TOTAL) return [this.totalVals, this.totals];
    if (market === POINTS_HCP) return [this.diffVals, this.diffs];
    if (market === G1_POINTS_TOTAL) return [this.g1Total.map((_, i) => i), this.g1Total];
    throw new Error(market);
  }
  expectedPoints() { let s = 0, t = 0; this.totals.forEach((p, i) => { s += p * i; t += p; }); return s / t; }
  medianLine(market) {
    const [vals, probs] = this.dist(market);
    const tot = probs.reduce((a, b) => a + b, 0);
    let c = 0, idx = 0;
    for (; idx < probs.length; idx++) { c += probs[idx] / tot; if (c >= 0.5) break; }
    const lo = vals[idx] - 0.5, hi = lo + 1;
    const pLo = threshold(vals, probs, lo)[0], pHi = threshold(vals, probs, hi)[0];
    return Math.abs(pLo - 0.5) <= Math.abs(pHi - 0.5) ? lo : hi;
  }
  outcome(market, sel, line = null) {
    if (market === ML || market === G1_ML) { const pa = market === ML ? this.pMatch : this.pGame; return sel === "A" ? [pa, 0, 1 - pa] : [1 - pa, 0, pa]; }
    if (market === CORRECT_SCORE) { const pr = this.scores[sel] || 0; return [pr, 0, 1 - pr]; }
    if (line == null) throw new Error(`${market} needs a line`);
    if (HCP.has(market)) {
      let [vals, probs] = this.dist(market);
      const sgn = sel === "B" ? -1 : 1;
      return threshold(vals.map((v) => sgn * v + line), probs, 0);
    }
    const [over, push, under] = threshold(...this.dist(market), line);
    return sel === "OVER" ? [over, push, under] : [under, push, over];
  }
  prob(market, sel, line = null) { const [w, , l] = this.outcome(market, sel, line); return w + l > 0 ? w / (w + l) : 0; }
  standardLines() {
    const w = gamesToWin(this.bestOf), out = [[ML, null], [G1_ML, null]];
    for (let k = 1; k < w; k++) out.push([GAME_HCP, -(k + 0.5)]);
    for (let k = 0; k < w - 1; k++) out.push([GAMES_TOTAL, w + k + 0.5]);
    out.push([POINTS_TOTAL, this.medianLine(POINTS_TOTAL)], [G1_POINTS_TOTAL, this.medianLine(G1_POINTS_TOTAL)], [CORRECT_SCORE, null]);
    return out;
  }
  fairTable() {
    const rows = [];
    for (const [market, line] of this.standardLines()) {
      const sels = HCP.has(market) ? [["A", line], ["B", -line]] : groupSelections(market, this.bestOf).map((s) => [s, line]);
      for (const [sel, sl] of sels) {
        const [win, push, lose] = this.outcome(market, sel, sl);
        rows.push({ market, selection: sel, line: sl, p_win: win, p_push: push, fair_price: win > 0 ? 1 + lose / win : null });
      }
    }
    return rows;
  }
}

function addInto(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }

const MODEL_CACHE = new Map();
function matchModel(p, bo = 5, sigma = 0) {
  const pr = Math.round(Math.min(Math.max(p, 1e-4), 1 - 1e-4) * 1e6) / 1e6, sr = Math.round(sigma * 1e4) / 1e4;
  const key = `${pr}|${bo}|${sr}`;
  let m = MODEL_CACHE.get(key);
  if (!m) { if (MODEL_CACHE.size > 20000) MODEL_CACHE.clear(); m = new MatchModel(pr, bo, sr); MODEL_CACHE.set(key, m); }
  return m;
}
function fitPointProb(targets, bo = 5, sigma = 0) {
  if (!targets.length) return null;
  const loss = (p) => { const m = matchModel(p, bo, sigma); let e = 0; for (const [mk, s, l, pr, w] of targets) e += w * (logit(m.prob(mk, s, l)) - logit(pr)) ** 2; return e; };
  let best = 0, bestL = Infinity; const grid = [];
  for (let i = 0; i <= 40; i++) grid.push(0.3 + 0.01 * i);
  grid.forEach((p, i) => { const l = loss(p); if (l < bestL) { bestL = l; best = i; } });
  let lo = grid[Math.max(best - 1, 0)], hi = grid[Math.min(best + 1, 40)];
  const phi = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 25; i++) { const m1 = hi - phi * (hi - lo), m2 = lo + phi * (hi - lo); if (loss(m1) < loss(m2)) hi = m2; else lo = m1; }
  return 0.5 * (lo + hi);
}

// =================================================================== ratings
const playerKey = (name) => String(name).split(/\s+/).filter(Boolean).join(" ").toLowerCase();
const days = (t) => t == null ? null : t / 86400000;
const RATING_DEFAULTS = { prior_sd: 0.12, drift_sd_per_day: 0.01, min_sd: 0.015, overdispersion: 1.3, max_step: 1.0 };

class PointRating {
  constructor(cfg = {}) { this.cfg = { ...RATING_DEFAULTS, ...cfg }; this.players = new Map(); }
  _new(name) { return { name: String(name).trim(), mu: 0, var: this.cfg.prior_sd ** 2, matches: 0, wins: 0, pw: 0, pp: 0, last: null }; }
  _get(name) { const k = playerKey(name); let s = this.players.get(k); if (!s) { s = this._new(name); this.players.set(k, s); } return s; }
  _var(s, now) { let v = s.var; if (now != null && s.last != null && now > s.last) v += this.cfg.drift_sd_per_day ** 2 * (now - s.last); return Math.min(v, this.cfg.prior_sd ** 2); }
  predict(a, b, bo = 5, when = null) {
    const now = days(when), sa = this.players.get(playerKey(a)) || this._new(a), sb = this.players.get(playerKey(b)) || this._new(b);
    const d = sa.mu - sb.mu, v = this._var(sa, now) + this._var(sb, now);
    const p = sigmoid(d / Math.sqrt(1 + Math.PI * v / 8));
    return { p_point: p, p_match: matchWinProb(p, bo), diff_mean: d, diff_sd: Math.sqrt(v), matches_a: sa.matches, matches_b: sb.matches, min_matches: Math.min(sa.matches, sb.matches) };
  }
  update(a, b, score, when = null) {
    const c = this.cfg, now = days(when), sa = this._get(a), sb = this._get(b);
    const va = this._var(sa, now), vb = this._var(sb, now), d = sa.mu - sb.mu, p = sigmoid(d);
    let innov, obs;
    if (score.hasPoints) {
      const n = score.pointsA + score.pointsB, slope = p * (1 - p);
      innov = (score.pointsA / n - p) / slope; obs = c.overdispersion * p * (1 - p) / (n * slope ** 2);
    } else {
      const n = score.ga + score.gb, g = gameWinProb(p), eps = 1e-4;
      const slope = (gameWinProb(sigmoid(d + eps)) - gameWinProb(sigmoid(d - eps))) / (2 * eps);
      innov = (score.ga / n - g) / slope; obs = c.overdispersion * g * (1 - g) / (n * slope ** 2);
    }
    innov = Math.max(-c.max_step, Math.min(c.max_step, innov));
    const S = va + vb + obs, ka = va / S, kb = vb / S;
    sa.mu += ka * innov; sb.mu -= kb * innov;
    sa.var = Math.max(va * (1 - ka), c.min_sd ** 2); sb.var = Math.max(vb * (1 - kb), c.min_sd ** 2);
    for (const [s, won, w, l] of [[sa, score.ga > score.gb, score.pointsA, score.pointsB], [sb, score.gb > score.ga, score.pointsB, score.pointsA]]) {
      s.matches++; s.wins += won ? 1 : 0; s.pw += w; s.pp += w + l; if (now != null) s.last = now;
    }
  }
  leaderboard(minMatches = 1) {
    const rows = [];
    for (const s of this.players.values()) if (s.matches >= minMatches) rows.push({
      player: s.name, rating: s.mu, sd: Math.sqrt(s.var), matches: s.matches, wins: s.wins,
      win_pct: s.matches ? s.wins / s.matches : null, point_pct: s.pp ? s.pw / s.pp : null, p_vs_avg: sigmoid(s.mu) });
    return rows.sort((x, y) => y.rating - x.rating);
  }
}
class EloRating {
  constructor(k = 32) { this.k = k; this.players = new Map(); }
  _get(n) { const k = playerKey(n); let s = this.players.get(k); if (!s) { s = { mu: 1500, matches: 0 }; this.players.set(k, s); } return s; }
  predict(a, b, bo = 5) {
    const sa = this.players.get(playerKey(a)) || { mu: 1500, matches: 0 }, sb = this.players.get(playerKey(b)) || { mu: 1500, matches: 0 };
    return { p_match: 1 / (1 + 10 ** ((sb.mu - sa.mu) / 400)) };
  }
  update(a, b, score) {
    const sa = this._get(a), sb = this._get(b), pm = 1 / (1 + 10 ** ((sb.mu - sa.mu) / 400)), won = score.ga > score.gb ? 1 : 0;
    sa.mu += this.k * (sa.matches < 15 ? 2 : 1) * (won - pm); sb.mu -= this.k * (sb.matches < 15 ? 2 : 1) * (won - pm);
    sa.matches++; sb.matches++;
  }
}

// ====================================================================== data
function parseCSV(text) {
  const rows = []; let row = [], f = "", q = false, i = 0;
  text = String(text || "").replace(/^﻿/, "");
  while (i < text.length) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += ch; }
    else if (ch === '"') q = true;
    else if (ch === ",") { row.push(f); f = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(f); rows.push(row); row = []; f = ""; }
    else f += ch;
    i++;
  }
  if (f !== "" || row.length) { row.push(f); rows.push(row); }
  const nonEmpty = rows.filter((r) => r.some((c) => c.trim() !== ""));
  if (!nonEmpty.length) return [];
  const head = nonEmpty[0].map((h) => h.trim().toLowerCase());
  return nonEmpty.slice(1).map((r) => { const o = {}; head.forEach((h, j) => { o[h] = (r[j] ?? "").trim(); }); return o; });
}
function parseTime(text) {
  const t = String(text || "").trim();
  if (!t) return null;
  let m;
  if ((m = t.match(/^(\d{1,2})[./](\d{1,2})[./](\d{4})(?:[ T](\d{1,2}):(\d{2}))?$/))) return Date.UTC(+m[3], +m[2] - 1, +m[1], +(m[4] || 0), +(m[5] || 0));
  let iso = t.replace(" ", "T");
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) iso += "T00:00";
  if (!/[zZ]|[+-]\d{2}:?\d{2}$/.test(iso)) iso += "Z";
  const ms = Date.parse(iso);
  if (isNaN(ms)) throw new Error(`unrecognised time "${t}"`);
  return ms;
}
const matchIdOf = (r) => r.match_id || `${(r.start_time || "").slice(0, 10)}|${r.player_a}|${r.player_b}`;
const bestOfOf = (r) => { if (!r.best_of) return 5; const b = parseInt(r.best_of, 10); gamesToWin(b); return b; };
function loadResults(text) {
  const out = [], errors = [];
  parseCSV(text).forEach((r, i) => {
    try {
      if (!r.player_a || !r.player_b) throw new Error("player_a and player_b are required");
      out.push({ match_id: matchIdOf(r), start_time: parseTime(r.start_time), tournament: r.tournament || "", player_a: r.player_a,
        player_b: r.player_b, score: parseScore(r.score), best_of: bestOfOf(r) });
    } catch (e) { errors.push(`results row ${i + 2}: ${e.message}`); }
  });
  out.sort((x, y) => (x.start_time == null) - (y.start_time == null) || (x.start_time || 0) - (y.start_time || 0));
  return [out, errors];
}
function loadOdds(text) {
  const out = [], errors = [];
  parseCSV(text).forEach((r, i) => {
    try {
      if (!r.player_a || !r.player_b) throw new Error("player_a and player_b are required");
      const market = normalizeMarket(r.market || ""), selection = normalizeSelection(r.selection || "", r.player_a, r.player_b);
      let line = r.line ? parseFloat(r.line) : null;
      if (LINE_MKTS.has(market) && (line == null || isNaN(line))) throw new Error(`${market} needs a line`);
      if (!LINE_MKTS.has(market)) line = null;
      const snapshot = (r.snapshot || "current").toLowerCase();
      if (!["open", "close", "current"].includes(snapshot)) throw new Error(`snapshot must be open, close or current, got "${snapshot}"`);
      out.push({ match_id: matchIdOf(r), start_time: parseTime(r.start_time), tournament: r.tournament || "", player_a: r.player_a,
        player_b: r.player_b, book: r.book || "unknown", market, selection, line, price: parsePrice(r.price), snapshot, best_of: bestOfOf(r) });
    } catch (e) { errors.push(`odds row ${i + 2}: ${e.message}`); }
  });
  return [out, errors];
}

// =================================================================== scanner
const SCAN_DEFAULTS = {
  devig_method: "power", min_edge: 0.03, max_edge: 0.30, min_books: 1,
  sharp_books: ["pinnacle", "sbobet", "betfair", "matchbook", "demosharp"], sharp_weight: 3.0,
  model_weight: 0.3, min_model_matches: 15, derivative_consensus_weight: 0.5, game_sigma: 0.0,
  kelly_multiplier: 0.25, max_stake: 0.02, min_odds: 1.2, max_odds: 6.0, inconsistency_threshold: 0.04, bet_into_sharp: false,
};
const blend = (p1, p2, w2) => p1 == null ? p2 : (p2 == null || w2 <= 0) ? p1 : sigmoid((1 - w2) * logit(p1) + w2 * logit(p2));

class ScanEvent {
  constructor(quotes, cfg) {
    const f = quotes[0];
    Object.assign(this, { cfg, match_id: f.match_id, start_time: f.start_time, tournament: f.tournament, player_a: f.player_a, player_b: f.player_b, best_of: f.best_of });
    const ka = playerKey(this.player_a), kb = playerKey(this.player_b);
    this.prices = new Map(); // book -> Map(gk -> {sel: price})
    for (const q of quotes) {
      const qa = playerKey(q.player_a), qb = playerKey(q.player_b);
      let sel = q.selection;
      if (qa === kb && qb === ka) sel = swapSelection(q.market, sel);
      else if (qa !== ka || qb !== kb) continue;
      const gk = groupKey(q.market, sel, q.line);
      if (!this.prices.has(q.book)) this.prices.set(q.book, new Map());
      const g = this.prices.get(q.book);
      if (!g.has(gk)) g.set(gk, {});
      g.get(gk)[sel] = q.price;
    }
    this.books = [...this.prices.keys()].sort();
    this.fair = new Map(); this.margin = new Map();
    for (const [book, groups] of this.prices) {
      const fb = new Map(), mb = new Map();
      for (const [gk, sels] of groups) {
        const [market] = parseGroupKey(gk), need = groupSelections(market, this.best_of);
        if (need.every((s) => s in sels)) {
          const prices = need.map((s) => sels[s]), probs = devig(prices, cfg.devig_method);
          fb.set(gk, Object.fromEntries(need.map((s, i) => [s, probs[i]])));
          mb.set(gk, prices.reduce((a, p) => a + 1 / p, 0) - 1);
        }
      }
      this.fair.set(book, fb); this.margin.set(book, mb);
    }
    this.pmCache = new Map();
    this.trust = new Map();
    for (const [book, groups] of this.fair) {
      const pOwn = this.ownPointProb(book), model = pOwn != null ? matchModel(pOwn, this.best_of, cfg.game_sigma) : null, tb = new Map();
      for (const [gk, probs] of groups) {
        const [market, gl] = parseGroupKey(gk);
        if (market === ML) { tb.set(gk, 1); continue; }
        if (!model) { tb.set(gk, 0.5); continue; }
        let dev = 0;
        for (const s in probs) dev = Math.max(dev, Math.abs(probs[s] - model.prob(market, s, selectionLine(market, s, gl))));
        tb.set(gk, 1 / (1 + (dev / cfg.inconsistency_threshold) ** 2));
      }
      this.trust.set(book, tb);
    }
  }
  isSharp(book) { return this.cfg.sharp_books.includes(book.toLowerCase()); }
  sharpQuoting(gk, exclude) { let n = 0; for (const [b, g] of this.fair) if (b !== exclude && g.has(gk) && this.isSharp(b)) n++; return n; }
  weight(b) { return this.isSharp(b) ? this.cfg.sharp_weight : 1; }
  consensus(gk, exclude = null) {
    const books = [...this.fair.keys()].filter((b) => b !== exclude && this.fair.get(b).has(gk));
    if (books.length < Math.max(1, this.cfg.min_books)) return [null, books.length];
    const [market] = parseGroupKey(gk), sels = groupSelections(market, this.best_of);
    const w = books.map((b) => this.weight(b) * (this.trust.get(b).get(gk) ?? 1)), tw = w.reduce((a, x) => a + x, 0);
    const raw = {}; let norm = 0;
    for (const s of sels) { let acc = 0; books.forEach((b, i) => { acc += w[i] * logit(this.fair.get(b).get(gk)[s]); }); raw[s] = sigmoid(acc / tw); norm += raw[s]; }
    for (const s of sels) raw[s] /= norm;
    return [raw, books.length];
  }
  marketPointProb(exclude = null) {
    const key = exclude ?? "\u0000";
    if (this.pmCache.has(key)) return this.pmCache.get(key);
    const sigma = this.cfg.game_sigma, bo = this.best_of;
    const [ml] = this.consensus(`${ML}|`, exclude);
    let p;
    if (ml) p = pointProbFromMatchProb(ml.A, bo, sigma);
    else {
      const groups = new Set();
      for (const [b, g] of this.fair) if (b !== exclude) for (const gk of g.keys()) groups.add(gk);
      const targets = [];
      for (const gk of groups) {
        const [cons] = this.consensus(gk, exclude); if (!cons) continue;
        const [market, gl] = parseGroupKey(gk);
        for (const s in cons) targets.push([market, s, selectionLine(market, s, gl), cons[s], 1]);
      }
      p = targets.length ? fitPointProb(targets, bo, sigma) : null;
    }
    this.pmCache.set(key, p);
    return p;
  }
  ownPointProb(book) { const ml = this.fair.get(book)?.get(`${ML}|`); return ml ? pointProbFromMatchProb(ml.A, this.best_of, this.cfg.game_sigma) : null; }
}

function scanEvent(quotes, cfg, ratings = null) {
  const ev = new ScanEvent(quotes, cfg), sigma = cfg.game_sigma, bo = ev.best_of;
  let pRating = null, ratingMp = null, ratingMatches = null;
  if (ratings) {
    const pr = ratings.predict(ev.player_a, ev.player_b, bo, ev.start_time);
    ratingMatches = [pr.matches_a, pr.matches_b];
    if (pr.min_matches >= cfg.min_model_matches) { pRating = pr.p_point; ratingMp = pr.p_match; }
  }
  const evals = [];
  for (const book of ev.books) {
    const pMkt = ev.marketPointProb(book), pFair = blend(pMkt, pRating, cfg.model_weight);
    const model = pFair != null ? matchModel(pFair, bo, sigma) : null;
    const pOwn = ev.ownPointProb(book), ownModel = pOwn != null ? matchModel(pOwn, bo, sigma) : null;
    const marketMp = pMkt != null ? matchModel(pMkt, bo, sigma).pMatch : null;
    for (const [gk, sels] of ev.prices.get(book)) {
      const [market, gl] = parseGroupKey(gk), [cons, nCons] = ev.consensus(gk, book);
      for (const sel in sels) {
        const price = sels[sel], line = selectionLine(market, sel, gl), flags = [];
        const consP = cons ? cons[sel] : null;
        let modelP = null, pWin, pPush;
        if (model) {
          const [w, push, l] = model.outcome(market, sel, line);
          modelP = w / (w + l);
          const f = (market === ML || consP == null) ? modelP
            : sigmoid((1 - cfg.derivative_consensus_weight) * logit(modelP) + cfg.derivative_consensus_weight * logit(consP));
          pWin = f * (1 - push); pPush = push;
          if (pMkt == null) flags.push("ratings_only");
        } else if (consP != null) { pWin = consP; pPush = 0; flags.push("consensus_only"); }
        else continue;
        const edge = expectedValue(price, pWin, pPush), kelly = kellyFraction(price, pWin, pPush);
        const stake = Math.min(cfg.max_stake, cfg.kelly_multiplier * kelly);
        const bookP = ev.fair.get(book)?.get(gk)?.[sel] ?? null;
        let ownP = null;
        if (ownModel && market !== ML) {
          ownP = ownModel.prob(market, sel, line);
          if (bookP != null && Math.abs(bookP - ownP) >= cfg.inconsistency_threshold) flags.push("inconsistent_with_own_ml");
        }
        if (consP != null && price > 1 / consP) flags.push("beats_consensus");
        if (nCons && nCons < 2) flags.push("thin_consensus");
        if (ratingMp != null && marketMp != null && Math.abs(ratingMp - marketMp) > 0.15) flags.push("ratings_disagree_with_market");
        if (edge > cfg.max_edge) flags.push("suspicious_edge");
        let blocked = false;
        if (ev.isSharp(book) && !ev.sharpQuoting(`${ML}|`, book)) { flags.push("sharp_book_vs_soft_consensus"); blocked = !cfg.bet_into_sharp; }
        if (edge >= cfg.min_edge && price >= cfg.min_odds && price <= cfg.max_odds && edge <= cfg.max_edge && !blocked) flags.unshift("opportunity");
        evals.push({
          match_id: ev.match_id, start_time: ev.start_time, tournament: ev.tournament, player_a: ev.player_a, player_b: ev.player_b,
          best_of: bo, book, market, selection: sel, line, price, description: describe(market, sel, line, ev.player_a, ev.player_b),
          p_win: pWin, p_push: pPush, fair_price: fairPrice(pWin, pPush), edge, kelly, stake,
          consensus_prob: consP, consensus_books: nCons, model_prob: modelP, rating_match_prob: ratingMp, market_match_prob: marketMp,
          book_prob: bookP, book_margin: ev.margin.get(book)?.get(gk) ?? null, own_ml_prob: ownP, flags,
        });
      }
    }
  }
  const arbs = [], groups = new Set();
  for (const g of ev.prices.values()) for (const gk of g.keys()) groups.add(gk);
  for (const gk of groups) {
    const [market, gl] = parseGroupKey(gk), sels = groupSelections(market, bo), best = {};
    for (const book of ev.books) { const g = ev.prices.get(book).get(gk) || {}; for (const s in g) if (!best[s] || g[s] > best[s][0]) best[s] = [g[s], book]; }
    if (!sels.every((s) => best[s])) continue;
    const inv = sels.reduce((a, s) => a + 1 / best[s][0], 0);
    if (inv < 1) arbs.push({ match_id: ev.match_id, start_time: ev.start_time, player_a: ev.player_a, player_b: ev.player_b, market, line: gl,
      legs: sels.map((s) => ({ selection: s, description: describe(market, s, selectionLine(market, s, gl), ev.player_a, ev.player_b),
        book: best[s][1], price: best[s][0], stake_share: (1 / best[s][0]) / inv })), margin: 1 / inv - 1 });
  }
  const pAll = blend(ev.marketPointProb(null), pRating, cfg.model_weight);
  let fair = [];
  if (pAll != null) fair = matchModel(pAll, bo, sigma).fairTable().map((r) => ({ ...r, description: describe(r.market, r.selection, r.line, ev.player_a, ev.player_b) }));
  const summary = { match_id: ev.match_id, start_time: ev.start_time, tournament: ev.tournament, player_a: ev.player_a, player_b: ev.player_b,
    best_of: bo, books: ev.books, p_point: pAll, p_point_market: ev.marketPointProb(null), p_point_rating: pRating,
    p_match: pAll != null ? matchModel(pAll, bo, sigma).pMatch : null, rating_matches: ratingMatches, fair };
  return [evals, arbs, summary];
}

function scan(quotes, cfg, ratings = null, snapshot = "current") {
  const by = new Map();
  for (const q of quotes) if (snapshot == null || q.snapshot === snapshot) { if (!by.has(q.match_id)) by.set(q.match_id, []); by.get(q.match_id).push(q); }
  const evaluations = [], arbitrages = [], events = [];
  for (const qs of by.values()) { const [e, a, s] = scanEvent(qs, cfg, ratings); evaluations.push(...e); arbitrages.push(...a); events.push(s); }
  events.sort((x, y) => (x.start_time || 0) - (y.start_time || 0) || (x.match_id < y.match_id ? -1 : 1));
  arbitrages.sort((x, y) => y.margin - x.margin);
  const opportunities = evaluations.filter((e) => e.flags[0] === "opportunity").sort((x, y) => y.edge - x.edge);
  return { evaluations, arbitrages, events, opportunities };
}

// ================================================================== backtest
const tick = () => new Promise((r) => setTimeout(r, 0));

async function calibrateSigma(results, ratingCfg = {}, { grid = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4], minMatches = 15, onProgress } = {}) {
  const ratings = new PointRating(ratingCfg), S = grid.length, O = GH_X.length;
  const ll = new Float64Array(S); let nm = 0, ng = 0;
  const gameLike = (pp, loser) => { const q = 1 - pp; return loser <= 9 ? COMB[loser] * pp ** 11 * q ** loser : C2010 * pp ** 12 * q ** 10 * (2 * pp * q) ** (loser - 10); };
  for (let idx = 0; idx < results.length; idx++) {
    const r = results[idx], pr = ratings.predict(r.player_a, r.player_b, r.best_of, r.start_time);
    if (r.score.hasPoints && pr.min_matches >= minMatches) {
      nm++; ng += r.score.games.length;
      for (let si = 0; si < S; si++) {
        let like = 0;
        for (let o = 0; o < O; o++) {
          const d = pr.diff_mean + pr.diff_sd * Math.SQRT2 * GH_X[o];
          let ml = 1;
          for (const [a, b] of r.score.games) {
            const loser = Math.min(Math.min(a, b), 35); let gl = 0;
            if (grid[si] === 0) { const pp = sigmoid(d); gl = gameLike(a > b ? pp : 1 - pp, loser); }
            else for (let i = 0; i < O; i++) { const pp = sigmoid(d + Math.SQRT2 * grid[si] * GH_X[i]); gl += GH_W[i] * gameLike(a > b ? pp : 1 - pp, loser); }
            ml *= gl;
          }
          like += GH_W[o] * ml;
        }
        ll[si] += Math.log(Math.max(like, 1e-300));
      }
    }
    ratings.update(r.player_a, r.player_b, r.score, r.start_time);
    if (idx % 300 === 299) { onProgress?.(idx / results.length); await tick(); }
  }
  if (!nm) return { sigma: 0, matches: 0, games: 0, grid: [] };
  let best = 0; for (let i = 1; i < S; i++) if (ll[i] > ll[best]) best = i;
  return { sigma: grid[best], matches: nm, games: ng, grid: grid.map((s, i) => ({ sigma: s, log_lik: ll[i] })) };
}

async function runBacktest(results, quotes, cfg, ratingCfg = {}, { betSnapshot = "open", closeSnapshot = "close", bankroll = 100, onProgress } = {}) {
  const ratings = new PointRating(ratingCfg), elo = new EloRating(), odds = new Map();
  for (const q of quotes) { if (!odds.has(q.match_id)) odds.set(q.match_id, {}); const o = odds.get(q.match_id); (o[q.snapshot] = o[q.snapshot] || []).push(q); }
  const pick = (s, ...order) => { for (const k of order) if (s[k]?.length) return s[k]; return []; };
  const preds = [], calib = [], bets = []; let withOdds = 0;
  for (let idx = 0; idx < results.length; idx++) {
    const r = results[idx], snaps = odds.get(r.match_id) || {};
    const pred = ratings.predict(r.player_a, r.player_b, r.best_of, r.start_time), ep = elo.predict(r.player_a, r.player_b);
    const aWon = r.score.ga > r.score.gb ? 1 : 0;
    const closeQ = pick(snaps, closeSnapshot, "current"), closeEv = closeQ.length ? new ScanEvent(closeQ, cfg) : null;
    if (closeEv) {
      withOdds++;
      const [ml] = closeEv.consensus(`${ML}|`), flipped = playerKey(closeEv.player_a) !== playerKey(r.player_a);
      if (ml) {
        const mClose = flipped ? ml.B : ml.A, mult = [];
        for (const g of closeEv.prices.values()) { const s = g.get(`${ML}|`); if (s?.A && s?.B) { const pa = devig([s.A, s.B], "multiplicative")[0]; mult.push(flipped ? 1 - pa : pa); } }
        const mp = mult.length ? mult.reduce((a, b) => a + b, 0) / mult.length : mClose;
        calib.push([mClose, mp, aWon], [1 - mClose, 1 - mp, 1 - aWon]);
        if (pred.min_matches >= cfg.min_model_matches) {
          const pm = matchModel(pred.p_point, r.best_of, cfg.game_sigma).pMatch;
          preds.push([pm, ep.p_match, mClose, sigmoid((1 - cfg.model_weight) * logit(mClose) + cfg.model_weight * logit(pm)), aWon]);
        }
      }
    }
    const betQ = pick(snaps, betSnapshot, closeSnapshot, "current");
    if (betQ.length) {
      const [evals] = scanEvent(betQ, cfg, ratings), best = new Map();
      for (const e of evals) if (e.flags[0] === "opportunity") { const k = groupKey(e.market, e.selection, e.line) + e.selection; if (!best.has(k) || e.edge > best.get(k).edge) best.set(k, e); }
      for (const e of best.values()) {
        const score = playerKey(e.player_a) !== playerKey(r.player_a) ? r.score.swapped() : r.score;
        const res = settle(e.market, e.selection, e.line, score);
        if (res == null) continue;
        const stake = e.stake * bankroll; if (stake <= 0) continue;
        const profit = res === "win" ? stake * (e.price - 1) : res === "push" ? 0 : -stake;
        bets.push({ match_id: e.match_id, start_time: e.start_time, book: e.book, market: e.market, description: e.description, price: e.price,
          edge: e.edge, stake, result: res, profit, clv: closeEv && closeQ !== betQ ? clvOf(closeEv, cfg, e) : null });
      }
    }
    ratings.update(r.player_a, r.player_b, r.score, r.start_time); elo.update(r.player_a, r.player_b, r.score);
    if (idx % 100 === 99) { onProgress?.(idx / results.length); await tick(); }
  }
  return { matches: results.length, matches_with_odds: withOdds, prediction: predMetrics(preds), calibration: calibTable(calib), betting: betMetrics(bets), bankroll };
}
function clvOf(closeEv, cfg, e) {
  const flipped = playerKey(closeEv.player_a) !== playerKey(e.player_a), sel = flipped ? swapSelection(e.market, e.selection) : e.selection;
  const gk = groupKey(e.market, sel, e.line), [cons] = closeEv.consensus(gk), pp = closeEv.marketPointProb(null);
  let pWin, pPush = 0;
  if (cons) { pWin = cons[sel]; if (pp != null && e.line != null) { pPush = matchModel(pp, closeEv.best_of, cfg.game_sigma).outcome(e.market, sel, e.line)[1]; pWin *= 1 - pPush; } }
  else { if (pp == null) return null; [pWin, pPush] = matchModel(pp, closeEv.best_of, cfg.game_sigma).outcome(e.market, sel, e.line); }
  return pWin * (e.price - 1) - (1 - pWin - pPush);
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
function predMetrics(preds) {
  if (!preds.length) return { n: 0 };
  const out = { n: preds.length };
  ["ratings", "elo", "market", "blend"].forEach((name, i) => {
    const ll = mean(preds.map((r) => { const p = Math.min(Math.max(r[i], 1e-6), 1 - 1e-6); return -(r[4] * Math.log(p) + (1 - r[4]) * Math.log(1 - p)); }));
    out[name] = { log_loss: ll, brier: mean(preds.map((r) => (r[i] - r[4]) ** 2)), accuracy: mean(preds.map((r) => ((r[i] > 0.5) === (r[4] === 1)) ? 1 : 0)) };
  });
  return out;
}
function calibTable(rows) {
  const out = [];
  for (let b = 0; b < 10; b++) {
    const lo = b / 10, hi = (b + 1) / 10, sel = rows.filter((r) => r[1] >= lo && (b === 9 ? r[1] <= hi : r[1] < hi));
    if (!sel.length) continue;
    const actual = mean(sel.map((r) => r[2]));
    out.push({ bucket: `${b * 10}%-${(b + 1) * 10}%`, n: sel.length, implied_multiplicative: mean(sel.map((r) => r[1])),
      implied_devigged: mean(sel.map((r) => r[0])), actual, se: Math.sqrt(Math.max(actual * (1 - actual), 1e-9) / sel.length) });
  }
  return out;
}
function betMetrics(bets) {
  if (!bets.length) return { n: 0 };
  const staked = bets.reduce((a, b) => a + b.stake, 0), profit = bets.reduce((a, b) => a + b.profit, 0);
  const ret = bets.map((b) => b.profit / b.stake), m = mean(ret);
  const se = bets.length > 1 ? Math.sqrt(ret.reduce((a, x) => a + (x - m) ** 2, 0) / (bets.length - 1) / bets.length) : NaN;
  let cum = 0, peak = 0, dd = 0; const curve = [], step = Math.max(1, Math.floor(bets.length / 300));
  bets.forEach((b, i) => { cum += b.profit; peak = Math.max(peak, cum); dd = Math.max(dd, peak - cum); if (i % step === 0 || i === bets.length - 1) curve.push({ i, t: b.start_time, profit: cum }); });
  const clvs = bets.map((b) => b.clv).filter((c) => c != null);
  const group = (key) => { const t = {}; for (const b of bets) { const g = t[b[key]] = t[b[key]] || { n: 0, staked: 0, profit: 0, clv: [] }; g.n++; g.staked += b.stake; g.profit += b.profit; if (b.clv != null) g.clv.push(b.clv); }
    for (const k in t) { const g = t[k]; g.roi = g.staked ? g.profit / g.staked : null; g.avg_clv = g.clv.length ? mean(g.clv) : null; delete g.clv; } return t; };
  return { n: bets.length, staked, profit, roi: profit / staked, roi_se: se, hit_rate: mean(bets.map((b) => b.result === "win" ? 1 : 0)),
    avg_price: mean(bets.map((b) => b.price)), avg_edge: mean(bets.map((b) => b.edge)), avg_clv: clvs.length ? mean(clvs) : null,
    pct_positive_clv: clvs.length ? mean(clvs.map((c) => c > 0 ? 1 : 0)) : null, max_drawdown: dd, curve, by_market: group("market"), by_book: group("book") };
}

// ================================================================= synthetic
function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let spare = null;
  const normal = (mu = 0, sd = 1) => { if (spare != null) { const v = spare; spare = null; return mu + sd * v; }
    let a, b, r2; do { a = 2 * u() - 1; b = 2 * u() - 1; r2 = a * a + b * b; } while (r2 >= 1 || r2 === 0);
    const f = Math.sqrt(-2 * Math.log(r2) / r2); spare = b * f; return mu + sd * a * f; };
  const sample = (n, k) => { const idx = [...Array(n).keys()]; for (let i = 0; i < k; i++) { const j = i + Math.floor(u() * (n - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return idx.slice(0, k); };
  return { u, normal, sample };
}
const TRUE_SIGMA = 0.10;
const DEMO_BOOKS = [
  { name: "DemoSharp", open: 0.010, close: 0.005, margin: 0.035, style: "model", markets: [ML, GAME_HCP, GAMES_TOTAL, POINTS_TOTAL] },
  { name: "DemoSoftA", open: 0.015, close: 0.012, margin: 0.065, style: "model_naive", markets: [ML, GAME_HCP, GAMES_TOTAL, CORRECT_SCORE] },
  { name: "DemoSoftB", open: 0.012, close: 0.010, margin: 0.07, style: "lazy", markets: [ML, GAME_HCP, GAMES_TOTAL, G1_ML] },
];
function simulateMatch(R, d, bo) {
  const w = gamesToWin(bo); let ga = 0, gb = 0; const games = [];
  while (ga < w && gb < w) {
    const [wa, wb] = gameOutcomes(sigmoid(d + R.normal(0, TRUE_SIGMA)));
    let u = R.u(), k = 0, aWins = true, acc = 0;
    for (k = 0; k < N_LOSER; k++) { acc += wa[k]; if (u < acc) break; }
    if (k === N_LOSER) { aWins = false; for (k = 0; k < N_LOSER - 1; k++) { acc += wb[k]; if (u < acc) break; } }
    const wp = WINNER_PTS[k];
    games.push(aWins ? [wp, k] : [k, wp]); if (aWins) ga++; else gb++;
  }
  return new MatchScore(ga, gb, games);
}
function bookGroups(book, dEst, bo) {
  // books quote to 2 decimals, so rounding their belief to 1e-4 changes nothing visible and lets models be shared
  const model = matchModel(Math.round(sigmoid(dEst) * 1e4) / 1e4, bo, book.style === "model" ? TRUE_SIGMA : 0), m = model.pMatch;
  const fav = m >= 0.5 ? "A" : "B", favM = Math.max(m, 1 - m), out = [];
  for (const market of book.markets) {
    if (market === ML) out.push([market, null, ["A", "B"], [m, 1 - m]]);
    else if (market === G1_ML) { const g = book.style === "lazy" ? 0.5 + (m - 0.5) * 0.6 : model.pGame; out.push([market, null, ["A", "B"], [g, 1 - g]]); }
    else if (market === GAME_HCP) {
      const al = fav === "A" ? -1.5 : 1.5;
      const pa = book.style === "lazy" ? (fav === "A" ? Math.max(0.05, favM - 0.2) : 1 - Math.max(0.05, favM - 0.2)) : model.prob(market, "A", al);
      out.push([market, al, ["A", "B"], [pa, 1 - pa]]);
    } else if (market === GAMES_TOTAL) { const po = book.style === "lazy" ? 0.74 - 0.4 * Math.abs(m - 0.5) : model.prob(market, "OVER", 3.5); out.push([market, 3.5, ["OVER", "UNDER"], [po, 1 - po]]); }
    else if (market === POINTS_TOTAL) { const line = model.medianLine(market), po = model.prob(market, "OVER", line); out.push([market, line, ["OVER", "UNDER"], [po, 1 - po]]); }
    else if (market === CORRECT_SCORE) { const sels = correctScoreSelections(bo); out.push([market, null, sels, sels.map((s) => Math.max(model.prob(market, s), 1e-3))]); }
  }
  return out;
}
function generateDemo({ seed = 7, nPlayers = 40, days: nDays = 24, groupsPerDay = 6, groupSize = 5, upcomingGroups = 6 } = {}) {
  const R = rng(seed), now = new Date(), start = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 8) - nDays * 86400000;
  const names = Array.from({ length: nPlayers }, (_, i) => `Demo Player ${String(i + 1).padStart(2, "0")}`);
  let theta = names.map(() => R.normal(0, 0.10));
  const results = [], quotes = []; let no = 0;
  for (let day = 0; day <= nDays; day++) {
    const upcoming = day === nDays;
    theta = theta.map((t) => t + R.normal(0, 0.006));
    for (let grp = 0; grp < (upcoming ? upcomingGroups : groupsPerDay); grp++) {
      const players = R.sample(nPlayers, groupSize), league = grp % 2 === 0 ? "Demo League A" : "Demo League B";
      const t0 = start + day * 86400000 + grp * 3600000; let k = 0;
      for (let x = 0; x < players.length; x++) for (let y = x + 1; y < players.length; y++) {
        const i = players[x], j = players[y], when = t0 + 20 * 60000 * k++;
        const base = { match_id: `demo-${String(++no).padStart(5, "0")}`, start_time: when, tournament: league, player_a: names[i], player_b: names[j], best_of: 5 };
        const dTrue = theta[i] - theta[j], form = R.normal(0, 0.02);
        for (const book of DEMO_BOOKS) {
          const emit = (dEst, snapshot) => {
            for (const [market, al, sels, probs] of bookGroups(book, dEst, 5)) {
              const tot = probs.reduce((a, b) => a + b, 0), prices = applyMargin(probs.map((p) => p / tot), book.margin * (market === CORRECT_SCORE ? 1.6 : 1));
              sels.forEach((s, n) => quotes.push({ ...base, book: book.name, market, selection: s, line: selectionLine(market, s, al), price: Math.max(1.01, Math.floor(prices[n] * 100) / 100), snapshot }));
            }
          };
          const noiseOpen = R.normal(0, book.open);
          if (upcoming) emit(dTrue + noiseOpen, "current");
          else { emit(dTrue + noiseOpen, "open"); emit(dTrue + 0.7 * form + R.normal(0, book.close), "close"); }
        }
        if (!upcoming) results.push({ ...base, score: simulateMatch(R, dTrue + form, 5) });
      }
    }
  }
  return [results, quotes];
}

const TT = {
  logit, sigmoid, parsePrice, devig, applyMargin, expectedValue, fairPrice, kellyFraction, DEVIG_METHODS,
  ML, GAME_HCP, GAMES_TOTAL, CORRECT_SCORE, POINTS_TOTAL, POINTS_HCP, G1_ML, G1_POINTS_TOTAL, MARKETS, MARKET_LABELS,
  normalizeMarket, normalizeSelection, swapSelection, groupKey, describe, parseScore, settle, MatchScore,
  gameWinProb, matchWinProb, pointProbFromMatchProb, MatchModel, matchModel, fitPointProb,
  PointRating, EloRating, playerKey, parseCSV, parseTime, loadResults, loadOdds,
  SCAN_DEFAULTS, ScanEvent, scanEvent, scan, calibrateSigma, runBacktest, generateDemo,
};
if (typeof module !== "undefined" && module.exports) module.exports = TT; else root.TT = TT;
})(typeof window !== "undefined" ? window : globalThis);
