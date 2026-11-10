// Collection planner core: hierarchical learning-curve model + exact capture-budget allocation.
// Pure ES module, no dependencies. Runs in Node (tests, scripts) and in the browser (live re-plan).

// ---------- RNG ----------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function normal(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export function binomial(rng, n, p) {
  let k = 0;
  for (let i = 0; i < n; i++) if (rng() < p) k++;
  return k;
}

export const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

// Data-size feature: x = log(1 + n / n0). n0 = cycles worth of "prior experience" from pretraining.
export const N0 = 200;
export const xOf = (n) => Math.log1p(n / N0);

// ---------- Linear algebra (small dense) ----------
export function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('matrix not positive definite at ' + i);
        L[i][i] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  }
  return L;
}
export function cholSolve(L, b) {
  const n = L.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}
export function cholInverse(L) {
  const n = L.length;
  const inv = [];
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n);
    e[j] = 1;
    inv.push(cholSolve(L, e));
  }
  return inv; // symmetric, column j = row j
}

// ---------- Model layout ----------
// fail-logit(s, t) = A[st] + D[pr] + e[s] - (B + u[st] + v[s]) * x(n_{s,t}) + g[t]
// D[0] = 0 (first program is the reference), g[0] = 0 (first release is the reference).
export function layout(nStations, nPrograms, nReleases, slices) {
  let o = 0;
  const idx = {};
  idx.A = o; o += nStations;
  idx.D = o; o += nPrograms - 1;
  idx.e = o; o += slices.length;
  idx.B = o; o += 1;
  idx.u = o; o += nStations;
  idx.v = o; o += slices.length;
  idx.g = o; o += nReleases - 1;
  idx.P = o;
  return idx;
}

// sparse row: list of [col, value] for the linear predictor
function rowFor(idx, slice, s, x, t) {
  const r = [[idx.A + slice.station, 1], [idx.e + s, 1], [idx.B, -x], [idx.u + slice.station, -x], [idx.v + s, -x]];
  if (slice.program > 0) r.push([idx.D + slice.program - 1, 1]);
  if (t > 0) r.push([idx.g + t - 1, 1]);
  return r;
}
function etaOf(row, th) {
  let z = 0;
  for (const [c, v] of row) z += th[c] * v;
  return z;
}

export function priorVector(idx, scales) {
  const prec = new Float64Array(idx.P);
  const mean = new Float64Array(idx.P);
  const set = (start, len, sd, m = 0) => {
    for (let i = 0; i < len; i++) { prec[start + i] = 1 / (sd * sd); mean[start + i] = m; }
  };
  set(idx.A, idx.D - idx.A, 2.0);
  set(idx.D, idx.e - idx.D, 1.0);
  set(idx.e, idx.B - idx.e, scales.e);
  set(idx.B, 1, 1.0, 0.5);
  set(idx.u, idx.v - idx.u, scales.u);
  set(idx.v, idx.g - idx.v, scales.v);
  set(idx.g, idx.P - idx.g, 1.0);
  return { prec, mean };
}

// obs: [{slice, release, n, trials, fails}]
// Returns MAP estimate, Laplace covariance Cholesky, and EM-estimated pooling scales.
export function fitModel(data, opts = {}) {
  const { slices, nStations, nPrograms, nReleases } = data;
  const obs = data.obs.filter((o) => (opts.maxRelease == null ? true : o.release <= opts.maxRelease));
  const R = opts.maxRelease == null ? nReleases : opts.maxRelease + 1;
  const idx = layout(nStations, nPrograms, R, slices);
  const rows = obs.map((o) => rowFor(idx, slices[o.slice], o.slice, xOf(o.n), o.release));
  let scales = { e: 0.5, u: 0.3, v: 0.3, ...(opts.scales || {}) };
  let th = new Float64Array(idx.P);
  th[idx.B] = 0.5;
  let L = null, cov = null, iters = 0;
  const emRounds = opts.em === false ? 1 : 25;
  for (let em = 0; em < emRounds; em++) {
    const pr = priorVector(idx, scales);
    for (let it = 0; it < 50; it++) {
      iters++;
      const g = new Float64Array(idx.P);
      const H = Array.from({ length: idx.P }, () => new Float64Array(idx.P));
      for (let i = 0; i < obs.length; i++) {
        const o = obs[i], row = rows[i];
        const p = sigmoid(etaOf(row, th));
        const resid = o.fails - o.trials * p;
        const w = o.trials * p * (1 - p);
        for (const [c, v] of row) {
          g[c] += resid * v;
          for (const [c2, v2] of row) H[c][c2] += w * v * v2;
        }
      }
      for (let j = 0; j < idx.P; j++) {
        g[j] -= pr.prec[j] * (th[j] - pr.mean[j]);
        H[j][j] += pr.prec[j];
      }
      L = cholesky(H);
      const step = cholSolve(L, g);
      let mx = 0;
      for (let j = 0; j < idx.P; j++) { th[j] += step[j]; mx = Math.max(mx, Math.abs(step[j])); }
      if (mx < 1e-8) break;
    }
    if (opts.em === false) break;
    // EM update of the pooling scales from the Laplace posterior (second moments)
    cov = cholInverse(L);
    const upd = (start, len, floor) => {
      let s = 0;
      for (let i = 0; i < len; i++) s += th[start + i] ** 2 + cov[start + i][start + i];
      return Math.max(floor, Math.sqrt(s / len));
    };
    const next = { e: upd(idx.e, slices.length, 0.05), u: upd(idx.u, nStations, 0.05), v: upd(idx.v, slices.length, 0.05) };
    const delta = Math.abs(next.e - scales.e) + Math.abs(next.u - scales.u) + Math.abs(next.v - scales.v);
    scales = next;
    if (delta < 1e-4) break;
  }
  cov = cholInverse(L);
  const covL = cholesky(cov.map((r) => Array.from(r)));
  return { idx, theta: th, covL, cov, scales, iters, nReleases: R, slices, nObs: obs.length };
}

// Posterior draws (Laplace) of per-slice (alpha, beta) and the latest release effect.
export function drawPosterior(fit, nDraws, seed) {
  const rng = mulberry32(seed);
  const { idx, theta, covL, slices } = fit;
  const P = idx.P;
  const draws = [];
  for (let d = 0; d < nDraws; d++) {
    const z = new Float64Array(P);
    for (let i = 0; i < P; i++) z[i] = normal(rng);
    const th = new Float64Array(P);
    for (let i = 0; i < P; i++) {
      let s = theta[i];
      for (let k = 0; k <= i; k++) s += covL[i][k] * z[k];
      th[i] = s;
    }
    draws.push(sliceParams(fit, th));
  }
  return draws;
}
export function sliceParams(fit, th) {
  const { idx, slices, nReleases } = fit;
  const gLast = nReleases > 1 ? th[idx.g + nReleases - 2] : 0;
  return slices.map((sl, s) => ({
    alpha: th[idx.A + sl.station] + (sl.program > 0 ? th[idx.D + sl.program - 1] : 0) + th[idx.e + s] + gLast,
    beta: th[idx.B] + th[idx.u + sl.station] + th[idx.v + s],
  }));
}
export function pointParams(fit) { return sliceParams(fit, fit.theta); }

export const failRate = (par, n) => sigmoid(par.alpha - par.beta * xOf(n));

// Posterior predictive for one slice at n cycles: mean, quantiles of the true rate
export function predictSlice(draws, s, n) {
  const ps = draws.map((d) => failRate(d[s], n)).sort((a, b) => a - b);
  const q = (f) => ps[Math.min(ps.length - 1, Math.floor(f * ps.length))];
  return { mean: ps.reduce((a, b) => a + b, 0) / ps.length, lo: q(0.1), hi: q(0.9) };
}

// ---------- Planner ----------
// knobs: [{station, line, maxShifts, cyclesPerShift, annotHoursPerShift, slices:[{slice, share}]}]
// weights[s] = vehicles/day through that slice (failure cost weight)
// Objective: expected failed cycles per day = sum_s w_s * E[p_s(n_s + added_s)] over posterior draws.
export function knobCurve(knob, draws, nNow, weights) {
  const out = new Float64Array(knob.maxShifts + 1);
  for (let x = 0; x <= knob.maxShifts; x++) {
    let tot = 0;
    for (const { slice, share } of knob.slices) {
      const n = nNow[slice] + x * knob.cyclesPerShift * share;
      let m = 0;
      for (const d of draws) m += failRate(d[slice], n);
      tot += weights[slice] * (m / draws.length);
    }
    out[x] = tot; // expected failed cycles/day contributed by this knob's slices
  }
  return out;
}

// Exact optimum by dynamic programming over (rig-shifts used, annotation units used).
// Separable but NOT assumed concave (fail curves are S-shaped in log data), so greedy is not exact.
export function planExact(knobs, curves, budget, annotUnit = 0.5) {
  const R = budget.rigShifts;
  const Hn = Math.floor(budget.annotHours / annotUnit);
  const cost = knobs.map((k) => Math.round(k.annotHoursPerShift / annotUnit)); // annotation hours are whole half-hours
  const NEG = -Infinity;
  const width = (R + 1) * (Hn + 1);
  let best = new Float64Array(width).fill(NEG);
  best[0] = 0;
  const choice = [];
  for (let k = 0; k < knobs.length; k++) {
    const nb = new Float64Array(width).fill(NEG);
    const ch = new Int16Array(width).fill(-1);
    const c = curves[k];
    for (let r = 0; r <= R; r++) {
      for (let h = 0; h <= Hn; h++) {
        const cur = best[r * (Hn + 1) + h];
        if (cur === NEG) continue;
        for (let x = 0; x <= knobs[k].maxShifts && r + x <= R; x++) {
          const hu = h + x * cost[k];
          if (hu > Hn) break;
          const val = cur + (c[0] - c[x]);
          const id = (r + x) * (Hn + 1) + hu;
          if (val > nb[id]) { nb[id] = val; ch[id] = x; }
        }
      }
    }
    choice.push({ ch, prev: best });
    best = nb;
  }
  // best final state
  let bi = 0;
  for (let i = 0; i < width; i++) if (best[i] > best[bi]) bi = i;
  // backtrack
  const alloc = new Array(knobs.length).fill(0);
  let r = Math.floor(bi / (Hn + 1)), h = bi % (Hn + 1);
  for (let k = knobs.length - 1; k >= 0; k--) {
    const x = choice[k].ch[r * (Hn + 1) + h];
    alloc[k] = x;
    r -= x;
    h -= x * cost[k];
  }
  return { alloc, gain: best[bi] };
}

// Greedy by marginal gain (the usual heuristic) for comparison.
export function planGreedy(knobs, curves, budget) {
  const alloc = new Array(knobs.length).fill(0);
  let r = 0, h = 0;
  for (;;) {
    let bk = -1, bg = 0;
    for (let k = 0; k < knobs.length; k++) {
      const x = alloc[k];
      if (x >= knobs[k].maxShifts) continue;
      if (h + knobs[k].annotHoursPerShift > budget.annotHours) continue;
      const g = curves[k][x] - curves[k][x + 1];
      if (g > bg) { bg = g; bk = k; }
    }
    if (bk < 0 || r >= budget.rigShifts) break;
    alloc[bk]++; r++; h += knobs[bk].annotHoursPerShift;
  }
  return { alloc, gain: allocGain(curves, alloc) };
}
export function allocGain(curves, alloc) {
  let g = 0;
  for (let k = 0; k < alloc.length; k++) g += curves[k][0] - curves[k][alloc[k]];
  return g;
}
export function allocUse(knobs, alloc) {
  let r = 0, h = 0, cyc = 0;
  alloc.forEach((x, k) => { r += x; h += x * knobs[k].annotHoursPerShift; cyc += x * knobs[k].cyclesPerShift; });
  return { rigShifts: r, annotHours: h, cycles: cyc };
}

// Baselines: proportional targets, rounded down, then topped up one rig-shift at a time
// (largest remaining target first) until neither budget allows another shift. Uses the full budget fairly.
function fitToBudget(knobs, raw, budget) {
  const ok = (a) => { const u = allocUse(knobs, a); return u.rigShifts <= budget.rigShifts && u.annotHours <= budget.annotHours + 1e-9; };
  let f = 1;
  let alloc = raw.map((x, k) => Math.min(knobs[k].maxShifts, Math.floor(x)));
  while (!ok(alloc) && f > 0) { f -= 0.01; alloc = raw.map((x, k) => Math.min(knobs[k].maxShifts, Math.floor(x * f))); }
  for (;;) {
    let bk = -1, br = -Infinity;
    for (let k = 0; k < knobs.length; k++) {
      if (alloc[k] >= knobs[k].maxShifts || raw[k] <= 0) continue;
      alloc[k]++;
      const feasible = ok(alloc);
      alloc[k]--;
      if (!feasible) continue;
      const rem = raw[k] - alloc[k]; // how far below its proportional target this knob is
      if (rem > br) { br = rem; bk = k; }
    }
    if (bk < 0) break;
    alloc[bk]++;
  }
  return alloc;
}
export function planEven(knobs, budget) {
  const per = budget.rigShifts / knobs.length;
  return fitToBudget(knobs, knobs.map(() => per), budget);
}
export function planByVolume(knobs, budget, weights) {
  const vol = knobs.map((k) => k.slices.reduce((a, s) => a + weights[s.slice] * s.share, 0));
  const tot = vol.reduce((a, b) => a + b, 0);
  return fitToBudget(knobs, vol.map((v) => (budget.rigShifts * v) / tot), budget);
}
// Worst-first: rig-shifts proportional to current observed failure count per knob.
export function planWorstFirst(knobs, budget, curves) {
  const now = curves.map((c) => c[0]);
  const tot = now.reduce((a, b) => a + b, 0);
  return fitToBudget(knobs, now.map((v) => (budget.rigShifts * v) / tot), budget);
}

// Evaluate an allocation under given (e.g. true) per-slice params: failed cycles/day
export function evalAlloc(knobs, alloc, params, nNow, weights) {
  const added = new Float64Array(nNow.length);
  knobs.forEach((k, i) => k.slices.forEach(({ slice, share }) => { added[slice] += alloc[i] * k.cyclesPerShift * share; }));
  let tot = 0;
  for (let s = 0; s < nNow.length; s++) tot += weights[s] * failRate(params[s], nNow[s] + added[s]);
  return { failsPerDay: tot, added: Array.from(added) };
}

// ---------- Backtest ----------
// Fit on releases <= cut, forecast the observed failure rate of later releases given their actual cycle counts.
export function backtest(data, cut, nDraws = 600, seed = 7) {
  const fit = fitModel(data, { maxRelease: cut });
  const draws = drawPosterior(fit, nDraws, seed);
  const rng = mulberry32(seed + 1);
  const rows = [];
  for (const o of data.obs) {
    if (o.release <= cut) continue;
    // posterior predictive of observed failures out of o.trials (includes eval noise)
    const sims = draws.map((d) => binomial(rng, o.trials, failRate(d[o.slice], o.n)) / o.trials).sort((a, b) => a - b);
    const lo = sims[Math.floor(0.1 * sims.length)], hi = sims[Math.floor(0.9 * sims.length)];
    const mean = draws.reduce((a, d) => a + failRate(d[o.slice], o.n), 0) / draws.length;
    const obsRate = o.fails / o.trials;
    // naive persistence: last observed rate of the same slice at or before cut
    const prev = data.obs.filter((q) => q.slice === o.slice && q.release <= cut).sort((a, b) => b.release - a.release)[0];
    rows.push({ slice: o.slice, release: o.release, obs: obsRate, mean, lo, hi, covered: obsRate >= lo && obsRate <= hi, naive: prev ? prev.fails / prev.trials : null });
  }
  const mae = rows.reduce((a, r) => a + Math.abs(r.obs - r.mean), 0) / rows.length;
  const withNaive = rows.filter((r) => r.naive != null);
  const maeNaive = withNaive.reduce((a, r) => a + Math.abs(r.obs - r.naive), 0) / withNaive.length;
  const maeModelSame = withNaive.reduce((a, r) => a + Math.abs(r.obs - r.mean), 0) / withNaive.length;
  const coverage = rows.filter((r) => r.covered).length / rows.length;
  return { cut, rows, mae, maeNaive, maeModelSame, coverage, n: rows.length };
}
