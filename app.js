import {
  fitModel, drawPosterior, knobCurve, planExact, planGreedy, planEven, allocGain, allocUse, backtest, failRate, predictSlice, xOf, sigmoid,
} from './core/planner.mjs';

const P = window.PLANT, B = window.BUNDLE;
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (x, d = 0) => fmt(100 * x, d) + '%';
const PROG_COLOR = ['#061a1e', '#299093', '#ef6156'];
const NS = 'http://www.w3.org/2000/svg';
const tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip);
const showTip = (e, html) => { tip.innerHTML = html; tip.style.opacity = 1; tip.style.left = e.clientX + 12 + 'px'; tip.style.top = e.clientY + 12 + 'px'; };
const hideTip = () => { tip.style.opacity = 0; };

// ---------- model (recomputed in the browser from the committed plant data) ----------
const t0 = performance.now();
const fit = fitModel(P);
const draws = drawPosterior(fit, B.draws, B.seed);
const curves = P.knobs.map((k) => knobCurve(k, draws, P.nNow, P.weights));
const fitMs = performance.now() - t0;
const gLast = fit.theta[fit.idx.g + fit.nReleases - 2];
const gOf = (t) => (t === 0 ? 0 : fit.theta[fit.idx.g + t - 1]);

let budget = { ...P.budget };
let plan;

function addedFor(alloc) {
  const add = new Float64Array(P.slices.length);
  P.knobs.forEach((k, i) => k.slices.forEach(({ slice, share }) => { add[slice] += alloc[i] * k.cyclesPerShift * share; }));
  return add;
}
// per-draw total assists/day, for intervals on the totals
function totalDraws(add) {
  return draws.map((d) => P.slices.reduce((a, _, s) => a + P.weights[s] * failRate(d[s], P.nNow[s] + add[s]), 0)).sort((a, b) => a - b);
}
const q = (arr, f) => arr[Math.min(arr.length - 1, Math.floor(f * arr.length))];

function computePlan() {
  const t = performance.now();
  const ex = planExact(P.knobs, curves, budget);
  const ms = performance.now() - t;
  const gr = planGreedy(P.knobs, curves, budget);
  const now = curves.reduce((a, c) => a + c[0], 0);
  const add = addedFor(ex.alloc);
  const nowD = totalDraws(new Float64Array(P.slices.length));
  const aftD = totalDraws(add);
  // even-spread baseline under the model, same budget, for the hero comparison
  const evenAlloc = planEven(P.knobs, budget);
  const evenGain = allocGain(curves, evenAlloc);
  return { alloc: ex.alloc, gain: ex.gain, greedyGain: gr.gain, now, after: now - ex.gain, add, nowD, aftD, use: allocUse(P.knobs, ex.alloc), ms, evenGain };
}

// ---------- hero ----------
function renderHero() {
  const u = plan.use;
  const r2 = plan.alloc.reduce((a, x, k) => a + (P.knobs[k].line === 1 ? x : 0), 0);
  const share = u.rigShifts ? r2 / u.rigShifts : 0;
  const lineTxt = share >= 0.8 ? 'the R2 line' : share <= 0.2 ? 'the R1 line' : 'both lines';
  document.querySelector('.hero h1').innerHTML =
    `<span class="tag tag-teal">Where</span> the next <span class="tag tag-white">${fmt(u.rigShifts)} rig-shifts</span><br>of capture should go: <span class="tag tag-coral">${lineTxt}</span>`;
  $('site').textContent = P.meta.site;
  $('weeks').textContent = `Release ${P.meta.week} · plan for weeks ${P.meta.week + 1}–${P.meta.week + 2}`;
  const picked = plan.alloc.map((x, k) => ({ x, k })).filter((o) => o.x > 0).sort((a, b) => b.x - a.x);
  const cold = picked.filter((o) => P.knobs[o.k].slices.some((s) => P.nNow[s.slice] === 0));
  const stationsN = new Set(picked.map((o) => P.knobs[o.k].station)).size;
  const moreThanEven = plan.evenGain > 0 ? plan.gain / plan.evenGain - 1 : 0;
  $('lead').innerHTML =
    `<b>Bench evals</b> from ${P.nReleases} model releases show where more operator video still moves the policy. ` +
    (r2 === u.rigShifts ? `The plan puts <b>all ${fmt(r2)} rig-shifts on the R2 line, across ${stationsN} stations</b>` : `The plan puts <b>${fmt(r2)} of ${fmt(u.rigShifts)} rig-shifts on the R2 line</b>, across ${stationsN} stations`) +
    (cold.length ? `, including ${P.knobs[cold[0].k].label.split(' · ')[0]} on R2, which has no captured cycles yet` : '') +
    `. Expected assists drop from <b>${fmt(plan.now)}</b> to <b>${fmt(plan.after)} a day</b> (80% interval ${fmt(q(plan.aftD, 0.1))}–${fmt(q(plan.aftD, 0.9))}), ` +
    `${pct(moreThanEven)} more than spreading the same budget evenly.`;
  $('kpis').innerHTML = [
    { k: 'Assists / day today', v: fmt(plan.now), d: `80% interval ${fmt(q(plan.nowD, 0.1))}–${fmt(q(plan.nowD, 0.9))}` },
    { k: 'After this plan', v: fmt(plan.after), d: `−${fmt(plan.gain)} a day, expected` },
    { k: 'Versus even spread', v: `+${pct(moreThanEven)}`, d: `${fmt(plan.gain)} vs ${fmt(plan.evenGain)} removed a day` },
    { k: 'Forecast check', v: `${BT.rows.filter((r) => r.covered).length}<small>/${BT.n}</small>`, d: 'held-out evals inside the 80% band', dark: true },
  ].map((o) => `<div class="kpi${o.dark ? ' dark' : ''}"><div class="k">${o.k}</div><div class="v">${o.v}</div><div class="d">${o.d}</div></div>`).join('');
}

// ---------- plan table + side ----------
function knobGain(k, x) { return curves[k][0] - curves[k][x]; }
function renderPlan() {
  const rows = P.stations.map((S, st) => {
    const k1 = st * 2, k2 = st * 2 + 1;
    const chip = (k) => {
      const x = plan.alloc[k], kn = P.knobs[k];
      const cold = kn.slices.some((s) => P.nNow[s.slice] === 0);
      const cls = x === 0 ? 'off' : cold ? 'cold' : 'on';
      const next = curves[k][Math.min(x + 1, kn.maxShifts)] === curves[k][x] ? 0 : curves[k][x] - curves[k][x + 1];
      return `<span class="chip ${cls}" title="${kn.label}: ${x} of ${kn.maxShifts} rig-shifts · ${kn.cyclesPerShift} cycles and ${kn.annotHoursPerShift} annotation h per shift · next shift worth ${fmt(next, 2)} assists/day">${x === 0 ? '—' : x}</span><span class="cap">/${kn.maxShifts}</span>`;
    };
    const cyc = plan.alloc[k1] * P.knobs[k1].cyclesPerShift + plan.alloc[k2] * P.knobs[k2].cyclesPerShift;
    const g = knobGain(k1, plan.alloc[k1]) + knobGain(k2, plan.alloc[k2]);
    const nextBest = Math.max(...[k1, k2].map((k) => (plan.alloc[k] < P.knobs[k].maxShifts ? curves[k][plan.alloc[k]] - curves[k][plan.alloc[k] + 1] : 0)));
    return { st, S, chip1: chip(k1), chip2: chip(k2), cyc, g, nextBest };
  });
  const maxG = Math.max(...rows.map((r) => r.g), 1e-9);
  $('plan-table').innerHTML = `<thead><tr><th>Station</th><th>Shop</th><th>R1 line</th><th>R2 line</th><th class="r">New cycles</th><th>Assists / day removed</th><th class="r">Next shift worth</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td><div class="st-name">${r.S.name}</div><div class="st-code">${r.S.code}</div></td><td>${r.S.shop}</td><td>${r.chip1}</td><td>${r.chip2}</td><td class="r">${r.cyc ? fmt(r.cyc) : '—'}</td>` +
      `<td><div style="display:flex;align-items:center;gap:10px"><div class="bar" style="width:120px"><i style="width:${(100 * r.g) / maxG}%"></i></div><span style="font-variant-numeric:tabular-nums">${r.g > 0.05 ? fmt(r.g, 1) : '—'}</span></div></td>` +
      `<td class="r">${fmt(r.nextBest, 2)}</td></tr>`).join('') + '</tbody>';
  const u = plan.use;
  $('plan-sub').textContent = `${fmt(u.rigShifts)} rig-shifts and ${fmt(u.annotHours, 1)} annotation hours buy ${fmt(u.cycles)} new cycles. Chips show rig-shifts per station and line out of what the installed rigs allow (2 shifts × 10 days); coral marks a slice with no captured cycles yet.`;
  $('v-rig').textContent = fmt(budget.rigShifts);
  $('v-ann').textContent = fmt(budget.annotHours);
  $('m-rig').textContent = `${fmt(u.rigShifts)} / ${fmt(budget.rigShifts)}`;
  $('m-ann').textContent = `${fmt(u.annotHours, 1)} / ${fmt(budget.annotHours)}`;
  $('m-cyc').textContent = fmt(u.cycles);
  $('b-rig').style.width = Math.min(100, (100 * u.rigShifts) / budget.rigShifts) + '%';
  $('b-ann').style.width = Math.min(100, (100 * u.annotHours) / budget.annotHours) + '%';
  const states = (budget.rigShifts + 1) * (Math.floor(budget.annotHours / 0.5) + 1);
  $('replan-note').textContent = `Exact optimum over ${fmt(states)} budget states in ${fmt(plan.ms)} ms. Greedy reaches ${pct(plan.greedyGain / plan.gain, 1)} of it.`;
  // why not R1
  const r1 = P.knobs.map((k, i) => ({ k, i })).filter((o) => o.k.line === 0);
  const bestR1 = r1.map((o) => ({ ...o, g: curves[o.i][plan.alloc[o.i]] - curves[o.i][Math.min(plan.alloc[o.i] + 1, o.k.maxShifts)] })).sort((a, b) => b.g - a.g)[0];
  const lastBought = plan.alloc.map((x, i) => (x > 0 ? curves[i][x - 1] - curves[i][x] : Infinity)).reduce((a, b) => Math.min(a, b), Infinity);
  const r1Cycles = r1.reduce((a, o) => a + o.k.slices.reduce((b, s) => b + P.nNow[s.slice], 0), 0) / 2;
  const r2Cycles = P.knobs.filter((k) => k.line === 1).reduce((a, k) => a + P.nNow[k.slices[0].slice], 0);
  $('why').innerHTML = `R1 stations average ${fmt(r1Cycles / P.stations.length)} captured cycles per program, R2 stations ${fmt(r2Cycles / P.stations.length)}. The best R1 shift (${bestR1.k.label.split(' · ')[0]}) is worth ${fmt(bestR1.g, 2)} assists a day; the last shift this plan buys is worth ${fmt(lastBought === Infinity ? 0 : lastBought, 2)}.`;
}

// ---------- learning curves ----------
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function renderCurves() {
  $('legend').innerHTML = P.programs.map((p, i) => `<span><i style="background:${PROG_COLOR[i]}"></i>${p}</span>`).join('');
  const box = $('curves'); box.innerHTML = '';
  const W = 300, H = 170, m = { l: 34, r: 8, t: 8, b: 24 };
  const nMax = Math.max(...P.nNow.map((n, s) => n + plan.add[s])) * 1.08;
  const xs = Array.from({ length: 41 }, (_, i) => (i / 40) * nMax);
  const yMax = 0.8;
  const X = (n) => m.l + (n / nMax) * (W - m.l - m.r), Y = (p) => m.t + (1 - Math.min(p, yMax) / yMax) * (H - m.t - m.b);
  P.stations.forEach((S, st) => {
    const d = document.createElement('div'); d.className = 'curve';
    d.innerHTML = `<h3>${S.name}</h3><div class="cc">${S.code} · ${S.shop}</div>`;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'spark' });
    for (let g = 0; g <= 4; g++) {
      const p = (g / 4) * yMax;
      el('line', { x1: m.l, x2: W - m.r, y1: Y(p), y2: Y(p), stroke: 'rgba(6,26,30,.1)' }, svg);
      el('text', { x: m.l - 6, y: Y(p) + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = Math.round(p * 100) + '%';
    }
    [0, 0.5, 1].forEach((f) => { el('text', { x: X(f * nMax), y: H - 6, 'text-anchor': f === 0 ? 'start' : f === 1 ? 'end' : 'middle', 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = f === 0 ? '0' : fmt((f * nMax) / 1000, 0) + 'k cycles'; });
    for (let p = 0; p < 3; p++) {
      const s = st * 3 + p;
      const pr = xs.map((n) => predictSlice(draws, s, n));
      el('path', { d: 'M' + pr.map((o, i) => `${X(xs[i])},${Y(o.hi)}`).join('L') + 'L' + pr.slice().reverse().map((o, i) => `${X(xs[xs.length - 1 - i])},${Y(o.lo)}`).join('L') + 'Z', fill: PROG_COLOR[p], 'fill-opacity': 0.1 }, svg);
      el('path', { d: 'M' + pr.map((o, i) => `${X(xs[i])},${Y(o.mean)}`).join('L'), fill: 'none', stroke: PROG_COLOR[p], 'stroke-width': 2 }, svg);
      // bench evals, adjusted to the latest release so they sit on today's curve
      P.obs.filter((o) => o.slice === s && o.n > 0).forEach((o) => {
        const r = (o.fails + 0.5) / (o.trials + 1);
        const adj = sigmoid(Math.log(r / (1 - r)) + gLast - gOf(o.release));
        el('circle', { cx: X(o.n), cy: Y(adj), r: 1.8, fill: PROG_COLOR[p], 'fill-opacity': 0.45 }, svg);
      });
      const now = predictSlice(draws, s, P.nNow[s]).mean;
      const c1 = el('circle', { cx: X(P.nNow[s]), cy: Y(now), r: 4, fill: PROG_COLOR[p] }, svg);
      c1.addEventListener('mousemove', (e) => showTip(e, `${P.slices[s].label}: ${fmt(P.nNow[s])} cycles · ${pct(now, 1)} fail`));
      c1.addEventListener('mouseleave', hideTip);
      if (plan.add[s] > 0) {
        const aft = predictSlice(draws, s, P.nNow[s] + plan.add[s]).mean;
        el('line', { x1: X(P.nNow[s]), y1: Y(now), x2: X(P.nNow[s] + plan.add[s]), y2: Y(aft), stroke: PROG_COLOR[p], 'stroke-dasharray': '3 3', 'stroke-width': 1.5 }, svg);
        const c2 = el('circle', { cx: X(P.nNow[s] + plan.add[s]), cy: Y(aft), r: 4.5, fill: '#fff', stroke: PROG_COLOR[p], 'stroke-width': 2 }, svg);
        c2.addEventListener('mousemove', (e) => showTip(e, `${P.slices[s].label} after plan: +${fmt(plan.add[s])} cycles · ${pct(aft, 1)} fail`));
        c2.addEventListener('mouseleave', hideTip);
      }
    }
    d.appendChild(svg); box.appendChild(d);
  });
}

// ---------- coverage ----------
function renderCoverage() {
  const vals = P.slices.map((_, s) => {
    const a = predictSlice(draws, s, P.nNow[s]).mean, b = predictSlice(draws, s, P.nNow[s] + 1000).mean;
    return P.weights[s] * (a - b);
  });
  const hot = vals.slice().sort((a, b) => b - a)[4];
  const nMax = Math.max(...P.nNow);
  $('cov').innerHTML = `<thead><tr><th>Station</th>${P.programs.map((p) => `<th>${p}</th>`).join('')}</tr></thead><tbody>` +
    P.stations.map((S, st) => `<tr><td><div class="st-name" style="font-size:13px">${S.name}</div></td>` + [0, 1, 2].map((p) => {
      const s = st * 3 + p, n = P.nNow[s];
      return `<td><div class="cell${vals[s] >= hot ? ' hot' : ''}${n === 0 ? ' zero' : ''}"><span class="n">${fmt(n)}</span><div class="vbar"><i style="width:${(100 * n) / nMax}%"></i></div><span class="val">−${fmt(vals[s], 1)} / day per +1k</span></div></td>`;
    }).join('') + '</tr>').join('') + '</tbody>';
}

// ---------- backtest ----------
const BT = backtest(P, 7, 600, 7);
function renderBacktest() {
  $('bt-sub').textContent = `Refit on releases 1–8 only, then forecast the bench evals of releases 9 and 10 from the cycles actually captured. Each line is a forecast's 80% interval (eval noise included); dots are what the evals measured.`;
  const W = 620, H = 300, m = { l: 40, r: 10, t: 10, b: 30 };
  const rows = BT.rows.slice().sort((a, b) => a.mean - b.mean);
  const X = (i) => m.l + ((i + 0.5) / rows.length) * (W - m.l - m.r), Y = (p) => m.t + (1 - p / 0.9) * (H - m.t - m.b);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' });
  for (let g = 0; g <= 3; g++) { const p = g * 0.3; el('line', { x1: m.l, x2: W - m.r, y1: Y(p), y2: Y(p), stroke: 'rgba(6,26,30,.1)' }, svg); el('text', { x: m.l - 6, y: Y(p) + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = Math.round(p * 100) + '%'; }
  el('text', { x: m.l, y: H - 8, 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = `${rows.length} held-out evals (24 slices × releases 9–10), sorted by forecast`;
  rows.forEach((r, i) => {
    el('line', { x1: X(i), x2: X(i), y1: Y(r.lo), y2: Y(r.hi), stroke: r.covered ? 'rgba(41,144,147,.55)' : '#ef6156', 'stroke-width': 4, 'stroke-linecap': 'round' }, svg);
    el('circle', { cx: X(i), cy: Y(r.mean), r: 2, fill: '#fff' }, svg);
    const c = el('circle', { cx: X(i), cy: Y(r.obs), r: 3.2, fill: r.covered ? '#061a1e' : '#ef6156' }, svg);
    c.addEventListener('mousemove', (e) => showTip(e, `${P.slices[r.slice].label} · release ${r.release + 1}: forecast ${pct(r.mean)} (${pct(r.lo)}–${pct(r.hi)}), measured ${pct(r.obs)}`));
    c.addEventListener('mouseleave', hideTip);
  });
  $('bt').innerHTML = ''; $('bt').appendChild(svg);
  renderRelease();
  $('bt-stats').innerHTML = `<div><b>${pct(BT.coverage)}</b><span>inside the 80% interval (${BT.rows.filter((r) => r.covered).length} of ${BT.n})</span></div>` +
    `<div><b>${fmt(100 * BT.maeModelSame, 1)} pts</b><span>mean abs error, this model</span></div>` +
    `<div><b>${fmt(100 * BT.maeNaive, 1)} pts</b><span>mean abs error, "same as last release"</span></div>`;
}

function renderRelease() {
  const R = fit.nReleases;
  const g = [0], sd = [0];
  for (let t = 1; t < R; t++) { const i = fit.idx.g + t - 1; g.push(fit.theta[i]); sd.push(Math.sqrt(fit.cov[i][i])); }
  let jt = 1; for (let t = 1; t < R; t++) if (g[t] - g[t - 1] < g[jt] - g[jt - 1]) jt = t;
  const W = 620, H = 150, m = { l: 40, r: 10, t: 14, b: 24 };
  const lo = Math.min(...g.map((v, i) => v - 1.28 * sd[i])) - 0.05, hi = 0.15;
  const X = (t) => m.l + ((t + 0.5) / R) * (W - m.l - m.r), Y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' });
  [0, -0.25, -0.5].forEach((v) => { if (v < lo) return; el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: 'rgba(6,26,30,.1)' }, svg); el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = v.toFixed(2); });
  for (let t = 0; t < R; t++) {
    el('rect', { x: X(t) - 14, width: 28, y: Math.min(Y(0), Y(g[t])), height: Math.max(1, Math.abs(Y(g[t]) - Y(0))), rx: 6, fill: t === jt ? '#ffbd00' : '#061a1e' }, svg);
    if (t > 0) el('line', { x1: X(t), x2: X(t), y1: Y(g[t] - 1.28 * sd[t]), y2: Y(g[t] + 1.28 * sd[t]), stroke: '#299093', 'stroke-width': 2 }, svg);
    el('text', { x: X(t), y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: 'rgba(6,26,30,.55)' }, svg).textContent = 'R' + (t + 1);
  }
  const box = document.getElementById('rel');
  box.innerHTML = `<div class="rel-head"><b>Model-release effect</b><span>Estimated jointly with the curves, so release gains are not credited to data. Largest step: release ${jt + 1} (${fmt(g[jt] - g[jt - 1], 2)} logit, 80% bars in teal).</span></div>`;
  box.appendChild(svg);
}

// ---------- comparison vs ground truth ----------
function renderCompare() {
  const T = B.plans.truth, M = B.plans.model;
  const names = [
    ['planner', 'This plan', 'exact optimum, posterior-expected'],
    ['greedy', 'Greedy', 'best next shift, repeated'],
    ['worst', 'Worst-first', 'shifts ∝ today\'s assists'],
    ['volume', 'By volume', 'shifts ∝ vehicles per day'],
    ['even', 'Even spread', 'same shifts per station and line'],
  ];
  const max = Math.max(...names.map(([k]) => T[k].gain));
  $('cmp-sub').textContent = `Each plan spends at most the default budget (${fmt(P.budget.rigShifts)} rig-shifts, ${fmt(P.budget.annotHours)} annotation hours). Bars are assists per day removed when the plan is replayed through the simulator's true learning curves, which the planner never sees.`;
  $('cmp').innerHTML = names.map(([k, n, s]) => `<div class="cmp-row"><div class="nm">${n}<small>${s}</small></div><div class="cmp-bar"><i class="${k === 'planner' ? 'best' : ''}" style="width:${(100 * T[k].gain) / max}%"></i></div><div class="r"><b>${fmt(T[k].gain)}</b> / day removed<br><span class="st-code">model forecast ${fmt(M[k].gain)} · ${fmt(M[k].use.rigShifts)} shifts · ${fmt(M[k].use.cycles)} cycles</span></div></div>`).join('');
}

function renderSteps() {
  const sc = fit.scales;
  $('steps').innerHTML = [
    `Every weekly release is evaluated on ${P.obs[0].trials} held-out bench episodes per station × program slice (${fmt(fit.nObs)} evals so far, cold slices evaluated zero-shot).`,
    `Failure rate is modeled as <code>logit p = α − β·log(1 + n/${200})+ γ_release</code>: α and β per slice, γ per model release, so gains from a new architecture are not credited to data.`,
    `α and β are partially pooled across stations and programs (EM-estimated spreads: slice ${fmt(sc.e, 2)}, station slope ${fmt(sc.u, 2)}, slice slope ${fmt(sc.v, 2)}). A slice with no data borrows its slope from its station and its siblings.`,
    `MAP fit by Newton's method on ${fit.idx.P} parameters, Laplace posterior, ${B.draws} draws. Fit, draws and curves run in your browser in ${fmt(fitMs)} ms.`,
    `Each station × line is a knob: one rig-shift yields that line's vehicles × capture yield, split across programs by the line's mix, and costs annotation hours.`,
    `The objective is posterior-expected assists per day, weighted by each program's vehicles per day. Curves are S-shaped in data, so the plan is solved exactly by dynamic programming over both budgets, not greedily.`,
    `The forecast check refits on releases 1–8 and scores releases 9–10; the comparison replays each plan through the simulator's ground truth.`,
    `Limits: synthetic plant, a 1-D data-size feature per slice, and the next release is assumed to keep today's architecture (γ unchanged).`,
  ].map((s) => `<li>${s}</li>`).join('');
}

function renderAll() { plan = computePlan(); renderHero(); renderPlan(); renderCurves(); renderCoverage(); renderBacktest(); renderCompare(); renderSteps(); }

$('in-rig').value = budget.rigShifts; $('in-ann').value = budget.annotHours;
$('in-rig').addEventListener('input', (e) => { $('v-rig').textContent = e.target.value; });
$('in-ann').addEventListener('input', (e) => { $('v-ann').textContent = e.target.value; });
$('replan').addEventListener('click', () => {
  budget = { rigShifts: +$('in-rig').value, annotHours: +$('in-ann').value };
  plan = computePlan(); renderHero(); renderPlan(); renderCurves();
});
renderAll();
