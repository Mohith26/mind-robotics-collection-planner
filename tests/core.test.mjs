import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  cholesky, cholSolve, fitModel, drawPosterior, pointParams, predictSlice, knobCurve, planExact, planGreedy,
  planEven, planByVolume, allocUse, allocGain, backtest, failRate, mulberry32, binomial, sigmoid, xOf,
} from '../core/planner.mjs';

const plant = JSON.parse(readFileSync(new URL('../data/plant.json', import.meta.url)));
const truth = JSON.parse(readFileSync(new URL('../data/truth.json', import.meta.url)));
const fit = fitModel(plant);
const draws = drawPosterior(fit, 400, 11);
const curves = plant.knobs.map((k) => knobCurve(k, draws, plant.nNow, plant.weights));

test('cholesky solves a symmetric positive definite system', () => {
  const A = [[4, 2, 0.6], [2, 5, 1], [0.6, 1, 3]];
  const x = cholSolve(cholesky(A), [1, 2, 3]);
  for (let i = 0; i < 3; i++) {
    const r = A[i].reduce((s, a, j) => s + a * x[j], 0);
    assert.ok(Math.abs(r - [1, 2, 3][i]) < 1e-12);
  }
});

test('fit recovers per-slice learning curves (alpha and beta) from the generated history', () => {
  const est = pointParams(fit);
  const tb = truth.slices.map((t) => t.beta), eb = est.map((e) => e.beta);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const mt = mean(tb), me = mean(eb);
  let num = 0, dt = 0, de = 0;
  for (let i = 0; i < tb.length; i++) { num += (tb[i] - mt) * (eb[i] - me); dt += (tb[i] - mt) ** 2; de += (eb[i] - me) ** 2; }
  const corr = num / Math.sqrt(dt * de);
  assert.ok(corr > 0.6, 'beta correlation ' + corr);
  // current failure rate at today's data is well recovered
  const err = truth.slices.map((t, s) => Math.abs(failRate(t, plant.nNow[s]) - failRate(est[s], plant.nNow[s])));
  assert.ok(mean(err) < 0.04, 'mean abs error of current fail rate ' + mean(err));
});

test('release effects are separated from data growth (architecture change at release 6)', () => {
  const g = fit.theta.slice(fit.idx.g, fit.idx.P); // releases 1..9 relative to release 0
  const jump = g[4] - g[3]; // release 5 vs 4 (0-indexed release 5 = the change)
  const trueJump = truth.g[5] - truth.g[4];
  assert.ok(Math.abs(jump - trueJump) < 0.2, `estimated jump ${jump} vs true ${trueJump}`);
});

test('cold-start slice (no captured cycles) gets a wider interval than a data-rich slice', () => {
  const cold = plant.nNow.findIndex((n) => n === 0);
  assert.ok(cold >= 0);
  const rich = plant.nNow.indexOf(Math.max(...plant.nNow));
  const pc = predictSlice(draws, cold, 2000), pr = predictSlice(draws, rich, plant.nNow[rich] + 2000);
  assert.ok(pc.hi - pc.lo > 2 * (pr.hi - pr.lo));
});

test('exact DP matches brute force on a small problem and respects both budgets', () => {
  const knobs = [
    { maxShifts: 4, annotHoursPerShift: 1.5 },
    { maxShifts: 3, annotHoursPerShift: 2 },
    { maxShifts: 5, annotHoursPerShift: 0.5 },
  ];
  // deliberately non-concave curves (S-shaped) so greedy can fail
  const cv = [
    [10, 9.9, 9.7, 6, 5.8],
    [8, 6, 5, 4.5],
    [5, 4.8, 4.6, 4.4, 4.2, 4.0],
  ];
  const budget = { rigShifts: 6, annotHours: 7 };
  const res = planExact(knobs, cv, budget);
  let best = -1;
  for (let a = 0; a <= 4; a++) for (let b = 0; b <= 3; b++) for (let c = 0; c <= 5; c++) {
    if (a + b + c > budget.rigShifts) continue;
    if (a * 1.5 + b * 2 + c * 0.5 > budget.annotHours) continue;
    best = Math.max(best, cv[0][0] - cv[0][a] + cv[1][0] - cv[1][b] + cv[2][0] - cv[2][c]);
  }
  assert.ok(Math.abs(res.gain - best) < 1e-9, `dp ${res.gain} brute ${best}`);
  const greedy = planGreedy(knobs, cv, budget);
  assert.ok(res.gain >= greedy.gain - 1e-9);
  const u = allocUse(knobs.map((k) => ({ ...k, cyclesPerShift: 1 })), res.alloc);
  assert.ok(u.rigShifts <= budget.rigShifts && u.annotHours <= budget.annotHours);
});

test('plant plan: exact >= greedy and >= every baseline; all plans feasible', () => {
  const b = plant.budget;
  const ex = planExact(plant.knobs, curves, b);
  const gr = planGreedy(plant.knobs, curves, b);
  const ev = planEven(plant.knobs, b), vo = planByVolume(plant.knobs, b, plant.weights);
  for (const a of [ex.alloc, gr.alloc, ev, vo]) {
    const u = allocUse(plant.knobs, a);
    assert.ok(u.rigShifts <= b.rigShifts && u.annotHours <= b.annotHours + 1e-9);
    a.forEach((x, k) => assert.ok(x >= 0 && x <= plant.knobs[k].maxShifts));
  }
  assert.ok(ex.gain >= gr.gain - 1e-9);
  assert.ok(ex.gain >= allocGain(curves, ev) && ex.gain >= allocGain(curves, vo));
  assert.ok(Math.abs(allocGain(curves, ex.alloc) - ex.gain) < 1e-6);
});

test('backtest: 80% predictive intervals cover roughly 80% of held-out releases and beat persistence', () => {
  const bt = backtest(plant, 7, 400, 7);
  assert.equal(bt.n, plant.slices.length * 2);
  assert.ok(bt.coverage > 0.7 && bt.coverage < 0.98, 'coverage ' + bt.coverage);
  assert.ok(bt.maeModelSame < bt.maeNaive);
});

test('the committed page bundle reproduces from the committed data', async () => {
  const src = readFileSync(new URL('../data/bundle.js', import.meta.url), 'utf8');
  const win = {};
  new Function('window', src)(win);
  const ex = planExact(plant.knobs, curves, plant.budget);
  assert.deepEqual(win.BUNDLE.plans.model.planner.alloc, ex.alloc);
  assert.ok(Math.abs(win.BUNDLE.plans.model.planner.gain - ex.gain) < 1e-9);
});

test('binomial sampler and helpers are sane', () => {
  const rng = mulberry32(3);
  let s = 0;
  for (let i = 0; i < 2000; i++) s += binomial(rng, 80, 0.25);
  assert.ok(Math.abs(s / 2000 / 80 - 0.25) < 0.01);
  assert.equal(sigmoid(0), 0.5);
  assert.equal(xOf(0), 0);
});
