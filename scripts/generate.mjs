// Seeded synthetic plant history: 8 stations x 3 vehicle programs, 10 weekly model releases.
// Writes data/plant.json (what the planner sees) and data/truth.json (ground truth, used only for checks).
import { writeFileSync, mkdirSync } from 'node:fs';
import { mulberry32, normal, binomial, sigmoid, xOf } from '../core/planner.mjs';

const SEED = 20261110;
const rng = mulberry32(SEED);

const programs = ['R1T', 'R1S', 'R2'];
const lines = [
  { id: 'R1', name: 'R1 line', programs: [{ p: 0, share: 0.45 }, { p: 1, share: 0.55 }], vehiclesPerShift: 128 },
  { id: 'R2', name: 'R2 line', programs: [{ p: 2, share: 1 }], vehiclesPerShift: 116 },
];
// vehicles per day (2 shifts) by program -> failure-cost weights
const vehiclesPerDay = [lines[0].vehiclesPerShift * 2 * 0.45, lines[0].vehiclesPerShift * 2 * 0.55, lines[1].vehiclesPerShift * 2];

// station: shop, task, capture yield (usable cycles per vehicle passing the rig), annotation hrs per 100 cycles, rigs installed
const stations = [
  { code: 'TR-104', name: 'Door trim clip seating', shop: 'Trim 1', yield: 0.92, annot100: 2.4, rigs: [2, 2] },
  { code: 'TR-211', name: 'IP harness routing', shop: 'Trim 2', yield: 0.78, annot100: 6.2, rigs: [2, 1] },
  { code: 'TR-236', name: 'Rear harness connector mating', shop: 'Trim 2', yield: 0.85, annot100: 4.8, rigs: [1, 1] },
  { code: 'TR-318', name: 'Floor pan grommet insertion', shop: 'Trim 3', yield: 0.9, annot100: 2.0, rigs: [1, 1] },
  { code: 'CH-142', name: 'Thermal loop hose clamping', shop: 'Chassis 1', yield: 0.83, annot100: 3.6, rigs: [1, 2] },
  { code: 'CH-207', name: 'Brake line clip routing', shop: 'Chassis 2', yield: 0.81, annot100: 4.1, rigs: [1, 1] },
  { code: 'FN-055', name: 'Liftgate weatherstrip install', shop: 'Final 1', yield: 0.88, annot100: 3.0, rigs: [1, 1] },
  { code: 'FN-120', name: 'Headliner push-pin fastening', shop: 'Final 1', yield: 0.86, annot100: 2.8, rigs: [1, 1] },
];

// ---- ground truth learning curves ----
// fail-logit = a_station + d_program + e_slice - (b + u_station + v_slice) * x(n) + g_release
const aSt = [0.9, 1.9, 1.5, 0.5, 1.2, 1.4, 0.8, 0.6];
const uSt = [0.15, -0.05, 0.05, 0.25, -0.15, 0.1, 0.0, 0.2];
const dPr = [0, 0.1, 0.55]; // R2 is new geometry: harder at zero data
const b = 0.62;
const slices = [];
const truth = [];
for (let st = 0; st < stations.length; st++) {
  for (let p = 0; p < programs.length; p++) {
    slices.push({ station: st, program: p, line: p < 2 ? 0 : 1, label: `${stations[st].code} ${programs[p]}` });
    truth.push({ a: aSt[st] + dPr[p] + 0.35 * normal(rng), beta: b + uSt[st] + 0.18 * normal(rng) });
  }
}
// release effects: steady small gains, a bigger architecture change at release 6
const nReleases = 10;
const g = [0, -0.05, -0.08, -0.12, -0.15, -0.42, -0.45, -0.49, -0.52, -0.55];

// ---- historical capture policy (what the field team actually did) ----
// rig-shifts per (station, line) per week; R2 line capture starts week 5, some stations later
const r2Start = [4, 5, 7, 4, 6, 9, 10, 5]; // release index when R2 capture began (10 = not yet)
const knobs = [];
for (let st = 0; st < stations.length; st++) {
  for (let li = 0; li < 2; li++) {
    const S = stations[st];
    knobs.push({
      station: st,
      line: li,
      label: `${S.code} · ${lines[li].id}`,
      maxShifts: S.rigs[li] * 2 * 10, // rigs x 2 shifts x 10 working days (two-week plan)
      cyclesPerShift: Math.round(lines[li].vehiclesPerShift * S.yield),
      annotHoursPerShift: Math.round(lines[li].vehiclesPerShift * S.yield * S.annot100 / 100 * 2) / 2, // whole half-hours
      slices: lines[li].programs.map(({ p, share }) => ({ slice: st * 3 + p, share })),
    });
  }
}

const n = new Float64Array(slices.length);
const obs = [];
const capture = []; // weekly capture log
const TRIALS = 80; // held-out bench evaluation episodes per slice per release
for (let t = 0; t < nReleases; t++) {
  // capture during week t (before release t is trained)
  for (const k of knobs) {
    let shifts;
    if (k.line === 0) shifts = Math.round(stations[k.station].rigs[0] * 2 * 5 * (0.45 + 0.35 * rng()));
    else shifts = t >= r2Start[k.station] ? Math.round(stations[k.station].rigs[1] * 2 * 5 * (0.3 + 0.4 * rng())) : 0;
    // legacy habit: harness stations over-captured on R1 (they were the first rigs installed)
    if (k.line === 0 && (k.station === 1 || k.station === 0)) shifts = Math.round(shifts * 1.3);
    const cyc = shifts * k.cyclesPerShift;
    for (const { slice, share } of k.slices) n[slice] += Math.round(cyc * share);
    capture.push({ week: t + 1, knob: knobs.indexOf(k), shifts, cycles: cyc });
  }
  // release t: evaluate every slice on held-out bench episodes (zero-shot where no data)
  for (let s = 0; s < slices.length; s++) {
    const p = sigmoid(truth[s].a - truth[s].beta * xOf(n[s]) + g[t]);
    obs.push({ slice: s, release: t, n: n[s], trials: TRIALS, fails: binomial(rng, TRIALS, p) });
  }
}

const weights = slices.map((sl) => vehiclesPerDay[sl.program]);
const budget = { rigShifts: 90, annotHours: 340 };
const plant = {
  meta: { seed: SEED, site: 'Normal, IL', generatedBy: 'scripts/generate.mjs', week: nReleases, note: 'Synthetic data' },
  programs, lines: lines.map(({ id, name, vehiclesPerShift }) => ({ id, name, vehiclesPerShift })), stations, slices, knobs,
  vehiclesPerDay, weights, nNow: Array.from(n), obs, capture, budget,
  nStations: stations.length, nPrograms: programs.length, nReleases,
};
const truthOut = { slices: truth.map((t) => ({ alpha: t.a + g[nReleases - 1], beta: t.beta })), g, seed: SEED };
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../data/plant.json', import.meta.url), JSON.stringify(plant));
writeFileSync(new URL('../data/truth.json', import.meta.url), JSON.stringify(truthOut));
console.log('slices', slices.length, 'obs', obs.length, 'cycles', Math.round(n.reduce((a, b) => a + b, 0)));
console.log('nNow', Array.from(n).map((x) => Math.round(x)).join(' '));
