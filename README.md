# Collection Planner: where the next rig-shifts of capture should go

**Live demo:** https://mohith26.github.io/mind-robotics-collection-planner/

Unofficial concept demo by Mohith Gajjela. Not affiliated with Mind Robotics. All data is synthetic.

## What it is

A page for a robotics data platform that answers the question every data team running a capture operation on a live line gets asked weekly: *we have N rig-shifts of field-tech time and M annotation hours for the next two weeks; which stations, on which line, should we record?*

Mind Robotics describes its data platform as owning "what data we have, what's missing, and where to focus collection" (Software Engineer, Data Platform posting). This demo is one concrete version of the last part: it reads the bench evaluations of every weekly model release, fits a learning curve for every station × vehicle-program slice, and returns the capture plan that removes the most expected operator assists per day under both budgets.

On the synthetic plant (8 stations, R1T/R1S on the R1 line, R2 on the R2 line, 10 weekly releases):

- The plan puts all 89 rig-shifts it uses on the R2 line, including a station whose R2 slice has no captured cycles yet.
- Replayed through the simulator's true learning curves (which the planner never sees), it removes **181 assists/day**, versus **106** for spreading the same budget evenly (**+71%**), 123 for allocating by vehicle volume, and 135 for "worst station first".
- Forecast check: refit on releases 1-8, forecast releases 9-10 from the cycles actually captured: **44 of 48** held-out evals fall inside the 80% predictive interval; mean absolute error 3.2 points vs 4.9 for "same as last release".

## How it works

1. **Data.** Each weekly release is evaluated on 80 held-out bench episodes per slice. Slices with no captured cycles are still evaluated zero-shot, which pins down their intercept.
2. **Model.** `logit p_fail = α_slice − β_slice · log(1 + n/200) + γ_release`. α and β per slice; γ per model release, so a new architecture's gains are not credited to data (the synthetic plant has one at release 6; the fit recovers −0.25 logit vs −0.27 true).
3. **Pooling.** α = station + program + slice terms; β = shared + station + slice terms. The spreads of the station and slice terms are estimated by EM on the Laplace posterior. A cold slice borrows its slope from its station and siblings.
4. **Fit.** MAP by Newton's method (76 parameters), Laplace covariance, 400 posterior draws. Runs in the browser in ~40 ms.
5. **Knobs.** Each station × line is a decision variable in rig-shifts (cap = installed rigs × 2 shifts × 10 days). One shift yields that line's vehicles × the station's capture yield, split across programs by the line mix, and costs annotation hours.
6. **Objective.** Posterior-expected assists per day = Σ vehicles/day × E[p_fail(n + added)]. Fail curves are S-shaped in data, so the problem is not concave; it is solved **exactly by dynamic programming over both budgets** (rig-shifts × half-hour annotation units). Greedy is shown for comparison (98.6% of optimum here).
7. **Interactive.** Change the budgets and press Re-plan; the DP re-solves in the page.

## Files

- `core/planner.mjs`: model, EM/Newton fit, Laplace sampling, knob curves, exact DP, greedy and baselines, backtest. No dependencies.
- `scripts/generate.mjs`: seeded synthetic plant history (`data/plant.json`) and its ground truth (`data/truth.json`, used only for checks).
- `scripts/run.mjs`: fits, plans, backtests and scores every plan against ground truth; writes `data/bundle.js`.
- `index.html`, `app.js`, `styles.css`: the page. It recomputes the fit and the plan from `data/plant.json` in the browser; only the ground-truth comparison comes from the bundle.

## Tests

```
node scripts/generate.mjs && node scripts/run.mjs
node --test tests/core.test.mjs
```

9 tests: Cholesky solve; recovery of per-slice curves (β correlation, current fail-rate error < 4 pts); recovery of the release-6 architecture jump; cold-start interval width; DP equals brute force on a non-concave toy and beats greedy; plant plan feasibility and dominance over baselines; backtest calibration; committed bundle reproduces bit-for-bit; sampler sanity.

## Honest limitations

- The plant, the curves and the bench evals are synthetic. The generator and the fitted model share the same functional form, so the recovery and calibration numbers are optimistic compared with real data.
- Data size is the only covariate per slice. Real capture value also depends on diversity (operators, parts, lighting, failure modes), which this does not model; a natural next step is to replace `n` with an effective-sample-size from embedding coverage.
- The next release is assumed to keep today's architecture (γ unchanged). A planned model change would shift every curve.
- Bench-eval failure rate is a proxy for production assists.
- The page hotlinks Mind's own web font from mindrobotics.com (served with `Access-Control-Allow-Origin: *`) to match the site; swap it if that is a concern.

## Novelty note

Mind Robotics has no public product, docs or blog describing collection planning. Learning-curve-guided data acquisition is published research: Slice Tuner (Tae & Whang, SIGMOD 2021) fits per-slice power laws and optimizes acquisition, and Factored Scaling Curves (2025) guides robot demonstration collection per environment factor; ReMix (CoRL 2024) and DataMIL (ICLR 2026) reweight or select existing robot data rather than plan new capture. What this demo adds for a live plant: partial pooling so cold slices are plannable, separating model-release gains from data gains, a line-mix-aware knob model with rig and annotation budgets solved exactly, and a held-out forecast check.
