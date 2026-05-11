# expel-hallucinations

Three-state output contract for Claude. Eval that scores the wrapped model on calibration, not accuracy.

## Run

```
uv sync
export ANTHROPIC_API_KEY=...
uv run python eval.py            # full run, ~10 min
uv run python eval.py --limit 6  # smoke test, ~30 s
```

Results below get rewritten in place. `reliability.png` lands at the repo root.

## Thesis

Hallucination is a calibration failure. A model that asserts a true claim at 0.95 when its warranted confidence is 0.5 has reported a falsehood about its own state. Honesty is the match between stated and warranted confidence.

The eval targets ECE and Brier, not accuracy. A model that lifts accuracy 70% to 80% by getting more confident is less honest. A model that holds accuracy steady and matches stated confidence to actual reliability is more honest.

## Operational definition

A hallucinated claim is a span asserting `p` with stated confidence `c` such that:

1. The request concerned some state of the world `X`.
2. No source available to the model entails `p`.
3. `c` exceeds the calibration-warranted confidence by more than threshold τ.

Falsifiable per span. That's what `eval.py` does.

## Mechanism

Every span is one of:

- `[CITED:<source_id>]...[/CITED]`. Entailed by a labeled source. Confidence bounded by source reliability.
- `[INF:<confidence>]...[/INF]`. Model judgment. Confidence in [0, 1], calibrated: 0.9 means wrong about 10% of the time on similar claims.
- `[ABSTAIN]<reason>[/ABSTAIN]`. Model declines.

Anything else is a parse error. The wrapper retries up to N times with a correction prompt. On persistent failure it downgrades to a single Abstention flagged `contract_failure=True`. No silent best-effort parsing; that would conceal the dishonesty the contract exists to expose.

The contract enforces legibility, not correctness. It does not check that citations match source text (that's the scorer). It does not check calibration (that's ECE). It does not stop the model from being wrong (that's the eval).

## Measurement

`eval.py` runs raw and wrapped on every example with identical sampling parameters (model defaults; the latest models reject explicit `temperature`). Each non-abstention span is scored by `claude-opus-4-7` against the gold answer with a fixed two-line rubric.

Confidence anchors:

- Cited: 0.95. Caps at source reliability.
- Inference: the model's stated confidence.
- Raw: 1.0. Raw mode has no contract; treating it as fully confident is the most charitable comparison.

Metrics:

- Hallucination rate (raw vs. wrapped), bootstrap 95% CIs at N=1000.
- ECE over 10 confidence bins.
- Brier on inference spans.
- Selective accuracy at coverage {25, 50, 75, 100}%.
- Reliability diagram (`reliability.png`) and ASCII fallback.

The judge is itself Claude. Circularity is real. Fix: sample a 10-example holdout, score by hand, report judge accuracy on that holdout in Results. v1 does not do this yet; the rubric is constrained enough (paraphrase-tolerant exact-fact comparison) that judge agreement should be high.

## Results

<!-- RESULTS:START -->

_Run `python eval.py` to populate._

<!-- RESULTS:END -->

## Extensions

Three sketches at the level an engineer could begin from.

**Data curation.** Apply the wrapper to a stream of training prompts. Flag every case where the pre-wrapper response makes a claim the post-wrapper response downgrades or abstains on. Those are the cases where the model knew that it didn't know but expressed itself as if it did. Score with the same path as `eval.py`. Filter or rewrite the confirmed cases.

**RL reward.** Ratios matter; units don't:

```
cited-correct                         +1
inference, calibrated confidence      +0.5 * calibration_score
abstain on unanswerable               +0.5
abstain on answerable                 -0.2
fabrication (uncited, wrong, conf>τ)  -10
```

One fabrication wipes out twenty calibrated answers. `calibration_score` is `1 - |stated - empirical|` over a rolling buffer of recent same-domain inference verdicts. Abstention rewarded enough to be a reliable escape hatch, not so much that the agent abstains on everything.

**Human feedback UI.** Labeler sees prompt, response with spans color-coded by tag, and the source(s) the model had. Three buttons per non-abstention span: correct, incorrect, underconfident. The third covers symmetric dishonesty (model said 0.4 about something it should have said 0.9 on). Inter-labeler disagreement above threshold triggers a third labeler and category review.

## What would change my mind

- Wrapped ECE not lower than raw on a held-out v2 set: mechanism is wrong, not the eval. Do not tune the prompt against v2 to rescue.
- Wrapped accuracy collapses while ECE improves: contract is suppressing signal. Model is over-abstaining. Rebalance the reward asymmetry, don't claim the calibration win.
- Judge accuracy on the human-labeled holdout below 90%: numbers aren't trustworthy. Report with the caveat.

Boundaries:

- Cited-claim calibration is bounded by source reliability. If a source is wrong, a faithful citation is wrong; the model did its job.
- Dataset is small (51 in v1). Bootstrap CIs are wide. Effect sizes need to clear that width.
- Confidence anchors (CITED 0.95, raw 1.0) are choices, not measurements. Different anchors shift the headline numbers but should not flip the direction of wrapped vs. raw if the mechanism is real.

## Not in v1

Multi-model comparison. Separate dataset file. Web UI for human feedback. Actual RL training loop. Standalone README.
