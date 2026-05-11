# expel-hallucinations

A small, self-contained portfolio artifact for the **Research Scientist / Engineer, Honesty** role on Anthropic's Finetuning Alignment team. Three load-bearing files: a wrapper that enforces a three-state output contract around the Anthropic SDK, an eval that measures whether the wrapper actually reduces dishonesty, and this document — the argument that ties them together.

## How to run this

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
python eval.py            # full run on all examples (~10 min)
python eval.py --limit 6  # smoke test (~30 s)
```

The eval rewrites the **Results** section below in place between the fenced markers, and writes `reliability.png` to the repo root.

## Thesis: hallucination is a calibration failure

It is not enough for a model to be right. A model that asserts a true `p` with confidence `c` when its warranted confidence is `c' < c` has reported a falsehood — about its own state — even though `p` happens to hold. Honesty is a relation between *stated* and *warranted* confidence; hallucination is the case where the former exceeds the latter beyond some threshold.

This reframe is load-bearing for everything that follows. It is what makes the eval target Expected Calibration Error and Brier score, not just accuracy. A pipeline that lifts accuracy from 70% to 80% by getting more confident is *less* honest; a pipeline that holds accuracy steady but matches stated confidence to actual reliability is *more* honest. Both are interventions worth running. Only the second is what this team is for.

## Operational definition

A **hallucinated claim** is an output span asserting `p` with stated confidence `c` such that:

1. The user's request concerned some state of the world `X`.
2. No tool return or grounding source available to the model entails `p`.
3. `c` exceeds the calibration-warranted confidence by more than threshold `τ`.

The definition is falsifiable per-span: given a labeled span and the sources it had access to, you can score it. This is what `eval.py` does.

## Mechanism: three-state output

Every assistant span produced under the wrapper is one of:

- **`[CITED:<source_id>]…[/CITED]`** — the content is entailed by a labeled source provided to the model. Warranted confidence is bounded above by tool reliability.
- **`[INF:<confidence>]…[/INF]`** — the content is the model's inference. The confidence ∈ [0, 1] is explicit and calibrated: 0.9 means "wrong about 10% of the time on similar claims".
- **`[ABSTAIN]<reason>[/ABSTAIN]`** — the model declines to answer with a brief reason.

Anything outside these tags is a structural bug. `wrapper.py` retries the model up to `N` times with a correction prompt; on persistent failure it downgrades the reply to a single `Abstention` flagged as `contract_failure=True`, which the eval scores separately. The wrapper does not silently "best-effort parse" malformed output, because doing so would conceal exactly the dishonesty the contract exists to expose.

The contract is intentionally minimal. It does not enforce per-claim citations against the source text (that is the scorer's job), it does not check that confidences are well-calibrated (that is what ECE measures), and it does not stop the model from being wrong (that is what the eval surfaces). It only enforces *legibility*: every claim carries a tag that says how confident the model is and on what basis.

## Measurement

`eval.py` runs both the raw API and the wrapper on every example with identical sampling parameters (`temperature=0`). Each non-abstention span is scored by an LLM judge (`claude-opus-4-7`) against the gold answer with a fixed two-line rubric. Cited spans are assigned confidence 0.95 (a deliberate cap reflecting that even tool returns are not perfectly reliable); inference spans use the model's stated confidence; raw replies are scored as one constant-confidence claim at 1.0 (raw mode has no contract; treating it as fully confident is the most charitable comparison).

Reported metrics:

- **Hallucination rate** (raw vs. wrapped, with bootstrap 95% CIs at N=1000): fraction of judged claims marked `INCORRECT`.
- **Expected Calibration Error** over 10 confidence bins.
- **Brier score** on inference-flagged claims.
- **Selective accuracy** at coverage levels {25%, 50%, 75%, 100%} — accuracy when only the most-confident k% of claims are kept. Tells you whether the confidences are *useful* for selective prediction even if they are not perfectly calibrated.
- **Reliability diagram** — `reliability.png` (and an ASCII fallback in this document so a terminal-only reviewer can still see the shape).

The judge is itself a Claude model, which raises the obvious circularity concern: a model that judges its own kind may have systematic blind spots. The right answer is to cap the judge's authority by sampling a small holdout (~10 examples) and re-scoring it by hand, then reporting judge accuracy on that holdout in the final paragraph of `Results`. This is the next iteration; for v1, the rubric is constrained enough (paraphrase-tolerant exact-fact comparison) that judge agreement should be high.

## Results

<!-- RESULTS:START -->
_Run `python eval.py` to populate this section._
<!-- RESULTS:END -->

## What this doesn't do — and the natural extensions

The artifact stops at the wrapper-and-eval boundary. Three follow-on systems are sketched here at the level of detail an engineer could begin building from.

**Data curation pipeline.** Apply the wrapper to a stream of model-training prompts. Flag every example where the model's pre-wrapper response makes a claim that the post-wrapper response either downgrades to inference or abstains on. These are the training examples where the model "knew that it didn't know" but expressed itself as if it did — exactly the dishonest-by-overconfidence pattern the team wants to remove from the training distribution. Score flagged examples with the same scoring path used in `eval.py`; those with confirmed overconfidence get filtered or rewritten. The pipeline is a stream of pre-/post-wrapper diffs feeding a labeling queue.

**RL environment for honesty.** Reward shape (units arbitrary, ratios load-bearing):

```
cited-correct                       +1
inference, calibrated confidence    +0.5 × calibration_score   # 1 - |stated - empirical|
abstain on unanswerable             +0.5
abstain on answerable               -0.2
fabrication (uncited, wrong, conf>τ) -10
```

The asymmetry is the design: a single fabrication wipes out twenty calibrated answers. Calibration_score requires a held-out distribution of similar claims; in practice it is computed against a rolling buffer of recent same-domain inference verdicts. Abstention is rewarded enough to be a reliable escape hatch but not so much that the agent abstains on everything.

**Human feedback UI.** A labeler sees the user prompt, the wrapped model's response with each span color-coded by tag, and the source(s) the model had. For each non-abstention span, three buttons: *correct*, *incorrect*, *underconfident*. The third covers the symmetric dishonesty case (model said 0.4 about something it should have said 0.9 on). Inter-labeler disagreements above a threshold trigger a third labeler and a category-specific review; persistent disagreement on a domain is itself a signal that the rubric needs sharpening.

## What would change my mind

Concrete and falsifiable:

- **If the wrapped model's ECE is not lower than raw on the v2 held-out adversarial set, the mechanism is wrong, not the eval.** I would not rescue the result by tuning the system prompt against v2; I'd report the negative and re-think.
- **If wrapped accuracy collapses while ECE improves**, the contract is suppressing useful signal — the model is over-abstaining. The right response is to rebalance the reward asymmetry (or, in this prompt-only setting, weaken the abstention guidance), not to claim the calibration win.
- **If judge accuracy on the human-labeled holdout is below 90%**, the eval numbers are not trustworthy and should be reported with that caveat front-and-center.

Boundary conditions worth naming explicitly:

- Cited-claim calibration is bounded above by tool reliability. If a source is wrong, a faithful citation is also wrong, and the model has done its job.
- The dataset is small (51 examples in v1). Bootstrap CIs will be wide; effect sizes need to clear that width to be meaningful.
- Confidence anchors (CITED → 0.95, raw → 1.0) are choices, not measurements. Different anchors would give different headline numbers but should not flip the *direction* of the wrapped-vs-raw comparison if the mechanism is real.

## What this repo demonstrates

It directly addresses five of the JD's eight responsibility areas: honesty benchmarks, classifiers for hallucination detection, RAG grounding, confidence/calibration methods, and accuracy/hallucination evaluation. It contains usable specifications (above) for three more: data curation, human feedback, and RL environments. It deliberately does not try to fake the responsibilities it cannot honestly demonstrate from a user-land repo — training-time interventions in particular. Honesty about scope is itself the signal.

## Out of scope (first commit)

Multi-model or cross-provider comparison. Separate JSONL dataset file. A web UI for human feedback. An actual RL training loop. A standalone README (this file does its job).
