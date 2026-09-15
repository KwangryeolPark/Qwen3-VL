# Post-SFT Evaluation Plan

This document defines the steps to run after the GroundCUA full-parameter SFT finishes.

## 1. SFT outputs to keep

The training run should produce four evaluation snapshots plus the final output directory:

- ~0.5 epoch snapshot
- ~1.0 epoch snapshot
- ~1.5 epoch snapshot
- ~2.0 epoch snapshot
- final HF model in the main output directory

The four milestone snapshots are the primary checkpoints for analysis. The final 2.0-epoch snapshot and final HF model should represent the same end-of-training stage, so avoid double-counting them in result tables.

## 2. GroundCUA in-domain evaluation

Evaluate the following five model states on the fixed GroundCUA validation split (8K):

1. Base Qwen3-VL-8B-Instruct
2. SFT 0.5 epoch
3. SFT 1.0 epoch
4. SFT 1.5 epoch
5. SFT 2.0 epoch

Use the same prompt/tool-call format and coordinate normalization used during training and later EasyR1 RL.

Record at minimum:

- grounding accuracy / success metric used by the GroundCUA evaluator
- number of evaluated samples
- failed parsing count
- invalid action count, if applicable

Primary analysis:

- GUI gain at each SFT stage relative to Base
- whether GroundCUA performance saturates before 2 epochs
- checkpoint-to-checkpoint progression

## 3. Out-of-domain GUI grounding evaluation

Evaluate Base and the four SFT checkpoints on at least one held-out GUI grounding benchmark not used for GroundCUA SFT.

Preferred order:

1. ScreenSpot-Pro
2. OSWorld-G, if the evaluation pipeline is available and stable

The purpose is to separate:

- in-domain GroundCUA specialization
- transferable GUI grounding improvement
- possible overfitting to GroundCUA formatting/distribution

Do not use an external GUI benchmark for model selection after looking at final results unless that selection rule is declared beforehand. Treat it primarily as held-out generalization evidence.

## 4. General-performance evaluation

Base general-performance scores already exist and can be reused if the exact same evaluation setup is preserved.

Evaluate each of the four SFT checkpoints on:

- IFEval
- MMLU-Pro
- AIME-25
- MMMU-Pro

Use temperature = 0 for all evaluations, matching the established Base evaluation protocol.

Existing Base reference scores:

| Benchmark | Base |
| --- | ---: |
| IFEval | 83.92 |
| MMLU-Pro | 73.13 |
| AIME-25 | 40.00 |
| MMMU-Pro | 53.64 |
| Text Avg (IFEval/MMLU-Pro/AIME-25) | 65.68 |

For every checkpoint compute:

- absolute benchmark score
- delta from Base
- text-average delta
- MMMU-Pro delta separately

## 5. Main SFT result table

Prepare one consolidated table of the form:

| Model | GroundCUA Val | ScreenSpot-Pro | OSWorld-G | IFEval | MMLU-Pro | AIME-25 | MMMU-Pro | Text Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | optional | 83.92 | 73.13 | 40.00 | 53.64 | 65.68 |
| SFT 0.5 ep | TBD | TBD | optional | TBD | TBD | TBD | TBD | TBD |
| SFT 1.0 ep | TBD | TBD | optional | TBD | TBD | TBD | TBD | TBD |
| SFT 1.5 ep | TBD | TBD | optional | TBD | TBD | TBD | TBD | TBD |
| SFT 2.0 ep | TBD | TBD | optional | TBD | TBD | TBD | TBD | TBD |

Primary quantities:

- GUI Gain = GUI(stage) - GUI(Base)
- General Delta = General(stage) - General(Base)

Plot specialization trade-off where useful, e.g. GUI gain versus general-performance degradation across SFT progress.

## 6. Select the checkpoint for RL

Default plan: start GroundCUA RL from the 2.0-epoch SFT checkpoint to match the intended GroundCUA-style SFT -> RL recipe.

However, before starting RL, inspect the SFT trajectory. If GUI performance clearly saturates substantially earlier while general capability degrades afterward, preserve that observation for analysis. Do not silently change the RL starting checkpoint based on post-hoc benchmark cherry-picking; if a different starting checkpoint is tested, treat it as an explicit additional experiment.

## 7. GroundCUA RL / GRPO

Use EasyR1 and initialize from the chosen SFT HF checkpoint.

Planned setup:

- GroundCUA RL subset: approximately 10K samples
- fixed subset generated deterministically
- no overlap with GroundCUA validation 8K
- same `computer_use` tool-call output format used in SFT
- run a 1-2 step handoff smoke test before the full RL run

After RL, evaluate the final SFT+RL checkpoint on the same benchmark suite:

- GroundCUA validation
- ScreenSpot-Pro
- OSWorld-G if included in the SFT evaluation
- IFEval
- MMLU-Pro
- AIME-25
- MMMU-Pro

## 8. Final decomposition

Report the following deltas:

- `Delta_SFT = SFT-final - Base`
- `Delta_RL = SFT+RL - SFT-final`
- `Delta_Total = SFT+RL - Base`

This separates three questions:

1. How much GUI ability is gained through full-parameter SFT?
2. How much general capability changes during SFT?
3. Does RL add GUI performance, and if so, does it introduce additional general-performance degradation?

## 9. Execution order after training finishes

1. Verify all four milestone snapshots are loadable as normal Hugging Face Qwen3-VL checkpoints.
2. Run GroundCUA validation for Base + 4 SFT checkpoints.
3. Run held-out GUI grounding evaluation for Base + 4 SFT checkpoints.
4. Run general-performance benchmarks for the 4 SFT checkpoints; reuse Base scores only under the identical protocol.
5. Consolidate results and inspect the SFT specialization trajectory.
6. Prepare deterministic ~10K GroundCUA RL subset.
7. Run a short SFT -> EasyR1 handoff smoke test.
8. Run GroundCUA GRPO.
9. Repeat the same GUI/general evaluations on the final SFT+RL checkpoint.
10. Produce the final Base vs SFT vs SFT+RL comparison and trade-off plots.
