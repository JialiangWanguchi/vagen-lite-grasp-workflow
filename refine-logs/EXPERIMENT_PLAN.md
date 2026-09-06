# Experiment Plan: length-aware judging and case-disjoint GraSP evaluation

## Scope and claims

Primary claim C1: a deterministic length-first judge prevents repetitive limit-hitting completions from receiving reward while recovering a narrow set of semantically correct answer-format variants.

Primary claim C2: case-first question generation plus manifest-enforced materialization provides a leakage-resistant estimate of unseen-case performance.

Anti-claims: this script-scale run does not establish that SFT→GRPO is superior, clinically useful, or statistically stable. A gain observed after tuning on test, or on sample-random splits with shared cases, does not support either primary claim.

## Fixed experimental contract

- Model family: Qwen3-VL; current smoke model Qwen3-VL-2B-Instruct.
- Inputs: 2D frames plus text; one environment action (`max_turns=1`).
- Arms: base, SFT-only LoRA, GRPO-only LoRA, independent SFT→GRPO LoRA.
- Rollout and evaluation generation: vLLM; optimization: pinned VAGEN-Lite/VERL stack.
- Dataset target: A2/P1 each 100, split per task 70/10/20 after case-first regeneration.
- Selection data: train/validation only. Test is sealed until rules and cap are frozen.
- Formal comparison: three fixed seeds; current one-seed/four-step run is implementation evidence only.

## KPIs and gates

| KPI | Purpose | Gate |
|---|---|---|
| hard-negative correctness | ensure every trusted length stop gets reward 0 | 100% synthetic and replay tests |
| fallback precision | avoid false-positive rewards | 100% curated adversarial suite; manual audit of all fallback positives |
| strict exact / accepted match | separate protocol compliance from semantic correctness | report both, never accepted alone |
| length-stop rate | diagnose repetition/runaway generation | report by arm and task |
| case/image overlap | protect generalization estimate | zero across all split pairs |
| nonzero policy updates | distinguish execution from learning | report, not guaranteed by step count |

## Experiment blocks

### B1 — Judge unit and adversarial validation (must run)

Run CPU tests for strict JSON, lowercase labels, descriptions, P1 order variants, malformed JSON, duplicated answers, reasoning-body mentions, and correct text with length metadata. Acceptance: all cases deterministic; no length-limited output receives reward; ambiguous cases receive 0 and `review_required=true`.

### B2 — Validation-only length calibration (must run)

Generate the 20-row validation split for four arms at diagnostic cap 2048, temperature 0. Exclude length-limited outputs and compute pooled/per-arm mean, median, P90, P95 and max. Select the smallest `k*512` strictly greater than the pooled normal maximum. If too few normal completions remain or the max is near 2048, raise the diagnostic cap and repeat on validation only.

### B3 — Historical-output replay and manual fallback audit (must run)

Re-score saved predictions without regenerating. Inspect every fallback-positive and every `review_required` record. Acceptance: zero false-positive fallback rewards in the audited set. Do not change rules using test labels; test replay is descriptive only after freeze.

### B4 — Case-first regeneration and split audit (must run before effect claims)

Freeze a 5/4/4 case manifest, regenerate A2/P1 within each pool, materialize splits, and verify zero case/image/near-duplicate leakage. The existing 200 rows have one 13-case connected component and therefore fail this gate by design.

### B5 — Final training/evaluation (formal, three seeds)

Run all three training arms independently for each seed, evaluate once on sealed test with the frozen cap/judge, and report per-task strict/accepted accuracy, length-stop rate, fallback count, paired confidence intervals, GPU/time cost, adapter provenance, and all failures. Nice-to-have: ablate fallback reward 0 versus 0.5 and compare 512 versus calibrated rollout cap on larger hardware.

## Failure policy

- OOM or runtime failure is reported as failure, not silently replaced with another configuration.
- Length-limited generations are incorrect regardless of visible partial answer.
- A split audit failure blocks generalization claims but not local script debugging.
- The test set never determines the cap, fallback vocabulary, checkpoint, or stopping point.

