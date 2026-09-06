# Experiment Tracker

| Block | Status | Evidence / next action |
|---|---|---|
| B1 judge tests | passed locally | 39 total repository tests passed on 2026-09-06; remote VAGEN integration still to verify |
| B2 validation calibration | pending remote run | run `run_length_calibration.sh lengthcal_v2` on existing four checkpoints |
| B3 replay/manual audit | pending | audit all fallback-positive and ambiguous validation records after B2 |
| B4 case-first split | blocked on regeneration | audit complete: 200 rows, 13 cases, one connected component; regenerate from frozen manifest |
| B5 formal three-seed study | not started | requires regenerated data and larger training budget/GPU |

The previous one-seed, four-step runs remain workflow smoke tests and are not promoted to model-quality evidence.

