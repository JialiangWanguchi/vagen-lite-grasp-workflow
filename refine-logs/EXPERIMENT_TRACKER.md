# Experiment Tracker

| Block | Status | Evidence / next action |
|---|---|---|
| B1 judge tests | passed | 41 repository tests passed locally, 28 focused tests remotely; VAGEN bridge and real environment reward check passed |
| B2 validation calibration | passed | 80/80 validation generations completed; 21 length stops excluded; normal max 820; selected cap 1024 |
| B3 replay/manual audit | passed for current validation | 80 records replayed; 3 fallback-positive records manually checked, 0 false positives; 0 review-required |
| B4 case-first split | blocked on regeneration | audit complete: 200 rows, 13 cases, one connected component; regenerate from frozen manifest |
| B5 formal three-seed study | not started | requires regenerated data and larger training budget/GPU |

The previous one-seed, four-step runs remain workflow smoke tests and are not promoted to model-quality evidence.
