# Changelog

All notable changes to covaudit are documented here.
The format follows Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

### Added
- Clopper-Pearson exact confidence interval for coverage (via SciPy).
- FAIL / LOW / OK status rule: FAIL only when the whole interval is below 1 - alpha.
- Per-group coverage table (ALL row + one row per group) with interval, set size and status.
- Worst-group gap.
- `covaudit audit --group --out`: prints the table with readable names, saves
  `audit_split_<group>.csv`.
- Tests for interval width, empty groups, status rules, table shape, and a
  property-based (hypothesis) test that intervals stay in [0, 1] and contain the estimate.

### Notes
- First audit, split CP, CA 2018, alpha 0.1, seeds 0-2 (the seeds reshuffle the same
  data, so they are not independent replications):
  - RAC1P: Black (n ~1,700 test) below target on all 3 seeds (0.885, 0.885, 0.899;
    FAIL, FAIL, LOW). Native Hawaiian/PI (n ~140) below on all 3 (0.874, 0.878, 0.852;
    LOW each time: too few people for the interval to exclude 90%).
    Alaska Native has ~2 test people: its status carries no information.
    Worst-group gap 0.026 / 0.022 / 0.048.
  - SEX: no failure (Female ~0.906, Male ~0.898, i.e. at target).
  - Shortfalls are small (1-5 points). Large groups at 0.899 show "LOW" but are at target.
- Worst-group gap uses raw coverage, so it is driven by the smallest groups;
  read it together with the status column.

## [0.1.0] - 2026-09-21

### Added
- Installable package with `covaudit` command (`--version`, `data`, `audit`).
- Download of ACSIncome (California 2018) via folktables with local cache and SHA-256 checksum.
- Seeded 60/20/20 train/calibration/test split, tested for no overlap and reproducibility.
- Gradient boosting base model.
- SplitConformal with the finite-sample (n + 1) threshold and infinite threshold when data is too small.
- Overall coverage and average set size.
- Synthetic data generator with a deliberately hard group (for tests).
- Tests for the threshold formula (hand example), the infinite case, and ~90% average coverage.
- Dockerfile and .dockerignore.

### Notes
- CA 2018: 195,665 rows, positive rate 0.4106; data identical in Colab and Docker (same SHA-256).
- Seed 0, alpha 0.1: coverage 0.9032, average set size 1.1818.
- Changing (n + 1) to n is caught by the hand-example and infinite-threshold tests,
  but not by the 20-seed coverage test.
