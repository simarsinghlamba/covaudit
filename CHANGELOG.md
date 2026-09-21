# Changelog

All notable changes to covaudit are documented here.
The format follows Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

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
