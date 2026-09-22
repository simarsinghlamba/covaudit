# covaudit

![tests](https://github.com/simarsinghlamba/covaudit/actions/workflows/tests.yml/badge.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simarsinghlamba/covaudit/blob/main/notebooks/demo.ipynb)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

**Audit and repair subgroup coverage of conformal prediction.**

Conformal prediction wraps any classifier and promises that the true label lands inside
its prediction set at least 90% of the time (for alpha = 0.1). That promise is an
**average over everyone**. covaudit checks it **group by group**, attaches an exact
(Clopper-Pearson) interval to every group so small groups are not falsely accused,
repairs under-coverage with **Mondrian** (per-group) conformal prediction, and reports
what the repair costs. On US Census data (ACSIncome, California 2018, 20 seeds), split
conformal keeps its 90% promise overall (0.900) while repeatedly failing specific groups:
Black residents are flagged FAIL on 5 of 20 seeds (mean coverage 0.892) and men on 12 of
20 (0.895). Mondrian brings every group with enough data to 0.900-0.908 at almost no
overall cost (average set size 1.1795 to 1.1803). Under distribution shift (California
thresholds used in other states) coverage drops to 0.869-0.884 for both methods.

---

## Why this matters

A 90% promise that holds on average can hide a group that is covered far less often,
and nobody notices because everybody checks only the overall number. When the groups
are patients, borrowers or job applicants, that means some people are systematically
served by a less reliable system.

## What covaudit does

| Step | What | How |
|---|---|---|
| Audit | coverage and set size for every group, not only overall | `group_coverage_table` |
| Honest uncertainty | exact 95% Clopper-Pearson interval per group; FAIL only if the whole interval is below 90% | `clopper_pearson`, FAIL / LOW / OK |
| Repair | one threshold per group from that group's own calibration data | `MondrianConformal` |
| Cost | how much bigger the prediction sets get, per group | average set size |
| Shift | calibrate in one state, test in others | `covaudit shift` |

---

## Results

All numbers come from the committed evidence in [`results/`](results/) (20 seeds unless
stated), produced with the pinned library versions in `requirements.lock`. The full
report is [`results/report/report.md`](results/report/report.md).

### By race: split hides repeated failures, Mondrian repairs them

![Mean coverage per group over 20 seeds](results/report/repair_coverage.png)

| Group | People (test) | Coverage split | Coverage Mondrian | FAIL seeds split | FAIL seeds Mondrian | Set size split | Set size Mondrian |
|---|---|---|---|---|---|---|---|
| everyone | 39,133 | 0.9002 | 0.9003 | 1 | 0 | 1.1795 | 1.1803 |
| White | 24,207 | 0.8991 | 0.8997 | 2 | 2 | 1.1839 | 1.1857 |
| Black | 1,701 | **0.8917** | 0.9007 | **5** | 1 | 1.2181 | 1.2458 |
| Asian | 6,552 | **0.8974** | 0.9010 | **4** | 1 | 1.1860 | 1.1963 |
| Native Hawaiian/PI | 127 | **0.8846** | 0.9008 | 0 | 2 | 1.2428 | 1.2852 |
| Some other race | 4,555 | 0.9101 | 0.9014 | 0 | 1 | 1.1313 | 1.1083 |
| Two or more races | 1,640 | 0.9072 | 0.9020 | 0 | 1 | 1.1766 | 1.1615 |
| American Indian | 259 | 0.9069 | 0.9016 | 0 | 2 | 1.1889 | 1.1777 |
| AI/AN tribes | 90 | 0.9213 | 0.9084 | 1 | 1 | 1.1849 | 1.1629 |
| Alaska Native | 3 | 0.9396 | 1.0000 | 0 | 0 | 1.0530 | 2.0000 |

- Under split, under-covered groups sit below 0.90 on average and some are flagged FAIL
  on the same group again and again (Black 5/20, Asian 4/20). Native Hawaiian/PI is
  lowest on average but, with about 127 people, never has enough data for a FAIL.
- Under Mondrian every group with enough data averages 0.900-0.908, and the remaining
  FAILs are scattered (at most 2 of 20 seeds per group), as expected when a guarantee
  holds on average over calibration draws.
- The cost is honest and small overall: under-covered groups get bigger sets (Black
  1.218 to 1.246), over-covered groups get smaller ones (Some other race 1.131 to 1.108).
- Alaska Native has about 3 test people and 2 calibration people, so Mondrian keeps both
  labels for everyone in it (set size 2.0): honest, but uninformative.

![Prediction-set size per group](results/report/repair_set_size.png)

### By sex

| Group | Coverage split | Coverage Mondrian | FAIL seeds split | FAIL seeds Mondrian |
|---|---|---|---|---|
| Male | **0.8954** | 0.9008 | **12** | 1 |
| Female | 0.9056 | 0.8995 | 0 | 3 |

Split conformal slightly under-covers men and over-covers women, consistently; Mondrian
brings both to about 0.90.

### Distribution shift: calibrated in California, used elsewhere (seed 0)

![Coverage under distribution shift](results/report/shift.png)

| State | People | Coverage split | Coverage Mondrian |
|---|---|---|---|
| California (no shift) | 39,133 | 0.9032 | 0.9032 |
| Texas | 135,924 | 0.8836 | 0.8839 |
| New York | 103,021 | 0.8841 | 0.8848 |
| Florida | 98,925 | 0.8690 | 0.8695 |

The guarantee needs calibration and new people to come from the same population. In
other states it breaks for everyone, prediction sets barely grow (1.18 to 1.20-1.21), and
Mondrian does not help: it can even transfer worse for a group whose members differ
between states (American Indian in Florida: 0.828 Mondrian vs 0.854 split).

---

## Quick start (Google Colab, no setup)

Click the **Open in Colab** badge above and choose **Runtime -> Run all**, or install the
released version yourself:

```python
%pip install "git+https://github.com/simarsinghlamba/covaudit@v1.0.0"
```

```python
from covaudit import MondrianConformal, SplitConformal, group_coverage_table
from covaudit.data import load_acs_income, split_indices
from covaudit.model import train_model

X, y = load_acs_income("CA", "2018")          # downloads ~270 MB from census.gov once
s = split_indices(len(X), seed=0)
model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=0)
g_cal, g_test = X["RAC1P"].iloc[s["cal"]], X["RAC1P"].iloc[s["test"]]

mond = MondrianConformal(model, alpha=0.1).calibrate(X.iloc[s["cal"]], y.iloc[s["cal"]], g_cal)
sets = mond.predict_sets(X.iloc[s["test"]], g_test)
print(group_coverage_table(y.iloc[s["test"]], sets, g_test, model.classes_, alpha=0.1))
```

The installed package also provides the `covaudit` command; `covaudit run` and
`covaudit shift` work without a config file (the default settings ship with the package).

## Reproduce everything (Docker)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running).

```bash
git clone https://github.com/simarsinghlamba/covaudit.git
cd covaudit
git switch --detach v1.0.0
docker compose up --build                       # audit and shift, then report
docker compose --profile test run --rm tests    # the test suite in the container
docker compose --profile notebook up notebook   # optional: JupyterLab on http://127.0.0.1:8888
docker compose down
```

- `audit` runs the 20-seed split vs Mondrian experiment (race); `shift` runs the
  California to Texas / New York / Florida experiment in parallel; `report` starts only
  after both finish successfully.
- Everything appears on your computer in `outputs/`: `outputs/repair/`,
  `outputs/shift/` and `outputs/report/` (figures and `report.md`).
- The first run builds the image (it installs pinned libraries and can take a while on a
  slow connection) and downloads four Census files into `data/` (several hundred MB).
  Later runs reuse both; the experiments then take about a minute each.
- `outputs/repair/summary.csv`, `outputs/repair/group_coverage.csv` and
  `outputs/shift/shift_coverage.csv` are byte-for-byte identical to the committed files
  in `results/` (library versions are pinned in `requirements.lock`).
- If you ran an older version of covaudit's containers, delete `outputs/` once first:
  files created by the old root user cannot be overwritten by the new non-root user.

## Usage

| Command | What it does |
|---|---|
| `covaudit data --state CA --year 2018` | download (or reuse) the data; print rows, positive rate and SHA-256 |
| `covaudit audit --group RAC1P --method split` | one seed: per-group table with intervals and status; saves a CSV |
| `covaudit audit --group SEX --method mondrian` | the same with Mondrian, grouped by sex |
| `covaudit run [--config FILE]` | the multi-seed split vs Mondrian experiment |
| `covaudit shift [--config FILE]` | calibrate in one state, audit in others |
| `covaudit report --results outputs --out outputs/report` | figures and `report.md` from whatever results exist |

All commands accept `--root` (data folder) and `--out` (output folder) where relevant;
`covaudit <command> --help` lists every option. Experiment settings live in
[`configs/`](configs/).

## Data

- **ACSIncome** from [folktables](https://github.com/socialfoundations/folktables)
  (MIT License), built on the US Census Bureau's American Community Survey (ACS) Public
  Use Microdata Sample, 2018, 1-Year, person records.
- Task: predict whether a working adult earns more than $50,000. California 2018 has
  **195,665 people** (41.06% positive). Groups: race (`RAC1P`) and sex (`SEX`).
- Data are downloaded at run time from census.gov and **not redistributed** in this
  repository. Use is governed by the Census Bureau's terms of use; no attempt is made to
  identify individuals. The SHA-256 of each downloaded file is recorded in every
  `run_info.json` (California: `dc2187fc...b43e0`).
- Race and sex are used only to check whether the system treats groups equally.

## Methods

**Split conformal prediction.** A model is trained on 60% of the data. Its nonconformity
scores (1 minus the probability of the true label) are computed on a separate 20%
calibration pile, and one threshold is taken as the ceil((n + 1)(1 - alpha))-th smallest
score. Every label whose score is at or below the threshold goes into the prediction set.
This guarantees at least 1 - alpha coverage on average over everyone, not within groups.

**Mondrian conformal prediction.** The same model and scores, but each group gets its own
threshold from its own calibration people, using the same (n + 1) rule. Calibration and
test people within one group are still exchangeable, so each group gets its own
guarantee, on average over calibration draws. Groups with 8 or fewer calibration people
(alpha = 0.1) get an infinite threshold, so every label is kept.

The full explanation, with formulas and worked examples, is in
[`docs/metrics.md`](docs/metrics.md).

## How it works

- [`docs/metrics.md`](docs/metrics.md): scores, the (n + 1) threshold, the guarantee and
  exchangeability, Clopper-Pearson and the FAIL / LOW / OK rule, Mondrian, the
  worst-group gap.
- [`docs/architecture.md`](docs/architecture.md): module map, data flow, containers,
  reproducibility and CI.
- [`notebooks/demo.ipynb`](notebooks/demo.ipynb): a walkthrough you can run in Colab.

## Testing

24 tests cover the threshold formula (a hand-computed example that catches the n vs n + 1
bug), the infinite threshold, no leakage between piles, exact intervals (including a
property-based test with hypothesis), the status rule, the Mondrian repair on synthetic
data with a known hard group, tiny and unseen groups, configs, headless plotting and
reproducibility. None downloads data.

```bash
pip install -r requirements.lock && pip install --no-deps -e .
pytest -q
```

GitHub Actions runs `ruff` and `pytest` on every push and pull request, both on Python
3.12 with the pinned versions and inside the Docker image.

## Project structure

```
covaudit/
├── src/covaudit/        the package (conformal, metrics, data, model, experiment, report, cli)
├── tests/               24 tests (pytest, hypothesis)
├── configs/             experiment settings (repair.yaml, shift.yaml)
├── results/             committed evidence: CSVs, run_info.json, figures, report.md
├── notebooks/demo.ipynb walkthrough (opens in Colab)
├── docs/                metrics.md, architecture.md
├── Dockerfile, docker-compose.yml, requirements.lock
└── README.md, CHANGELOG.md, LICENSE, NOTICE
```

## Limitations

- Coverage guarantees are averages over people and calibration draws. A single run can
  fall short, even overall (the "everyone" row was FAIL on 1 of 20 split seeds).
- Mondrian needs each person's group at prediction time, which may not be legal or
  ethical in some applications; the grouping should be chosen with domain and legal input.
- Groups with very little calibration data receive full, uninformative sets.
- Only predefined groups are protected; failing intersections (for example race x sex x
  age) can stay hidden.
- Checking many groups makes a false FAIL more likely (multiple comparisons); no
  correction is applied.
- The worst-group gap uses raw coverage, so it is driven by the smallest groups; read it
  next to the per-group intervals.
- Distribution shift is measured (one seed), not repaired.
- One dataset and one model type; category codes are used as numbers; binary
  classification only.

## Version history

See [CHANGELOG.md](CHANGELOG.md).

- **1.0.0**: pinned and reproducible (byte-identical results in Docker), public API,
  built-in configs, non-root container, documentation.
- **0.4.0**: distribution-shift experiment, demo notebook, Docker notebook service.
- **0.3.0**: Mondrian repair, 20-seed evidence, Docker Compose.
- **0.2.0**: per-group audit with exact intervals, CI, first figures.
- **0.1.0**: split conformal prediction baseline.

## License

Apache-2.0, see [LICENSE](LICENSE). Data sources and their terms are listed in
[NOTICE](NOTICE).

## Citation and acknowledgements

- Ding, Hardt, Miller, Schmidt (2021). *Retiring Adult: New Datasets for Fair Machine
  Learning.* NeurIPS. (folktables and ACSIncome)
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.* Springer.
  (conformal and Mondrian prediction)
- Angelopoulos, Bates (2021). *A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Romano, Barber, Sabatti, Candes (2020). *With Malice Toward None: Assessing Uncertainty
  via Equalized Coverage.* Harvard Data Science Review.
- Clopper, Pearson (1934). *The Use of Confidence or Fiducial Limits Illustrated in the
  Case of the Binomial.* Biometrika.
- Tibshirani, Barber, Candes, Ramdas (2019). *Conformal Prediction Under Covariate
  Shift.* NeurIPS.

Author: Simar Singh Lamba.
