# covaudit architecture

How the code is organised, how data flows through it, and why each container service
exists. For the statistics behind the numbers, see [metrics.md](metrics.md).

---

## 1. Module map

Everything users install lives in `src/covaudit/`. Each module has one job.

| Module | Job | Main public names |
|---|---|---|
| `__init__.py` | version and the public API | `__version__`, `SplitConformal`, `MondrianConformal`, `group_coverage_table`, `coverage`, `average_set_size`, `clopper_pearson` |
| `data.py` | download ACSIncome, split rows, fingerprint files | `load_acs_income`, `split_indices`, `file_checksum`, `RACE_NAMES`, `SEX_NAMES` |
| `model.py` | train the (deliberately ordinary) base model | `train_model` |
| `conformal.py` | scores, the (n + 1) threshold, prediction sets | `conformal_threshold`, `true_label_scores`, `SplitConformal`, `MondrianConformal` |
| `metrics.py` | coverage, set size, exact intervals, the per-group audit | `is_covered`, `coverage`, `average_set_size`, `clopper_pearson`, `coverage_status`, `group_coverage_table`, `worst_group_gap` |
| `synthetic.py` | fake data with a known hard group, for tests | `make_hard_group_data` |
| `config.py` | read and check experiment settings | `load_config`, `load_default_config` |
| `defaults/` | built-in copies of `configs/repair.yaml` and `configs/shift.yaml` (a test keeps them identical) | |
| `experiment.py` | the multi-seed repair experiment and the shift experiment | `run_repair_experiment`, `summarise_repair`, `run_shift_experiment`, `summarise_shift` |
| `report.py` | figures and `report.md` | `plot_group_coverage`, `plot_repair_summary`, `plot_set_size`, `plot_shift`, `write_report` |
| `cli.py` | the `covaudit` command (`data`, `audit`, `run`, `shift`, `report`) | `main` |

**Dependency direction.** The core modules (`data`, `model`, `conformal`, `metrics`,
`synthetic`) do not import each other's callers. `experiment` builds on the core, `report`
builds on `metrics`, and `cli` sits on top and wires everything together:

```
            cli.py
           /   |   \
 experiment  report  config (+ defaults/)
     |    \    |
     |     metrics
     |    /
 data  model  conformal  synthetic
```

**Shared conventions.**

- A prediction set is a boolean array of shape (people, classes); column order is
  `model.classes_`; True means the label is in the set.
- Every random step takes a seed. There is no global randomness.
- Library functions return data. Only `cli.py` prints. Experiments write files into an
  output folder they create.

## 2. Data flow

```
census.gov --folktables--> data/2018/1-Year/psam_p06.csv   (downloaded once, SHA-256 recorded)
                                   |
                        load_acs_income  ->  X (10 features), y (income > $50,000)
                                   |
                        split_indices(n, seed)
               ┌───────────────────┼───────────────────┐
          train 60%          calibration 20%        test 20%
               |                   |                   |
          train_model       SplitConformal or     predict_sets
               |            MondrianConformal          |
               └──────> thresholds ─────────> group_coverage_table
                                                        |
                                    CSV tables -> report.py -> figures + report.md
```

The **repair experiment** repeats this for 20 seeds with both methods on the same model
and split. The **shift experiment** calibrates once in California and applies the same
thresholds to all rows of Texas, New York and Florida.

## 3. Outputs and evidence

| Folder | What | In Git? |
|---|---|---|
| `data/` | downloaded Census files | no (`.gitignore`; Census terms, size) |
| `outputs/` | anything Docker or local runs produce | no |
| `results/` | the committed evidence: CSVs, `run_info.json`, figures, `report.md` | yes |

Every experiment writes a `run_info.json` recording the covaudit version, the full
config, row counts, the data files' SHA-256, runtime, Python and library versions. It is
the receipt that lets anyone check how a result was produced.

## 4. Containers

The `Dockerfile` builds one image in two layers:

1. `requirements.lock` is copied and installed first. This heavy layer changes only when
   the lock file changes, so Docker caches it.
2. The code is copied and covaudit itself is installed with `--no-deps` (dependencies are
   already pinned). Code changes rebuild only this quick layer.

`docker-compose.yml` runs the same image as several services:

| Service | Command | Why it exists |
|---|---|---|
| `audit` | `covaudit run` with `configs/repair.yaml` | the 20-seed split vs Mondrian experiment |
| `shift` | `covaudit shift` with `configs/shift.yaml` | the California to other states experiment; runs in parallel with `audit` |
| `report` | `covaudit report` | starts only after **both** `audit` and `shift` finish successfully (`depends_on: service_completed_successfully`), so a report is never built from half-finished results |
| `tests` | `pytest -q` | optional (`--profile test`): the test suite inside the container |
| `notebook` | JupyterLab | optional (`--profile notebook`): the demo notebook, reachable only from the local machine (`127.0.0.1:8888`) |

`./data` and `./outputs` are mounted into the containers, so downloads are reused and
results appear on the host.

## 5. Reproducibility, layer by layer

| Layer | Mechanism |
|---|---|
| randomness | a seed for every split, model and synthetic dataset |
| data | downloaded by code, fingerprinted with SHA-256, never committed |
| libraries | exact versions in `requirements.lock`, used by Docker and CI |
| operating system | the `python:3.12-slim` image; CI pinned to `ubuntu-24.04` |
| settings | YAML configs, recorded in every `run_info.json` |
| checks | `tests/test_reproducibility.py`: same seed gives identical tables, a different seed gives different ones, and the base model is seeded |

With these pinned, `docker compose up --build` reproduces the committed `results/`
exactly.

## 6. Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request, on `ubuntu-24.04`:

- **python-tests**: installs `requirements.lock` and covaudit, then `ruff check .` and
  `pytest -q`.
- **docker-tests**: builds the image from the Dockerfile and runs `pytest -q` inside it.

A pull request is merged only when both are green.

## 7. Tests

| File | Protects against |
|---|---|
| `test_import.py` | broken packaging |
| `test_data.py` | leakage between piles; non-reproducible splits |
| `test_conformal.py` | the n vs n + 1 bug; a missing infinite case; broken coverage |
| `test_metrics.py` | wrong intervals (including a property-based test), wrong statuses, bad table shape |
| `test_report.py` | plots that need a screen; a report that crashes on missing results |
| `test_mondrian.py` | a repair that does not repair; crashes on tiny or unseen groups |
| `test_config.py` | missing or misspelt settings; built-in defaults drifting from `configs/` |
| `test_reproducibility.py` | hidden randomness |

Tests use synthetic data or row numbers only; none downloads anything.
