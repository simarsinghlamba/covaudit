# covaudit

![tests](https://github.com/simarsinghlamba/covaudit/actions/workflows/tests.yml/badge.svg)

Audit and repair subgroup coverage of conformal prediction.

Conformal prediction wraps any classifier and promises that the true label lands inside its
prediction set at least 90% of the time (for α = 0.1). That promise is an *average over everyone*:
it can hold overall while one group of people is covered far less often, and the overall number
hides it. **covaudit** audits coverage group by group, attaches an exact (Clopper–Pearson)
confidence interval to every group so small groups are not falsely accused, repairs
under-coverage with Mondrian (per-group) conformal prediction, and reports what the repair costs
in prediction-set size. It is a small, reproducible, containerised Python package built on the
US Census ACSIncome data (via folktables).

**Status:** under development. Results will be added as experiments are run.

## Run with Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running).

```bash
git clone https://github.com/simarsinghlamba/covaudit.git
cd covaudit
docker build -t covaudit:local .
docker run --rm covaudit:local covaudit --version
docker run --rm covaudit:local pytest -q
```

Download the data (ACSIncome, California 2018) into a local `data/` folder. The folder is
shared with the container, so the download happens only once:

```bash
# Linux / macOS
docker run --rm -v "$(pwd)/data:/app/data" covaudit:local covaudit data --state CA --year 2018
# Windows PowerShell
docker run --rm -v "${PWD}/data:/app/data" covaudit:local covaudit data --state CA --year 2018
```

The command prints the row count, the share of people earning over $50,000, and a SHA-256
checksum of the downloaded file, so you can confirm you have exactly the same data.
The first build takes a few minutes; later builds reuse cached layers.

## License

Apache-2.0. See [LICENSE](LICENSE).
