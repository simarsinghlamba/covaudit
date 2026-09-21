# covaudit

Audit and repair subgroup coverage of conformal prediction.

Conformal prediction wraps any classifier and promises that the true label lands inside its
prediction set at least 90% of the time (for α = 0.1). That promise is an *average over everyone*:
it can hold overall while one group of people is covered far less often, and the overall number
hides it. **covaudit** audits coverage group by group, attaches an exact (Clopper–Pearson)
confidence interval to every group so small groups are not falsely accused, repairs
under-coverage with Mondrian (per-group) conformal prediction, and reports what the repair costs
in prediction-set size. It is a small, reproducible, containerised Python package built on the
US Census ACSIncome data (via folktables).

**Status:** under development (Day 1: package skeleton). Results will be added as experiments
are run.

## License

Apache-2.0. See [LICENSE](LICENSE).
