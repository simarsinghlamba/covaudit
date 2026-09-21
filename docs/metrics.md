# How covaudit computes its numbers

This page explains the statistics behind covaudit: what each number means, why it is
computed the way it is, and what it does **not** promise. Every example uses covaudit's
own results (ACSIncome, California 2018, alpha = 0.1, seed 0 unless stated).

---

## 1. The nonconformity score

For a person with features $x$ and a candidate label $y$, the score is

$$s(x, y) = 1 - \hat p_y(x),$$

where $\hat p_y(x)$ is the probability the model gives to label $y$.
A low score means the model found that label plausible; a high score means it found it
unlikely. During calibration we use the score of the **true** label, because that is
exactly "how wrong was the model about this person".

With two classes the two scores of a person always add up to 1:
$(1 - \hat p_0) + (1 - \hat p_1) = 1$. So at least one label always has a score of 0.5 or
less, which matters in section 2.

## 2. The threshold, and why it uses n + 1

Sort the $n$ calibration scores and take the $k$-th smallest, where

$$k = \lceil (n + 1)(1 - \alpha) \rceil .$$

That score is the threshold $q$. The prediction set for a new person keeps every label
whose score is at or below $q$:

$$C(x) = \{\, y : s(x, y) \le q \,\}.$$

**Worked example** (from the Concept Guide, also a unit test): ten scores
0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.45, 0.60, 0.70, 0.85 with $\alpha = 0.2$ give
$k = \lceil 11 \times 0.8 \rceil = \lceil 8.8 \rceil = 9$, so $q = 0.70$, the 9th smallest.

**Why $n + 1$.** Treat the new person as one more member of the calibration group, giving
$n + 1$ people in total. If they are exchangeable (section 3), the new person's score is
equally likely to fall at any of the $n + 1$ ranks. It is at or below the $k$-th smallest
calibration score in at least $k$ of those $n + 1$ positions, so the probability of
coverage is at least $k / (n + 1) \ge 1 - \alpha$. Using $n$ instead of $n + 1$ can make
$k$ one smaller, the threshold slightly too small, and the promise quietly fails. In the
hand example it would pick 0.60 instead of 0.70. covaudit's tests catch this: replacing
$(n + 1)$ by $n$ makes the hand-example and infinite-threshold tests fail. (A statistical
test of "coverage is about 90%" does **not** catch it with 1,000 calibration points, which
is why exact formula tests exist.) A common related bug is `np.quantile` with its default
interpolation, which gives 0.62 on the same example.

**When $k > n$.** With too few calibration scores, no score is large enough to guarantee
$1 - \alpha$. The only honest threshold is **infinity**: every label is kept. For
$\alpha = 0.1$ this happens whenever $n \le 8$ (for $n = 8$, $k = \lceil 8.1 \rceil = 9$).

**Upper bound.** If scores have no ties, coverage is also at most
$1 - \alpha + 1/(n + 1)$. With 39,133 calibration people that is 0.90003, which is why
split conformal lands so close to 0.900 overall.

In California (seed 0): $q = 0.6588$, coverage **0.9032**, average set size **1.1818**.
Because $q > 0.5$, no set is ever empty (section 1).

## 3. The guarantee, and exchangeability

Split conformal prediction promises

$$P\big(Y_{\text{new}} \in C(X_{\text{new}})\big) \ge 1 - \alpha .$$

The probability is taken **over both the new person and the random calibration pile**.
So it is a statement about the average over many people and many calibration draws. It is
**not** a promise about any single person, and **not** a promise about any particular
group.

It requires **exchangeability**: calibration people and new people must come from the
same population in random order, like cards dealt from one shuffled deck. When the new
people come from a different population the promise can break. covaudit measures this:
thresholds calibrated in California give overall coverage of 0.884 in Texas, 0.884 in
New York and 0.869 in Florida (section 6).

## 4. Coverage and set size

- **Coverage** is the fraction of people whose set contains their true label. It checks
  whether the promise is kept.
- **Average set size** is the average number of labels per set, between 0 and 2 here. It
  measures how useful the sets are.

Both are needed. A set of {yes, no} for everyone has perfect coverage and is useless;
very small sets are useful only if coverage still holds. The goal is sets as small as
possible while keeping the promise. Tightening the promise costs size: at
$\alpha = 0.05$ covaudit gets coverage 0.9523 but average set size 1.3597 (threshold 0.7848).

## 5. Clopper-Pearson intervals and the FAIL / LOW / OK rule

Coverage inside a group is $k$ covered people out of $n$: a binomial proportion. Small
groups vary a lot by chance. A group whose true coverage is 90% can easily show 80% in a
sample of 25.

For each group covaudit reports a 95% **Clopper-Pearson** interval:

$$\text{low} = B^{-1}\!\left(0.025;\ k,\ n - k + 1\right), \qquad
\text{high} = B^{-1}\!\left(0.975;\ k + 1,\ n - k\right),$$

where $B^{-1}$ is the inverse Beta distribution (low = 0 when $k = 0$, high = 1 when
$k = n$). It is called **exact** because it reaches at least 95% for every possible true
value, even for tiny groups, where normal-approximation intervals are too narrow. For an
audit that may accuse a system, being conservative is the right choice. (covaudit computes
it with SciPy; all 520 committed result rows were re-derived from this Beta formula and
matched.)

The status rule, with target $1 - \alpha = 0.90$:

| Status | Rule | Meaning |
|---|---|---|
| **FAIL** | upper end of interval < 0.90 | even the most optimistic believable value breaks the promise |
| **LOW** | coverage < 0.90, upper end >= 0.90 | below target, but chance could explain it |
| **OK** | coverage >= 0.90 | promise kept in this sample |

Examples from California, split conformal, seed 0:

- Black: 1,503 of 1,698 covered = 0.8852, interval 0.8690 to 0.8999, so **FAIL**.
- Native Hawaiian/Pacific Islander: 125 of 143 = 0.8741, interval 0.8084 to 0.9237, so
  **LOW**. Lower coverage than Black, but too few people to rule out 90%.
- Alaska Native: 2 of 2 = 1.0, interval 0.158 to 1.0, so **OK**. Here "OK" only means
  "no evidence either way".

Two cautions:

1. **Multiple comparisons.** Checking about nine groups at 95% each makes a false FAIL
   somewhere more likely than 5%. covaudit reports this as a limitation; a Bonferroni-style
   correction (wider intervals when many groups are checked) is a simple extension.
2. **What the interval covers.** It accounts for randomness in the **test** people only,
   for one fixed calibration pile. It does not include the randomness of the calibration
   pile itself. Section 6 explains why that matters for Mondrian.

## 6. Mondrian conformal prediction

**How.** Train one model as before. Split the calibration scores **by group** and compute
a separate threshold for each group with the same rule, where $n$ is now that group's
calibration size. A new person gets their own group's threshold.

**Why it is valid.** The $n + 1$ argument only needs exchangeability. Among the people of
one group, calibration people and a new person of that group are still exchangeable, so
the argument holds **inside each group**, and each group gets its own guarantee:
$P(Y \in C(X) \mid \text{group} = g) \ge 1 - \alpha$.

**What it costs and needs.**

- The group must be **known at prediction time**. For attributes such as race this raises
  legal and ethical questions.
- Groups with 8 or fewer calibration people get an infinite threshold. Alaska Native has
  about 2 calibration people in California, so under Mondrian everyone in it gets both
  labels (set size 2.0): honest, but uninformative.
- Harder groups get looser thresholds and bigger sets; easier groups get stricter
  thresholds and smaller sets. In California (seed 0) Black's threshold is 0.6798 against
  the shared 0.6588, and its average set size rises from 1.225 to 1.255. Overall set size
  barely moves (1.1795 to 1.1803 averaged over 20 seeds).
- It protects only the groups you define; a failing intersection you did not define stays
  hidden. It does not make the model more accurate or "fair" in general.

**"On average over calibration draws".** Mondrian's per-group promise, like split's
overall promise, is an average over calibration piles. On a single run a small group's
threshold is noisy, so its coverage can land a few points short, and the FAIL rule
(section 5, caution 2) may flag it. Over 20 seeds, Mondrian brings each group's mean
coverage to about 0.90, while occasional single-run FAILs remain. With groups this varied,
"seeds with any FAIL group" is a weak summary: for race it changes little (10 of 20 seeds
under split, 9 under Mondrian), while for sex, where both groups are large, it drops from
12 to 4.

**Not a fix for distribution shift.** Mondrian corrects differences **between groups
within the calibration population**. It cannot know that Texans differ from Californians:
under shift its overall coverage is as low as split's (Florida 0.8695 vs 0.8690). It can
even transfer worse for a group whose members differ between states: American Indian in
Florida gets 0.828 under Mondrian and 0.854 under split, because California's per-group
threshold for that group was stricter than the shared one. Repairing shift needs methods
such as weighted conformal prediction (future work).

## 7. The worst-group gap

$$\text{gap} = \max\Big(0,\ (1 - \alpha) - \min_{g:\, n_g > 0} \text{coverage}_g\Big)$$

It is how far the lowest-covered group falls below the promise, or 0 if none does.

Its weakness: it uses raw coverage and ignores the intervals, so it is driven by the
**smallest, noisiest** groups. Alaska Native has about 3 test people; on a seed where one
of them is missed, its coverage is 0.67 and the gap jumps. That is why, for race, split's
mean gap over 20 seeds is 0.0635 with a standard deviation of 0.0964. Under Mondrian that
group always gets full sets (coverage 1.0), so it never drives the gap, and the mean is
0.0296. Always read the gap next to the per-group statuses and intervals, never alone.

---

**Reproducibility.** All numbers above come from committed results in `results/`,
produced with the pinned library versions in `requirements.lock`, and they are
regenerated identically by `docker compose up --build`.
