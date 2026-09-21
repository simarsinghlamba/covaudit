"""Figures and the markdown report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without a screen (containers, CI)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLORS = {"FAIL": "#b03a3a", "LOW": "#8a8f98", "OK": "#2a7f8e", "ALL": "#1f3a5f"}


def _label(row):
    name = row["name"] if "name" in row and str(row["name"]) not in ("", "nan") else row["group"]
    return f"{name} (n={int(row['n'])})"


def plot_group_coverage(table, alpha, path, title=None):
    """Dot per group with its confidence interval; dashed line at 1 - alpha.

    Red = FAIL, grey = LOW, teal = OK, navy = ALL. Intervals running past the
    left edge (tiny groups) are clipped and marked with an arrow. Returns path.
    """
    rows = table[table["n"] > 0].reset_index(drop=True)
    target = 1 - alpha
    big = rows[rows["n"] >= 30]
    left = min((big["ci_low"].min() if len(big) else target) - 0.02, target - 0.05)
    left = max(0.0, left)

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.4))
    for i, row in rows.iterrows():
        y = len(rows) - 1 - i
        color = COLORS["ALL"] if row["group"] == "ALL" else COLORS.get(row["status"], "#555")
        lo = max(row["ci_low"], left)
        ax.plot([lo, row["ci_high"]], [y, y], color=color, lw=2)
        ax.plot(row["coverage"], y, "o", color=color, ms=6)
        if row["ci_low"] < left:
            ax.plot(left, y, "<", color=color, ms=6)
    ax.axvline(target, ls="--", color="black", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r) for _, r in rows.iloc[::-1].iterrows()])
    ax.set_xlim(left, 1.005)
    ax.set_xlabel("coverage with 95% Clopper-Pearson interval")
    ax.set_title(title or f"Per-group coverage (target {target:.0%})")
    handles = [Line2D([0], [0], color=COLORS[s], marker="o", lw=2, label=s)
               for s in ("FAIL", "LOW", "OK")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
