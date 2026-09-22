"""covaudit: audit and repair subgroup coverage of conformal prediction."""

__version__ = "1.0.0"

from covaudit.conformal import MondrianConformal, SplitConformal  # noqa: E402
from covaudit.metrics import (  # noqa: E402
    average_set_size,
    clopper_pearson,
    coverage,
    group_coverage_table,
)

__all__ = [
    "__version__",
    "SplitConformal",
    "MondrianConformal",
    "group_coverage_table",
    "coverage",
    "average_set_size",
    "clopper_pearson",
]
