import numpy as np

from covaudit.metrics import clopper_pearson, coverage_status, group_coverage_table


def test_smaller_groups_get_wider_intervals():
    small = clopper_pearson(20, 25)
    big = clopper_pearson(2000, 2500)
    assert (small[1] - small[0]) > (big[1] - big[0])


def test_empty_group_gives_full_interval():
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_status_rules():
    assert coverage_status(0.84, 0.853, alpha=0.1) == "FAIL"
    assert coverage_status(0.80, 0.932, alpha=0.1) == "LOW"
    assert coverage_status(0.92, 0.99, alpha=0.1) == "OK"


def test_group_table_has_all_row_and_one_row_per_group():
    y = np.array([0, 1, 1, 0])
    sets = np.array([[True, False], [False, True], [True, False], [True, True]])
    table = group_coverage_table(y, sets, groups=["a", "a", "b", "b"], classes=[0, 1])
    assert list(table["group"]) == ["ALL", "a", "b"]
    assert table.loc[table.group == "b", "coverage"].item() == 0.5
