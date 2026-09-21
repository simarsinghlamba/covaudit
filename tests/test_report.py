import pandas as pd

from covaudit.report import plot_group_coverage, write_report


def _tiny_table():
    return pd.DataFrame({
        "group": ["ALL", "a", "b"], "name": ["everyone", "A", "B"],
        "n": [100, 60, 40], "covered": [88, 57, 31],
        "coverage": [0.88, 0.95, 0.775], "ci_low": [0.80, 0.86, 0.62],
        "ci_high": [0.94, 0.99, 0.89], "avg_set_size": [1.2, 1.1, 1.4],
        "status": ["LOW", "OK", "FAIL"],
    })


def test_plot_writes_png_without_a_display(tmp_path):
    path = plot_group_coverage(_tiny_table(), alpha=0.1, path=tmp_path / "fig.png")
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_report_skips_missing_parts_instead_of_crashing(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path, notes = write_report(results, tmp_path / "report")
    assert path.exists()
    assert len(notes) == 3  # no audits, no repair results, no shift results
