import pytest

from covaudit.config import load_config


def test_repair_config_loads():
    cfg = load_config("configs/repair.yaml")
    assert cfg["alpha"] == 0.1 and len(cfg["seeds"]) == 20


def test_missing_setting_is_an_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("group_column: RAC1P\naplha: 0.1\nmethods: [split]\n")  # typo
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_unknown_method_is_an_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("group_column: RAC1P\nalpha: 0.1\nmethods: [split, mondrain]\n")
    with pytest.raises(ValueError, match="mondrain"):
        load_config(p)
