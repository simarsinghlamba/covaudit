"""Read experiment settings from YAML files (or the defaults shipped with covaudit)."""
from importlib.resources import files

import yaml

REQUIRED = ("group_column", "alpha", "methods")
METHODS = {"split", "mondrian"}
DEFAULTS = ("repair", "shift")


def _check(cfg, source):
    """Raise ValueError unless cfg has the settings every experiment needs."""
    if not isinstance(cfg, dict):
        raise ValueError(f"{source}: expected a mapping of settings")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{source}: missing required settings {missing}")
    if not 0 < float(cfg["alpha"]) < 1:
        raise ValueError(f"{source}: alpha must be between 0 and 1")
    unknown = set(cfg["methods"]) - METHODS
    if unknown:
        raise ValueError(f"{source}: unknown methods {sorted(unknown)}")
    return cfg


def load_config(path):
    """Load a YAML config file as a dict and check it.

    Raises ValueError for missing keys, alpha outside (0, 1), or unknown methods.
    """
    with open(path) as f:
        return _check(yaml.safe_load(f), path)


def load_default_config(name):
    """Load a default config shipped inside the package: 'repair' or 'shift'."""
    if name not in DEFAULTS:
        raise ValueError(f"unknown default config {name!r}; choose from {DEFAULTS}")
    text = files("covaudit").joinpath("defaults").joinpath(f"{name}.yaml").read_text()
    return _check(yaml.safe_load(text), f"built-in {name}.yaml")
