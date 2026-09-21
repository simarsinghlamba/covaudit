"""Read experiment settings from YAML files."""
import yaml

REQUIRED = ("group_column", "alpha", "methods")
METHODS = {"split", "mondrian"}


def load_config(path):
    """Load a YAML config as a dict and check the settings every experiment needs.

    Raises ValueError for missing keys, alpha outside (0, 1), or unknown methods.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping of settings")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required settings {missing}")
    if not 0 < float(cfg["alpha"]) < 1:
        raise ValueError(f"{path}: alpha must be between 0 and 1")
    unknown = set(cfg["methods"]) - METHODS
    if unknown:
        raise ValueError(f"{path}: unknown methods {sorted(unknown)}")
    return cfg
