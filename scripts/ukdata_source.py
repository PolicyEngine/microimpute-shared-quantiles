"""Load uk-data's WAS cleaning code and constants straight from git.

The licensed-data scripts must clean WAS exactly as policyengine-uk-data
does, so they take ``generate_was_table`` and the variable lists from a pinned
uk-data commit instead of a hand copy. Importing the module itself would pull
in policyengine-uk and the developer's working tree, so the pieces are parsed
out of ``git show <ref>:<path>`` and executed on their own.
"""

import ast
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

UKDATA_REPO = Path(
    os.environ.get("UKDATA_REPO", "~/PolicyEngine/policyengine-uk-data")
).expanduser()
UKDATA_REF = os.environ.get("UKDATA_REF", "origin/main")
STORAGE = UKDATA_REPO / "policyengine_uk_data" / "storage"
WAS_TAB = STORAGE / "was_2006_22" / "was_round_8_hhold_eul_may_2025_230525.tab"

WEALTH_PATH = "policyengine_uk_data/datasets/imputations/wealth.py"
UPRATING_PATH = "policyengine_uk_data/datasets/imputations/regional_property_uprating.py"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(UKDATA_REPO), *args], text=True)


def ukdata_commit() -> str:
    return _git("rev-parse", UKDATA_REF).strip()


def _exec_parts(path: str, names: set) -> dict:
    tree = ast.parse(_git("show", f"{UKDATA_REF}:{path}"))
    keep = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    found = {
        t.id for n in keep if isinstance(n, ast.Assign) for t in n.targets
    } | {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    missing = names - found
    if missing:
        raise KeyError(f"{path} at {UKDATA_REF} has no {sorted(missing)}")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), namespace)
    return namespace


def wealth_parts() -> dict:
    """REGIONS, PREDICTOR_VARIABLES, IMPUTE_VARIABLES, WAS_RENAMES, generate_was_table."""
    return _exec_parts(
        WEALTH_PATH,
        {
            "REGIONS",
            "PREDICTOR_VARIABLES",
            "IMPUTE_VARIABLES",
            "WAS_RENAMES",
            "generate_was_table",
        },
    )


def calibrate_property_to_hpi(household: pd.DataFrame) -> pd.DataFrame:
    """uk-data's post-imputation HPI rescaling, run on a household frame."""
    parts = _exec_parts(
        UPRATING_PATH,
        {"_load_regional_house_prices", "_calibrate_property_to_hpi"},
    )
    parts["REGIONAL_LAND_VALUES_CSV"] = STORAGE / "regional_land_values.csv"
    return parts["_calibrate_property_to_hpi"](household)


def load_was_table() -> pd.DataFrame:
    parts = wealth_parts()
    raw = pd.read_csv(WAS_TAB, sep="\t", low_memory=False)
    return parts["generate_was_table"](raw)
