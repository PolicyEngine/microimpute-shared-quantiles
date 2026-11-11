"""Aggregate diagnostics for an imputed wealth block. Never returns records.

The headline symptom is property_wealth falling short of
main + other residential + non-residential property. In the WAS donor
property_wealth is the gross sum of seven property values, so it never falls
short. The share of violators depends on which households the share is taken
over, and the imputed "owners with other property" group changes size with the
draws, so every share is reported over three populations:

* the imputed group: imputed main > 0 and imputed other residential > 0;
* all households (a fixed denominator);
* the donor group, where a donor is available: households that own a main
  residence and other residential property in the donor (fixed population).

Groups with fewer than MIN_CELL records are suppressed (None).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_CELL = 10
TOLERANCE = 1.0  # GBP; the donor identity holds to within GBP 1

MAIN = "main_residence_value"
OTHER = "other_residential_property_value"
NONRES = "non_residential_property_value"
PROPERTY = "property_wealth"
GFW = "gross_financial_wealth"
NFW = "net_financial_wealth"

SPEARMAN_PAIRS = [
    ("savings", GFW),
    ("corporate_wealth", GFW),
    (PROPERTY, GFW),
    (PROPERTY, MAIN),
    (MAIN, OTHER),
    (GFW, NFW),
]


def _share(numerator: np.ndarray, denominator: np.ndarray, w: np.ndarray):
    if denominator.sum() < MIN_CELL:
        return None
    return float((w * (numerator & denominator)).sum() / (w * denominator).sum())


def violations(frame: pd.DataFrame) -> np.ndarray:
    components = frame[MAIN] + frame[OTHER] + frame[NONRES]
    return (frame[PROPERTY] + TOLERANCE < components).to_numpy()


def owners_with_other(frame: pd.DataFrame) -> np.ndarray:
    return ((frame[MAIN] > 0) & (frame[OTHER] > 0)).to_numpy()


def wealth_metrics(
    imputed: pd.DataFrame,
    weights: np.ndarray,
    variables: list,
    donor: pd.DataFrame | None = None,
) -> dict:
    w = np.asarray(weights, dtype=float)
    everyone = np.ones(len(imputed), dtype=bool)
    violate = violations(imputed)
    group = owners_with_other(imputed)
    below_main = (imputed[PROPERTY] + TOLERANCE < imputed[MAIN]).to_numpy()
    out = {
        "n_households": int(len(imputed)),
        "n_imputed_group": int(group.sum()) if group.sum() >= MIN_CELL else None,
        "imputed_group_share_of_households": _share(group, everyone, w),
        "violation_share_in_imputed_group": _share(violate, group, w),
        "below_main_share_in_imputed_group": _share(below_main, group, w),
        "violators_in_imputed_group_per_household": _share(violate & group, everyone, w),
        "violation_share_all_households": _share(violate, everyone, w),
        "nfw_above_gfw_share": _share(
            (imputed[NFW] > imputed[GFW] + TOLERANCE).to_numpy(), everyone, w
        ),
        "positive_share": {
            v: _share((imputed[v] > 0).to_numpy(), everyone, w) for v in variables
        },
        "weighted_total_bn": {
            v: float((w * imputed[v].to_numpy()).sum() / 1e9) for v in variables
        },
        "spearman": {
            f"{a}~{b}": float(spearmanr(imputed[a], imputed[b]).statistic)
            for a, b in SPEARMAN_PAIRS
        },
    }
    if donor is not None:
        donor_group = owners_with_other(donor)
        out["n_donor_group"] = (
            int(donor_group.sum()) if donor_group.sum() >= MIN_CELL else None
        )
        out["violation_share_in_donor_group"] = _share(violate, donor_group, w)
        out["imputed_group_violators_in_donor_group"] = _share(
            donor_group, violate & group, w
        )
    return out


def flatten(metrics: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def summarise(cells: list, mode_a: str, mode_b: str, cluster: str = "split") -> dict:
    """Mean per mode and the paired a-minus-b difference, SE clustered by split."""
    frame = pd.DataFrame(
        [{"mode": c["mode"], cluster: c[cluster], "base": c["base"], **flatten(c["metrics"])} for c in cells]
    )
    numeric = [
        c
        for c in frame.columns
        if c not in ("mode", cluster, "base") and pd.api.types.is_numeric_dtype(frame[c])
    ]
    summary = {}
    for mode, rows in frame.groupby("mode"):
        summary[mode] = {c: float(rows[c].mean()) for c in numeric}
    a = frame[frame["mode"] == mode_a].set_index([cluster, "base"])[numeric]
    b = frame[frame["mode"] == mode_b].set_index([cluster, "base"])[numeric]
    diff = (a - b).dropna(how="all")
    by_cluster = diff.groupby(level=0).mean()
    k = len(by_cluster)
    summary[f"{mode_a}_minus_{mode_b}"] = {
        c: {
            "mean": float(diff[c].mean()),
            "se_clustered": float(by_cluster[c].std(ddof=1) / np.sqrt(k)) if k > 1 else None,
            "se_cells": float(diff[c].std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None,
        }
        for c in numeric
    }
    return summary
