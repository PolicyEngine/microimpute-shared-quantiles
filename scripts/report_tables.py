"""Render the README's result tables from the aggregate JSON files in results/.

    python scripts/report_tables.py > results/tables.md

Reads only the summaries (means across cells and paired differences); no
records are involved at any point.
"""

import json
import os
import sys
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))

SHARES = [
    ("imputed_group_share_of_households", "Imputed owners with other property, share of households"),
    ("violation_share_in_imputed_group", "Property below components, share of imputed owners with other property"),
    ("violators_in_imputed_group_per_household", "Those violators, share of all households"),
    ("violation_share_all_households", "Property below components, share of all households"),
    ("violation_share_in_donor_group", "Property below components, share of donor owners with other property"),
    ("below_main_share_in_imputed_group", "Property below main residence alone, share of imputed owners with other property"),
    ("nfw_above_gfw_share", "Net financial wealth above gross, share of all households"),
]
SPEARMAN = [
    "savings~gross_financial_wealth",
    "corporate_wealth~gross_financial_wealth",
    "property_wealth~gross_financial_wealth",
    "property_wealth~main_residence_value",
    "main_residence_value~other_residential_property_value",
    "gross_financial_wealth~net_financial_wealth",
]
TOTALS = [
    "property_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "corporate_wealth",
    "gross_financial_wealth",
    "savings",
]


def _get(summary: dict, mode: str, key: str):
    return summary.get(mode, {}).get(key)


def _pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def _pp(d):
    if d is None or d.get("mean") is None:
        return ""
    se = d.get("se_clustered") if d.get("se_clustered") is not None else d.get("se_cells")
    return f"{100 * d['mean']:+.1f}pp" + ("" if se is None else f" ({100 * se:.1f})")


def _num(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def _diff(d, digits=2):
    if d is None or d.get("mean") is None:
        return ""
    se = d.get("se_clustered") if d.get("se_clustered") is not None else d.get("se_cells")
    return f"{d['mean']:+.{digits}f}" + ("" if se is None else f" ({se:.{digits}f})")


def table(summary: dict, columns: list, diff_key: str | None, diff_label: str) -> str:
    """columns: list of (mode, label)."""
    header = "| Metric | " + " | ".join(label for _, label in columns)
    rule = "|---|" + "---:|" * len(columns)
    if diff_key:
        header += f" | {diff_label}"
        rule += "---:|"
    lines = [header + " |", rule]
    diffs = summary.get(diff_key, {}) if diff_key else {}
    for key, label in SHARES:
        if all(_get(summary, m, key) is None for m, _ in columns):
            continue
        row = [label] + [_pct(_get(summary, m, key)) for m, _ in columns]
        if diff_key:
            row.append(_pp(diffs.get(key)))
        lines.append("| " + " | ".join(row) + " |")
    for key in SPEARMAN:
        full = f"spearman.{key}"
        if all(_get(summary, m, full) is None for m, _ in columns):
            continue
        row = [f"Spearman, {key.replace('~', ' vs ')}"] + [
            _num(_get(summary, m, full)) for m, _ in columns
        ]
        if diff_key:
            row.append(_diff(diffs.get(full)))
        lines.append("| " + " | ".join(row) + " |")
    for key in TOTALS:
        full = f"weighted_total_bn.{key}"
        if all(_get(summary, m, full) is None for m, _ in columns):
            continue
        row = [f"Weighted total, {key}, £bn"] + [
            _num(_get(summary, m, full), 0) for m, _ in columns
        ]
        if diff_key:
            row.append(_diff(diffs.get(full), 0))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    out = []
    was = RESULTS / "was_holdout.json"
    if was.exists():
        r = json.loads(was.read_text())
        m = r["meta"]
        sharing = [s["draw_sharing_production"]["share_rows_all_vectors_identical"] for s in m["per_split"]]
        out += [
            "## WAS round 8 holdout",
            "",
            f"microimpute {m['microimpute']}, quantile-forest {m['quantile_forest']}, "
            f"scikit-learn {m['scikit_learn']}; uk-data {m['ukdata_commit'][:8]}; "
            f"{m['n_was_households']:,} WAS households; {m['splits']} splits "
            f"({m['split_rule']}) x {len(m['bases'])} predict-seed bases {m['bases']}. "
            f"Production predict drew identical quantile indices for all "
            f"{m['per_split'][0]['draw_sharing_production']['n_draw_vectors']} variables "
            f"on {min(sharing):.0%} of holdout rows in every split. "
            "Differences are shared minus per-variable on the same forests, "
            "with standard errors clustered by split in brackets.",
            "",
            table(
                r["summary"],
                [("donor", "Donor (WAS)"), ("shared", "Shared draws (production)"), ("independent", "Per-variable draws")],
                "shared_minus_independent",
                "Shared minus per-variable",
            ),
            "",
        ]
    frs = RESULTS / "frs_counterfactual.json"
    if frs.exists():
        r = json.loads(frs.read_text())
        m = r["meta"]
        out += [
            "## FRS counterfactual",
            "",
            f"microimpute {m['microimpute']}, policyengine-uk {m['policyengine_uk']}; "
            f"uk-data {m['ukdata_commit'][:8]}; {m['frs_file']} ({m['n_frs_households']:,} "
            f"households, {m['frs_year']}); weights: {m['weights']}; "
            f"{len(m['bases'])} predict-seed bases {m['bases']}. "
            "The FRS has no wealth questions, so there is no donor column. "
            "Standard errors across bases in brackets.",
            "",
        ]
        for stage, title in (("imputed", "As imputed"), ("after_hpi", "After uk-data's regional HPI uprating")):
            out += [
                f"### {title}",
                "",
                table(
                    r["summary"][stage],
                    [("shared", "Shared draws (production)"), ("independent", "Per-variable draws")],
                    "shared_minus_independent",
                    "Shared minus per-variable",
                ),
                "",
            ]
        if "enhanced_frs" in r:
            e = r["enhanced_frs"]
            out += [
                "### Built enhanced FRS file",
                "",
                f"{e['file']} ({e['n_households']:,} households, {e['year']}), {e['weights']}.",
                "",
                table({"efrs": _flat(e["metrics"])}, [("efrs", "Enhanced FRS")], None, ""),
                "",
            ]
    syn = RESULTS / "synthetic_effect.json"
    if syn.exists():
        r = json.loads(syn.read_text())
        m = r["meta"]
        s = r["summary"]
        out += [
            "## Synthetic experiment (MOCK DATA)",
            "",
            f"MOCK DATA: {m['data']}. microimpute {m['microimpute']}; {m['reps']} replicates, "
            f"{m['n_train']:,} training and {m['n_test']:,} test households each. "
            "Differences are shared minus per-variable, standard errors across replicates in brackets.",
            "",
            table(
                s,
                [("truth", "Truth (simulated)"), ("shared", "Shared draws"), ("independent", "Per-variable draws")],
                "shared_minus_independent",
                "Shared minus per-variable",
            ),
            "",
            "### Which sharing matters (MOCK DATA)",
            "",
            "Each layout shares one draw among the listed variables and gives every other variable its own. "
            "Rows are differences from all-per-variable draws on the same forests.",
            "",
            "| Layout | Shared variables | Property below components, imputed owners with other property | All households | Donor owners with other property | Partial rank-correlation excess over truth |",
            "|---|---|---:|---:|---:|---:|",
        ]
        rows = {"shared": s["shared_minus_independent"], **r["ablations_vs_independent"]}
        for layout, d in rows.items():
            shared_vars = m["layouts"][layout]
            names = "all 11" if len(shared_vars) == 11 else ", ".join(shared_vars)
            out.append(
                f"| {layout} | {names} | {_pp(d.get('violation_share_in_imputed_group'))} | "
                f"{_pp(d.get('violation_share_all_households'))} | "
                f"{_pp(d.get('violation_share_in_donor_group'))} | "
                f"{_diff(d.get('partial_rank_excess_vs_truth'), 3)} |"
            )
        out.append("")
    sys.stdout.write("\n".join(out))


def _flat(metrics: dict) -> dict:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[f"{key}.{k}"] = v
        else:
            flat[key] = value
    return flat


if __name__ == "__main__":
    main()
