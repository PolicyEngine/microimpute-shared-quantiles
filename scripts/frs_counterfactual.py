"""FRS counterfactual: what independent draws would change in uk-data's wealth block.

Reproduces uk-data's ``impute_wealth`` on a base FRS file: fits the wealth QRF on
all of WAS round 8 (microimpute, default settings, unweighted, as
``save_imputation_models`` does), computes the predictors with policyengine-uk
(``map_to="household"``, Northern Ireland sent to Wales), and imputes twice per
predict-seed base on the same forests: shared seeds (production; base 42 is
exactly production) and independent per-variable seeds. It then applies uk-data's
``uprate_property_by_region`` rescaling, which multiplies main_residence_value
and property_wealth (not the other components) by a regional HPI factor, and
reports the metrics before and after that step.

FRS has no wealth questions, so there is no donor truth here. Shares are given
over the imputed group and over all households. The script also reports the
same metrics on a built enhanced FRS file, if one is given.

Run with uk-data's environment (microimpute 1.8.1), e.g.
    ~/PolicyEngine/policyengine-uk-data/.venv/bin/python scripts/frs_counterfactual.py
Prints aggregates only.
"""

import argparse
import json
import logging
import shutil
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from microimpute.models import QRF
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from metrics import summarise, wealth_metrics
from ukdata_source import (
    STORAGE,
    calibrate_property_to_hpi,
    load_was_table,
    ukdata_commit,
    wealth_parts,
)

logging.getLogger("microimpute").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
BASES = [42 + 1000 * k for k in range(100)]


@contextmanager
def dataset_copy(path: str):
    """Load a dataset from a temporary copy.

    policyengine-uk's loader opens the file with pd.HDFStore's default append
    mode, so loading the original would open it for writing.
    """
    with tempfile.TemporaryDirectory(prefix="frs_copy_") as folder:
        copy = Path(folder) / Path(path).name
        shutil.copy2(path, copy)
        yield UKSingleYearDataset(file_path=str(copy))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frs", default=str(STORAGE / "frs_2023_24.h5"))
    parser.add_argument("--enhanced", default=str(STORAGE / "enhanced_frs_2023_24.h5"))
    parser.add_argument("--bases", type=int, default=5)
    parser.add_argument("--out", default="results/frs_counterfactual")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = wealth_parts()
    predictors, variables = parts["PREDICTOR_VARIABLES"], parts["IMPUTE_VARIABLES"]
    was = load_was_table()
    fitted = QRF().fit(was[predictors + variables], predictors, variables)

    with dataset_copy(args.frs) as dataset:
        sim = Microsimulation(dataset=dataset)
        frame = sim.calculate_dataframe(predictors, map_to="household")
        household = pd.DataFrame(dataset.household).reset_index(drop=True)
        frs_year = int(dataset.time_period)
    frame["region"] = frame["region"].replace("NORTHERN_IRELAND", "WALES")
    weights = household["household_weight"].to_numpy(dtype=float)
    region = household["region"].to_numpy()

    cells_path = out.with_suffix(".cells.jsonl")
    cells = (
        [json.loads(l) for l in cells_path.read_text().splitlines()]
        if cells_path.exists()
        else []
    )
    done = {(c["mode"], c["base"]) for c in cells}
    for mode in ("shared", "independent"):
        for base in BASES[: args.bases]:
            if (mode, base) in done:
                continue
            for i, variable in enumerate(variables):
                fitted.models[variable].seed = base if mode == "shared" else base + i
            imputed = fitted.predict(frame).reset_index(drop=True)
            uprated = calibrate_property_to_hpi(imputed.assign(region=region))
            for stage, table in (("imputed", imputed), ("after_hpi", uprated)):
                cell = {
                    "split": stage,
                    "mode": mode,
                    "base": base,
                    "metrics": wealth_metrics(table, weights, variables),
                }
                cells.append(cell)
                with cells_path.open("a") as f:
                    f.write(json.dumps(cell) + "\n")
            print(f"{mode} base {base} done", flush=True)

    result = {
        "meta": {
            "microimpute": version("microimpute"),
            "policyengine_uk": version("policyengine-uk"),
            "ukdata_commit": ukdata_commit(),
            "frs_file": Path(args.frs).name,
            "frs_year": frs_year,
            "n_frs_households": int(len(household)),
            "bases": BASES[: args.bases],
            "model_classes": {v: type(fitted.models[v]).__name__ for v in variables},
            "weights": "base FRS household_weight (before calibration)",
        },
        "summary": {
            stage: summarise(
                [c for c in cells if c["split"] == stage], "shared", "independent"
            )
            for stage in ("imputed", "after_hpi")
        },
    }
    if args.enhanced and Path(args.enhanced).exists():
        with dataset_copy(args.enhanced) as enhanced:
            table = pd.DataFrame(enhanced.household).reset_index(drop=True)
            enhanced_year = int(enhanced.time_period)
        present = [v for v in variables if v in table.columns]
        result["enhanced_frs"] = {
            "file": Path(args.enhanced).name,
            "year": enhanced_year,
            "n_households": int(len(table)),
            "weights": "calibrated household_weight",
            "metrics": wealth_metrics(
                table, table["household_weight"].to_numpy(dtype=float), present
            ),
        }
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["meta"], indent=1))


if __name__ == "__main__":
    main()
