"""WAS round 8 holdout: shared vs per-variable QRF draws on the same forests.

Fits uk-data's wealth QRF (microimpute, default settings, unweighted, as
``save_imputation_models`` does) on a 70% household split of WAS round 8, then
imputes the 30% holdout twice per predict-seed base:

* shared: every per-variable model gets the same seed, as in production
  (base 42 is exactly production);
* independent: variable i gets base + i, so each draws its own quantiles.

In microimpute 1.8.1 a model's seed is read at predict time to build a fresh
generator, and the forests keep the random_state they were fitted with, so the
two modes share forests and differ only in the draws. Prints aggregates only.

Run with uk-data's environment (microimpute 1.8.1), e.g.
    ~/PolicyEngine/policyengine-uk-data/.venv/bin/python scripts/was_holdout.py
"""

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from microimpute.models import QRF

from metrics import summarise, wealth_metrics
from ukdata_source import load_was_table, ukdata_commit, wealth_parts

logging.getLogger("microimpute").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
BASES = [42 + 1000 * k for k in range(100)]


def set_seeds(fitted, variables: list, base: int, mode: str) -> None:
    for i, variable in enumerate(variables):
        fitted.models[variable].seed = base if mode == "shared" else base + i


def recorded_draw_sharing(fitted, test: pd.DataFrame) -> dict:
    """Run one production predict and record every per-variable beta draw."""
    draws = []
    real = np.random.default_rng

    class Recorder:
        def __init__(self, seed):
            self._gen = real(seed)

        def beta(self, a, b, size=None):
            out = self._gen.beta(a, b, size=size)
            draws.append(np.asarray(out).copy())
            return out

        def __getattr__(self, name):
            return getattr(self._gen, name)

    with mock.patch.object(np.random, "default_rng", Recorder):
        fitted.predict(test)
    index = np.vstack([np.clip((d * 10).astype(int), 0, 9) for d in draws])
    return {
        "n_draw_vectors": len(draws),
        "share_rows_all_vectors_identical": float((index == index[0]).all(axis=0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--bases", type=int, default=5)
    parser.add_argument("--out", default="results/was_holdout")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cells_path = out.with_suffix(".cells.jsonl")
    done = set()
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            cell = json.loads(line)
            done.add((cell["split"], cell["mode"], cell["base"]))

    parts = wealth_parts()
    predictors, variables = parts["PREDICTOR_VARIABLES"], parts["IMPUTE_VARIABLES"]
    was = load_was_table().reset_index(drop=True)
    meta = {
        "microimpute": version("microimpute"),
        "quantile_forest": version("quantile-forest"),
        "scikit_learn": version("scikit-learn"),
        "ukdata_commit": ukdata_commit(),
        "n_was_households": int(len(was)),
        "splits": args.splits,
        "bases": BASES[: args.bases],
        "split_rule": "70/30 by household, np.random.default_rng(1000 + split).permutation",
    }

    for split in range(args.splits):
        todo = [
            (mode, base)
            for mode in ("shared", "independent")
            for base in BASES[: args.bases]
            if (split, mode, base) not in done
        ]
        if not todo:
            continue
        order = np.random.default_rng(1000 + split).permutation(len(was))
        cut = int(0.7 * len(was))
        train, test = was.iloc[order[:cut]], was.iloc[order[cut:]]
        fitted = QRF().fit(train[predictors + variables], predictors, variables)
        set_seeds(fitted, variables, 42, "shared")
        split_meta = {
            "split": split,
            "mode": "meta",
            "base": -1,
            "model_classes": {v: type(fitted.models[v]).__name__ for v in variables},
            "draw_sharing_production": recorded_draw_sharing(fitted, test[predictors]),
        }
        with cells_path.open("a") as f:
            f.write(json.dumps(split_meta) + "\n")
        weights = test["household_weight"].to_numpy()
        donor = test[variables].reset_index(drop=True)
        for mode, base in todo:
            set_seeds(fitted, variables, base, mode)
            imputed = fitted.predict(test[predictors]).reset_index(drop=True)
            cell = {
                "split": split,
                "mode": mode,
                "base": base,
                "metrics": wealth_metrics(imputed, weights, variables, donor=donor),
                "donor": wealth_metrics(donor, weights, variables, donor=donor),
            }
            with cells_path.open("a") as f:
                f.write(json.dumps(cell) + "\n")
            print(f"split {split} {mode} base {base} done", flush=True)

    cells = [json.loads(l) for l in cells_path.read_text().splitlines()]
    modelled = [c for c in cells if c["mode"] in ("shared", "independent")]
    donor_cells = [
        {"mode": "donor", "split": c["split"], "base": c["base"], "metrics": c["donor"]}
        for c in modelled
        if c["mode"] == "shared"
    ]
    meta["per_split"] = [c for c in cells if c["mode"] == "meta"]
    result = {
        "meta": meta,
        "summary": summarise(modelled + donor_cells, "shared", "independent"),
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
