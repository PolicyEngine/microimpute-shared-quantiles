"""Synthetic wealth block: what shared per-row quantiles do, and which sharing matters.

MOCK DATA: every household here is simulated. The generator is shaped loosely
like WAS round 8 (uk-data's 11 predictors and 11 targets, gross property
wealth = main residence + other residential + non-residential + land + a
non-imputed remainder, financial components with independent conditional
shocks). Its effect sizes are illustrative, not estimates for WAS or the FRS.

For each replicate it fits microimpute's QRF in uk-data's target order on one
draw, then imputes a fresh draw under several seed layouts on the same forests
(microimpute 1.8.1 reads each model's seed only at predict time):

* shared: every variable seed 42 (uk-data production);
* independent: variable i gets 42 + i;
* ablations that share one draw among a subset and keep the rest distinct.

Run with uk-data's pinned stack (microimpute 1.8.1, quantile-forest 1.4.1,
scikit-learn 1.7.2, numpy 2.1.3, pandas 2.3.3): ``run_synthetic.sh`` uses
uk-data's environment, or without it

    cd scripts && uv run --no-project --python 3.13 --with microimpute==1.8.1 \
        --with quantile-forest==1.4.1 --with scikit-learn==1.7.2 \
        --with numpy==2.1.3 --with pandas==2.3.3 python synthetic_effect.py
"""

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from microimpute.models import QRF
from scipy.stats import spearmanr

from metrics import GFW, MAIN, NFW, NONRES, OTHER, PROPERTY, summarise, wealth_metrics

logging.getLogger("microimpute").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)

PREDICTORS = [
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "num_bedrooms",
    "council_tax",
    "is_renting",
    "region",
]
TARGETS = [  # uk-data's IMPUTE_VARIABLES order
    "owned_land",
    PROPERTY,
    "corporate_wealth",
    GFW,
    NFW,
    MAIN,
    OTHER,
    NONRES,
    "savings",
    "num_vehicles",
    "student_loan_balance",
]
PROPERTY_BLOCK = [PROPERTY, MAIN, OTHER, NONRES]
FINANCIAL_BLOCK = ["corporate_wealth", GFW, NFW, "savings"]
LAYOUTS = {
    "shared": TARGETS,
    "independent": [],
    "share_property_and_components": PROPERTY_BLOCK,
    "share_property_and_other": [PROPERTY, OTHER],
    "share_components_only": [MAIN, OTHER, NONRES],
    "share_financial_only": FINANCIAL_BLOCK,
}


def simulate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    def zi_lognormal(p, mu, sigma):
        return np.where(rng.random(n) < p, np.exp(mu + rng.normal(0, sigma, n)), 0.0)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    z = rng.normal(size=n)  # latent household affluence
    adults = rng.integers(1, 4, n)
    df = pd.DataFrame(
        {
            "num_adults": adults,
            "num_children": rng.integers(0, 4, n),
            "employment_income": zi_lognormal(0.6, 10.2 + 0.5 * z, 0.6),
            "self_employment_income": zi_lognormal(0.12, 9.8 + 0.5 * z, 1.0),
            "private_pension_income": zi_lognormal(0.3, 9.0 + 0.4 * z, 0.8),
            "capital_income": zi_lognormal(sigmoid(-0.5 + 1.2 * z), 6.5 + 1.0 * z, 1.2),
            "num_bedrooms": np.clip(np.round(2.5 + 0.7 * z + rng.normal(0, 0.8, n)), 1, 6),
            "region": rng.choice([f"R{k}" for k in range(11)], n),
        }
    )
    df["household_net_income"] = 0.8 * (
        df.employment_income + df.self_employment_income + df.private_pension_income
    ) + df.capital_income + 6000
    df["council_tax"] = 900 + 350 * df.num_bedrooms + rng.normal(0, 150, n)
    owner = rng.random(n) < sigmoid(0.6 + 1.0 * z)
    df["is_renting"] = ~owner
    main = np.where(owner, np.exp(12.2 + 0.35 * z + 0.12 * df.num_bedrooms + rng.normal(0, 0.45, n)), 0.0)
    other = np.where(owner, zi_lognormal(sigmoid(-3.6 + 0.9 * z), 11.9 + 0.4 * z, 0.8), 0.0)
    nonres = zi_lognormal(sigmoid(-4.4 + 0.8 * z), 11.3 + 0.4 * z, 1.0)
    land = zi_lognormal(sigmoid(-4.6 + 0.6 * z), 11.0 + 0.3 * z, 1.1)
    rest = zi_lognormal(sigmoid(-4.0 + 0.8 * z), 11.5 + 0.3 * z, 0.9)
    df["owned_land"] = land
    df[MAIN], df[OTHER], df[NONRES] = main, other, nonres
    df[PROPERTY] = main + other + nonres + land + rest  # gross, as WAS DVPropertyR8
    df["savings"] = zi_lognormal(sigmoid(1.2 + 0.8 * z), 8.6 + 0.8 * z, 1.3)
    df["corporate_wealth"] = zi_lognormal(sigmoid(0.2 + 0.9 * z), 10.4 + 0.8 * z, 1.4)
    current = zi_lognormal(0.95, 7.5 + 0.4 * z, 1.0)
    df[GFW] = df.savings + 0.4 * df.corporate_wealth + current
    df[NFW] = df[GFW] - zi_lognormal(sigmoid(-0.6 - 0.3 * z), 8.3, 1.2)
    df["num_vehicles"] = rng.poisson(np.exp(-0.2 + 0.35 * z + 0.2 * (adults - 1)))
    df["student_loan_balance"] = zi_lognormal(sigmoid(-1.8 - 0.2 * z), 9.8, 0.6)
    df["weight"] = rng.uniform(800, 2400, n)
    return df


def residual_rank_excess(imputed, truth, test) -> float:
    """Mean |partial rank correlation| excess over truth, across target pairs.

    Ranks are residualised on household_net_income's rank to remove the shared
    signal; the excess is imputed minus true, averaged over pairs.
    """

    def partial(frame):
        income = test["household_net_income"].rank().to_numpy()
        res = {}
        for v in TARGETS:
            r = frame[v].rank().to_numpy()
            res[v] = r - np.polyval(np.polyfit(income, r, 1), income)
        out = []
        for i, a in enumerate(TARGETS):
            for b in TARGETS[i + 1 :]:
                out.append(spearmanr(res[a], res[b]).statistic)
        return np.array(out)

    return float(np.mean(partial(imputed) - partial(truth)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--out", default="results/synthetic_effect")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cells_path = out.with_suffix(".cells.jsonl")
    done = set()
    if cells_path.exists():
        done = {
            (c["split"], c["mode"])
            for c in map(json.loads, cells_path.read_text().splitlines())
        }

    for rep in range(args.reps):
        todo = [m for m in [*LAYOUTS, "truth"] if (rep, m) not in done]
        if not todo:
            continue
        rng = np.random.default_rng(rep)
        train, test = simulate(args.n, rng), simulate(args.n, rng)
        fitted = QRF().fit(train[PREDICTORS + TARGETS], PREDICTORS, TARGETS)
        truth = test[TARGETS].reset_index(drop=True)
        w = test["weight"].to_numpy()
        for mode in todo:
            if mode == "truth":
                table = truth
            else:
                shared = set(LAYOUTS[mode])
                for i, v in enumerate(TARGETS):
                    fitted.models[v].seed = 42 if v in shared else 1000 + i
                table = fitted.predict(test[PREDICTORS]).reset_index(drop=True)
            metrics = wealth_metrics(table, w, TARGETS, donor=truth)
            metrics["partial_rank_excess_vs_truth"] = residual_rank_excess(table, truth, test)
            with cells_path.open("a") as f:
                f.write(json.dumps({"split": rep, "mode": mode, "base": 0, "metrics": metrics}) + "\n")
            print(f"rep {rep} {mode} done", flush=True)

    cells = [json.loads(l) for l in cells_path.read_text().splitlines()]
    result = {
        "meta": {
            "data": "MOCK: simulated households, see simulate()",
            "microimpute": version("microimpute"),
            "reps": args.reps,
            "n_train": args.n,
            "n_test": args.n,
            "layouts": LAYOUTS,
        },
        "summary": summarise(cells, "shared", "independent"),
        "ablations_vs_independent": {
            mode: summarise(cells, mode, "independent")[f"{mode}_minus_independent"]
            for mode in LAYOUTS
            if mode not in ("shared", "independent")
        },
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["meta"], indent=1))


if __name__ == "__main__":
    main()
