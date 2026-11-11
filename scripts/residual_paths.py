"""Sharing that survives the per-variable seed fix (synthetic data only).

The 3.1.3 fix gives variable k of a QRF the seed ``seed + k``. Two paths still
hand identical streams to different variables:

1. separately fitted QRFs with the same base seed: variable k of each draws
   from seed + k, so same-position variables of two models predicted on the
   same rows share per-row quantiles;
2. ZeroInflatedImputer: each target's base model is a fresh single-target QRF,
   so every target's base draws come from seed + 0.

Each check fits on targets whose conditional shocks are independent, so the
true residual rank correlation is about 0. Reported over replicates.

    uv run --no-project --python 3.13 --with microimpute==3.4.1 python scripts/residual_paths.py
"""

import logging
from importlib.metadata import version

import numpy as np
import pandas as pd

from microimpute.models import QRF, ZeroInflatedImputer

logging.getLogger("microimpute").setLevel(logging.ERROR)


def frames(rng, targets, n_train=1000, n_test=600, positive=True):
    train = pd.DataFrame({"x": rng.normal(size=n_train)})
    test = pd.DataFrame({"x": rng.normal(size=n_test)})
    for t in targets:
        train[t] = np.exp(0.5 * train["x"] + rng.normal(0, 0.7, n_train))
        if not positive:
            train.loc[rng.random(n_train) < 0.4, t] = 0.0
    return train, test


def residual_rho(a, b, x):
    return float(np.corrcoef(pd.Series(a - x).rank(), pd.Series(b - x).rank())[0, 1])


def main(reps: int = 5) -> None:
    print(f"microimpute {version('microimpute')}; {reps} replicates; truth ~0 (null SE ~0.04)")
    same, diff, zi, zi_desync = [], [], [], []
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        # 1. two separately fitted default-seed QRFs on different targets
        train, test = frames(rng, ["a0", "a1", "b0", "b1"])
        first = QRF(log_level="ERROR").fit(train, ["x"], ["a0", "a1"], n_estimators=30)
        second = QRF(log_level="ERROR").fit(train, ["x"], ["b0", "b1"], n_estimators=30)
        p, q = first.predict(test), second.predict(test)
        la, lb = np.log(p), np.log(q)
        same.append(np.mean([residual_rho(la["a0"], lb["b0"], 0.5 * test["x"]),
                             residual_rho(la["a1"], lb["b1"], 0.5 * test["x"])]))
        diff.append(np.mean([residual_rho(la["a0"], lb["b1"], 0.5 * test["x"]),
                             residual_rho(la["a1"], lb["b0"], 0.5 * test["x"])]))
        # 2. ZeroInflatedImputer on three all-positive targets
        train, test = frames(rng, ["c", "d", "e"])
        model = ZeroInflatedImputer(log_level="ERROR").fit(train, ["x"], ["c", "d", "e"])
        out = np.log(model.predict(test)[["c", "d", "e"]])
        pairs = [("c", "d"), ("c", "e"), ("d", "e")]
        zi.append(np.mean([residual_rho(out[a], out[b], 0.5 * test["x"]) for a, b in pairs]))
        # same fitted model, each target's base QRF given its own generator
        for k, variable in enumerate(["c", "d", "e"]):
            for inner in model._per_variable[variable]["base"].models.values():
                inner._rng = np.random.default_rng(10_000 + k)
        out = np.log(model.predict(test)[["c", "d", "e"]])
        zi_desync.append(np.mean([residual_rho(out[a], out[b], 0.5 * test["x"]) for a, b in pairs]))

    fmt = lambda v: f"{np.mean(v):.3f} (SD {np.std(v, ddof=1):.3f})" if v else "n/a"
    print(f"separate QRFs, same variable position:      {fmt(same)}")
    print(f"separate QRFs, different variable position: {fmt(diff)}")
    print(f"ZeroInflatedImputer, all-positive targets:  {fmt(zi)}")
    print(f"ZeroInflatedImputer, base streams desynced: {fmt(zi_desync)}")


if __name__ == "__main__":
    main()
