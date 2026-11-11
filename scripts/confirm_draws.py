"""Record each per-variable QRF model's per-row quantile draw (synthetic data only).

Fits a 4-target QRF on synthetic data whose targets have independent
conditional shocks, wraps the random generator each per-variable model uses at
predict time, and reports whether the per-row draws coincide across variables.

    uv run --no-project --with microimpute==1.8.1 python scripts/confirm_draws.py
    uv run --no-project --with microimpute==3.4.1 python scripts/confirm_draws.py
"""

import logging
from importlib.metadata import version
from unittest import mock

import numpy as np
import pandas as pd

import microimpute.models.qrf as qrf_module
from microimpute.models import QRF

logging.getLogger("microimpute").setLevel(logging.ERROR)
MI = version("microimpute")
MAJOR = int(MI.split(".")[0])


class RecordingGenerator:
    def __init__(self, gen, sink, label):
        self._gen, self._sink, self._label = gen, sink, label

    def beta(self, a, b, size=None):
        out = self._gen.beta(a, b, size=size)
        self._sink.append((self._label, np.asarray(out).copy()))
        return out

    def __getattr__(self, name):
        return getattr(self._gen, name)


def to_index(u):
    if MAJOR < 2:  # 1.x: floor onto a 10-point grid
        return np.clip((u * 10).astype(int), 0, 9)
    return np.clip(np.rint(u * 100).astype(int), 0, 100)  # 2.0.3+: 101-point grid


rng = np.random.default_rng(0)
n_train, n_test = 800, 300
targets = ["a", "b", "c", "d"]
train = pd.DataFrame({"x1": rng.normal(size=n_train), "x2": rng.normal(size=n_train)})
for t in targets:  # independent conditional shocks
    train[t] = train.x1 + rng.normal(size=n_train)
test = pd.DataFrame({"x1": rng.normal(size=n_test), "x2": rng.normal(size=n_test)})

fitted = QRF(log_level="ERROR").fit(train, ["x1", "x2"], targets, n_estimators=50)
draws = []

if MAJOR < 2:  # 1.x builds a fresh generator inside every predict call
    real = np.random.default_rng
    counter = iter(range(10_000))

    def recording_rng(seed=None):
        return RecordingGenerator(real(seed), draws, f"call{next(counter)}:seed={seed}")

    with mock.patch.object(qrf_module.np.random, "default_rng", recording_rng):
        out = fitted.predict(test)
else:  # 2.0.3+ keeps one generator per model, built at fit time
    for name, model in fitted.models.items():
        model._rng = RecordingGenerator(model._rng, draws, f"{name}:seed={model.seed}")
    out = fitted.predict(test)

seeds = {name: getattr(m, "seed", None) for name, m in fitted.models.items()}
print(f"microimpute {MI}; per-variable model seeds: {seeds}")
u = np.vstack([d for _, d in draws])
idx = np.vstack([to_index(d) for d in u])
same_u = all(np.array_equal(u[0], u[k]) for k in range(1, len(u)))
share_same_idx = np.mean([(idx[0] == idx[k]).mean() for k in range(1, len(idx))])
print(f"draw vectors identical across all variables: {same_u}")
print(f"mean share of rows with the same grid index as variable 'a': {share_same_idx:.3f}")
resid = out[targets].sub(test.x1, axis=0).corr(method="spearman")
off = resid.to_numpy()[np.triu_indices(len(targets), k=1)]
print(f"Spearman of imputed residuals across variables (truth ~0): {off.min():.3f} to {off.max():.3f}")
second = fitted.predict(test)
print("second predict() identical to first:", bool(np.allclose(out[targets].values, second[targets].values)))
