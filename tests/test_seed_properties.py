"""Property-based tests for per-variable random streams in microimpute's QRF.

When several variables are imputed together, each one must draw its random
quantiles from its own stream. If two variables share a stream, every row gets
the same quantile for both, and the imputed variables come out comonotonic
whatever their dependence in the donor (PolicyEngine/microimpute#207).

Invariants, for every base seed in sklearn's uint32 range:

* per-variable seeds are pairwise distinct and stay in range;
* the per-variable generators produce pairwise different draw streams;
* the same base seed reproduces the same streams;
* end to end, targets with independent conditional shocks are not imputed
  with a shared per-row quantile.

These need microimpute 3.1.3 or later (QRF(seed=...) and _seed_for_variable).
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from microimpute.models.qrf import QRF

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
VARIABLE_NAMES = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12),
    min_size=2,
    max_size=8,
    unique=True,
)


def _qrf_streams(seed: int, variables: list, n_draws: int = 32) -> np.ndarray:
    """Draw the first quantiles each per-variable QRF model would use."""
    imputer = QRF(seed=seed)
    imputer.imputed_variables = variables
    return np.vstack(
        [
            imputer._create_model_for_variable(variable)._rng.beta(1, 1, n_draws)
            for variable in variables
        ]
    )


@given(seed=SEEDS, variables=VARIABLE_NAMES)
def test_qrf_per_variable_seeds_are_distinct_and_bounded(seed, variables) -> None:
    imputer = QRF(seed=seed)
    imputer.imputed_variables = variables
    seeds = [imputer._seed_for_variable(variable) for variable in variables]
    assert len(set(seeds)) == len(variables)
    assert all(0 <= child < 2**32 for child in seeds)


@given(seed=SEEDS, variables=VARIABLE_NAMES)
def test_qrf_per_variable_streams_differ(seed, variables) -> None:
    streams = _qrf_streams(seed, variables)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            assert not np.array_equal(streams[i], streams[j]), (
                f"{variables[i]!r} and {variables[j]!r} share a random stream"
            )


@given(seed=SEEDS, variables=VARIABLE_NAMES)
def test_qrf_streams_are_reproducible(seed, variables) -> None:
    np.testing.assert_array_equal(
        _qrf_streams(seed, variables), _qrf_streams(seed, variables)
    )


@settings(max_examples=5, deadline=None)
@given(seed=SEEDS, data_seed=st.integers(min_value=0, max_value=2**16))
def test_qrf_imputes_independent_shocks_without_shared_quantiles(
    seed, data_seed
) -> None:
    """Targets with nil conditional dependence must not come out comonotonic.

    The targets' shocks are independent given x, so the rank correlation of
    the imputed residuals should be about zero: each pair's null standard error
    is about 0.04 at 600 test rows, so a mean |correlation| of 0.15 sits more
    than three standard errors out. With a shared stream (the shared-seed
    mutant in conftest.py) the mean is 0.31 to 0.35 on microimpute 3.1.3 and
    3.4.1 in three data draws; with per-variable streams it is 0.02 to 0.06.
    """
    rng = np.random.default_rng(data_seed)
    n_train, n_test = 1000, 600
    targets = ["first", "second", "third"]
    train = pd.DataFrame({"x": rng.normal(size=n_train)})
    for target in targets:
        train[target] = train["x"] + rng.normal(size=n_train)
    test = pd.DataFrame({"x": rng.normal(size=n_test)})

    imputed = (
        QRF(seed=seed, log_level="ERROR")
        .fit(train, ["x"], targets, n_estimators=30)
        .predict(test)
    )
    residual_ranks = imputed[targets].sub(test["x"], axis=0).rank()
    correlations = residual_ranks.corr().to_numpy()[np.triu_indices(3, k=1)]
    assert np.abs(correlations).mean() < 0.15, correlations
