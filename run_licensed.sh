#!/bin/zsh
# Runs the two licensed-data scripts (WAS holdout, FRS counterfactual) with
# uk-data's environment. Both append per-cell results and resume if rerun.
# Prints aggregates only.
set -u
REPO=${0:A:h}
PY=${UKDATA_PYTHON:-$HOME/PolicyEngine/policyengine-uk-data/.venv/bin/python}
cd $REPO/scripts
$PY was_holdout.py --splits ${WAS_SPLITS:-3} --bases ${WAS_BASES:-4} --out ../results/was_holdout > ../results/was_holdout.log 2>&1 &
$PY frs_counterfactual.py --bases ${FRS_BASES:-4} --out ../results/frs_counterfactual > ../results/frs_counterfactual.log 2>&1 &
wait
echo "finished $(date -u +%FT%TZ)" >> ../results/run_licensed.done
