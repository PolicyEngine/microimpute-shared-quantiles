#!/bin/zsh
# Runs the synthetic (MOCK data) experiment with uk-data's pinned QRF stack
# (microimpute 1.8.1, quantile-forest 1.4.1, scikit-learn 1.7.2). Resumes if rerun.
set -u
REPO=${0:A:h}
PY=${UKDATA_PYTHON:-$HOME/PolicyEngine/policyengine-uk-data/.venv/bin/python}
cd $REPO/scripts
$PY synthetic_effect.py --reps ${SYN_REPS:-8} --n ${SYN_N:-5000} --out ../results/synthetic_effect > ../results/synthetic_effect.log 2>&1
echo "finished $(date -u +%FT%TZ) exit $?" >> ../results/run_synthetic.done
