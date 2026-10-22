#!/bin/bash
# SAGA runs of the main table, with the arguments of the shipped runs (run.json of
# results/main_table/<dataset>/baselines/saga/ holds the command of each run).
#
# Requirements
#   SystemDS 3.2.0 binary release, https://archive.apache.org/dist/systemds/3.2.0/systemds-3.2.0-bin.tgz
#   a Java 17 runtime (the runs used OpenJDK 17.0.15)
#   the Python environment of requirements.txt
#
# Usage (from the root of the repository)
#   SYSTEMDS_HOME=/path/to/systemds-3.2.0-bin JAVA=/path/to/java bash baselines/scripts/run_saga.sh [dataset ...]
# Outputs runs/saga/results/saga/<dataset>_saga/<dataset>_saga_cleaned.csv and one record per run in
# runs/saga/records.jsonl. baselines/scripts/saga_check_outputs.py checks the tables and
# baselines/scripts/saga_finalize_records.py writes the final records (the run.json files of results/).
set -u
: "${SYSTEMDS_HOME:?}" "${JAVA:?}"
SYSTEMDS_HOME=$(cd "$SYSTEMDS_HOME" && pwd)
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/saga/logs

run() {
  ds=$1; shift
  # SystemDS runs in its own directory, so every path is absolute
  "$PY" baselines/runners/run_saga.py --dataset "$ds" --repo "$PWD" --sysds_home "$SYSTEMDS_HOME" --java "$JAVA" \
      --work_dir "$PWD/runs/saga/work" --result_root "$PWD/runs/saga/results" --records "$PWD/runs/saga/records.jsonl" \
      --topk 3 --rv 20 --max_iter 10 --timeout 36000 "$@" > "runs/saga/logs/${ds}_saga.log" 2>&1
  echo "$ds exit $?"
}

# Java heap and threads of each shipped run, beers with the multiLogReg iteration limits of its run
declare -A ARGS=(
  [adult]="--xmx 12g --threads 2"
  [beers]="--xmx 8g --threads 1 --eval_maxi 100 --eval_maxii 20"
  [breast_cancer]="--xmx 12g --threads 2"
  [smartfactory]="--xmx 12g --threads 2"
  [bike]="--xmx 12g --threads 2"
  [mercedes]="--xmx 8g --threads 1"
  [nasa]="--xmx 8g --threads 1"
  [soilmoisture]="--xmx 8g --threads 1"
  [har]="--xmx 12g --threads 1 --cluster_objective silhouette"
)
DATASETS=${*:-adult beers breast_cancer smartfactory bike mercedes nasa soilmoisture har}
for ds in $DATASETS; do
  run "$ds" ${ARGS[$ds]}
done
