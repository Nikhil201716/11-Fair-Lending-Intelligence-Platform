#!/usr/bin/env bash
# Re-runs the Spark performance study against WSL-NATIVE ext4 storage.
#
# The first run of the study read from /mnt/c (the Windows filesystem via
# WSL's 9p bridge) and produced run-to-run variance larger than the effects
# being measured. This copies the feature table onto ext4 first so the
# measurement isn't dominated by cross-filesystem I/O.
set -e
PROJ=/mnt/c/Users/nikhi/OneDrive/Desktop/Projects/11-Fair-Lending-Intelligence-Platform
DEST=$HOME/p11_data

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$PROJ/database/features" "$DEST/"
cp "$PROJ/data/neighborhoods.csv" "$DEST/"
echo "staged on ext4: $(du -sh "$DEST" | cut -f1)"

cd "$PROJ"
export PERF_DATA_ROOT="$DEST"
export PERF_RUNS=5
python3 spark_layer/performance_study.py
