"""
performance_study.py
-----------------------
Two Spark tuning decisions, MEASURED rather than asserted:

  1. BROADCAST JOIN vs SHUFFLE JOIN
     Joining 400k applications to a 60-row dimension table. Broadcasting
     the small side should avoid a shuffle entirely. Measured by forcing
     each plan explicitly.

  2. PARTITION PRUNING vs FULL SCAN
     The feature table is partitioned by application_month. Filtering on
     that column should let Spark skip whole directories; filtering on a
     non-partition column with equivalent selectivity cannot.

Methodology notes, because timing microbenchmarks is easy to get wrong:
  * A warm-up run happens before timing to absorb JVM JIT and filesystem
    cache effects. Without it the first measurement is always the slowest
    regardless of which strategy it tests.
  * Each variant runs N_RUNS times and the MEDIAN is reported, not the
    best run - reporting the minimum flatters whichever variant got lucky.
  * spark.sql.adaptive.enabled is OFF (see get_spark), because adaptive
    query execution can silently convert a shuffle join into a broadcast
    join, which would make this comparison measure nothing.
  * This is local[*] on a 2GB WSL instance, not a cluster. The RELATIVE
    difference is the finding; the absolute seconds are not representative
    of cluster hardware and are not presented as if they were.

Output: reports/spark_performance_study.json/.md
"""

import json
import os
import statistics
import time
from pathlib import Path

from pyspark.sql import functions as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_layer.build_features import get_spark, FEATURES_PATH, DATA_DIR, REPORTS_DIR  # noqa: E402

N_RUNS = int(os.environ.get("PERF_RUNS", "5"))

# Where Spark reads the feature table from. Defaults to the project dir on
# /mnt/c, but the first run of this study showed that cross-filesystem I/O
# there produces run-to-run variance LARGER than the effects being
# measured (shuffle runs 37.3s/21.2s/22.8s for the same query). Setting
# PERF_DATA_ROOT to a WSL-native ext4 path removes that noise source.
_data_root = os.environ.get("PERF_DATA_ROOT")
FEATURES = Path(_data_root) / "features" if _data_root else FEATURES_PATH
NBH_CSV = (Path(_data_root) / "neighborhoods.csv" if _data_root
           else DATA_DIR / "neighborhoods.csv")
STORAGE_LABEL = "wsl_native_ext4" if _data_root else "windows_mnt_c"


def timed(fn) -> float:
    t = time.time()
    fn()
    return time.time() - t


def median_of(fn, n=N_RUNS) -> dict:
    fn()  # warm-up, not timed
    runs = [timed(fn) for _ in range(n)]
    return {"median_seconds": round(statistics.median(runs), 3),
            "runs": [round(r, 3) for r in runs]}


def main():
    spark = get_spark("fair-lending-perf")
    results = {}

    apps = spark.read.parquet(str(FEATURES))
    nbh = spark.read.csv(str(NBH_CSV), header=True, inferSchema=True)
    nbh_small = nbh.select("neighborhood_id", "neighborhood_risk_score")

    # ---------- 1. broadcast vs shuffle join ----------
    def shuffle_join():
        (apps.drop("neighborhood_risk_score")
         .join(nbh_small, on="neighborhood_id", how="inner")
         .agg(F.avg("neighborhood_risk_score")).collect())

    def broadcast_join():
        (apps.drop("neighborhood_risk_score")
         .join(F.broadcast(nbh_small), on="neighborhood_id", how="inner")
         .agg(F.avg("neighborhood_risk_score")).collect())

    print("Measuring join strategies...")
    results["join_strategy"] = {
        "shuffle_join": median_of(shuffle_join),
        "broadcast_join": median_of(broadcast_join),
    }

    # ---------- 2. partition pruning vs full scan ----------
    # Pick a real partition value and a non-partition filter of comparable
    # selectivity, so the comparison isn't rigged by scanning different
    # amounts of data.
    months = sorted(r["application_month"] for r in
                     apps.select("application_month").distinct().collect())
    target_month = months[len(months) // 2]
    month_rows = apps.filter(F.col("application_month") == target_month).count()
    total_rows = apps.count()

    def pruned_read():
        (spark.read.parquet(str(FEATURES))
         .filter(F.col("application_month") == target_month)
         .agg(F.avg("loan_amount")).collect())

    def full_scan_read():
        # same row count, but the filter column is NOT the partition column,
        # so every partition must be opened and read
        (spark.read.parquet(str(FEATURES))
         .filter(F.col("application_date") >= f"{target_month}-01")
         .filter(F.col("application_date") <= f"{target_month}-31")
         .agg(F.avg("loan_amount")).collect())

    print("Measuring partition pruning...")
    results["partition_pruning"] = {
        "target_month": target_month,
        "rows_in_partition": month_rows,
        "rows_total": total_rows,
        "selectivity": round(month_rows / total_rows, 4),
        "pruned_partition_filter": median_of(pruned_read),
        "full_scan_nonpartition_filter": median_of(full_scan_read),
    }

    # ---------- speedups ----------
    j = results["join_strategy"]
    p = results["partition_pruning"]
    results["summary"] = {
        "broadcast_speedup_x": round(j["shuffle_join"]["median_seconds"]
                                      / j["broadcast_join"]["median_seconds"], 2),
        "partition_pruning_speedup_x": round(
            p["full_scan_nonpartition_filter"]["median_seconds"]
            / p["pruned_partition_filter"]["median_seconds"], 2),
        "n_runs_per_variant": N_RUNS,
        "storage": STORAGE_LABEL,
        "note": "local[*] on a 2GB WSL instance; relative comparison is the finding, "
                "absolute seconds are not cluster-representative.",
    }

    out_name = f"spark_performance_study_{STORAGE_LABEL}.json"
    with open(REPORTS_DIR / out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(REPORTS_DIR / out_name.replace(".json", ".md"), "w", encoding="utf-8") as f:
        f.write("# Spark Performance Study\n\n")
        f.write(f"Median of {N_RUNS} timed runs after a warm-up run.\n\n")
        f.write("| Comparison | Variant | Median (s) | Speedup |\n|---|---|---|---|\n")
        f.write(f"| Join strategy | shuffle join | {j['shuffle_join']['median_seconds']} | - |\n")
        f.write(f"| Join strategy | broadcast join | {j['broadcast_join']['median_seconds']} | "
                f"**{results['summary']['broadcast_speedup_x']}x** |\n")
        f.write(f"| Partition pruning | non-partition filter (full scan) | "
                f"{p['full_scan_nonpartition_filter']['median_seconds']} | - |\n")
        f.write(f"| Partition pruning | partition filter (pruned) | "
                f"{p['pruned_partition_filter']['median_seconds']} | "
                f"**{results['summary']['partition_pruning_speedup_x']}x** |\n")

    print(json.dumps(results["summary"], indent=2))
    print(f"Saved {REPORTS_DIR / out_name}  [storage={STORAGE_LABEL}]")
    spark.stop()


if __name__ == "__main__":
    main()
