"""
build_features.py
--------------------
PySpark feature engineering over the raw loan application data, writing a
partitioned Parquet feature table that the downstream sklearn/SHAP/
fairness steps read.

RUNS UNDER WSL, NOT WINDOWS. Spark's Hadoop write path needs winutils.exe
on Windows; reads and computation work natively there, but every write
fails. Rather than install an unofficial third-party winutils binary,
the Spark layer runs inside WSL Ubuntu where Spark works normally. See
docs/architecture.md for the full diagnosis and the evidence behind it.

    wsl -d Ubuntu -- bash -c "cd <project> && python3 spark_layer/build_features.py"

The Spark -> pandas handoff is Parquet on disk, never .toPandas():
PySpark 4.2 warns it does not fully support pandas >= 3.0 (this
environment has pandas 3.0.3), and Parquet avoids that conversion path
entirely while also being what a real lakehouse would do.

Output: database/features/ (partitioned Parquet)
        reports/spark_feature_summary.json
"""

import json
import time
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

FEATURES_PATH = DB_DIR / "features"


def get_spark(app_name="fair-lending-features"):
    spark = (
        SparkSession.builder.appName(app_name).master("local[*]")
        .config("spark.driver.memory", "1500m")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build(spark):
    apps = spark.read.csv(str(DATA_DIR / "loan_applications.csv"), header=True, inferSchema=True)
    nbh = spark.read.csv(str(DATA_DIR / "neighborhoods.csv"), header=True, inferSchema=True)

    # --- derived per-applicant features -------------------------------
    df = (apps
          .withColumn("loan_to_income", F.round(F.col("loan_amount") / F.col("annual_income"), 5))
          .withColumn("log_income", F.round(F.log("annual_income"), 5))
          .withColumn("est_monthly_payment",
                       F.round(F.col("loan_amount") / F.col("loan_term_months"), 2))
          .withColumn("payment_to_income",
                       F.round((F.col("loan_amount") / F.col("loan_term_months"))
                               / (F.col("annual_income") / 12), 5))
          .withColumn("has_delinquency", (F.col("num_delinquencies_2y") > 0).cast("int"))
          .withColumn("thin_file", (F.col("credit_history_length_years") < 2).cast("int"))
          .withColumn("application_month", F.date_format(F.col("application_date"), "yyyy-MM"))
          )

    # --- neighborhood-level aggregates via window functions -----------
    # A real feature store would compute these as point-in-time aggregates;
    # here they are whole-period, which is stated plainly rather than
    # implied to be leakage-free.
    w_nbh = Window.partitionBy("neighborhood_id")
    df = (df
          .withColumn("nbh_median_income", F.round(F.avg("annual_income").over(w_nbh), 2))
          .withColumn("nbh_avg_dti", F.round(F.avg("debt_to_income").over(w_nbh), 5))
          .withColumn("income_vs_nbh",
                       F.round(F.col("annual_income") / F.avg("annual_income").over(w_nbh), 5))
          )

    # --- broadcast join with the small neighborhoods dimension --------
    df = df.join(F.broadcast(nbh.select("neighborhood_id", "center_lat", "center_lon",
                                          "group_b_share")),
                  on="neighborhood_id", how="left")

    return df


def main():
    spark = get_spark()
    t0 = time.time()

    df = build(spark)

    # Partition by application_month: a realistic lakehouse layout, and the
    # partition column the performance study uses for predicate pushdown.
    df.write.mode("overwrite").partitionBy("application_month").parquet(str(FEATURES_PATH))
    write_secs = time.time() - t0

    written = spark.read.parquet(str(FEATURES_PATH))
    n_rows = written.count()
    n_parts = len([p for p in FEATURES_PATH.iterdir() if p.is_dir()])

    default_rate_by_group = {
        r["demographic_group"]: round(r["default_rate"], 6)
        for r in (written.groupBy("demographic_group")
                  .agg(F.avg("defaulted").alias("default_rate"))
                  .orderBy("demographic_group").collect())
    }

    summary = {
        "spark_version": spark.version,
        "n_rows": n_rows,
        "n_columns": len(written.columns),
        "n_partitions_written": n_parts,
        "write_seconds": round(write_secs, 2),
        "columns": sorted(written.columns),
        "default_rate_by_group": default_rate_by_group,
    }
    with open(REPORTS_DIR / "spark_feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Spark {spark.version}: wrote {n_rows:,} rows x {len(written.columns)} cols "
          f"into {n_parts} month partitions in {write_secs:.1f}s")
    print(f"Default rate by group: {default_rate_by_group}")
    print(f"Saved {REPORTS_DIR / 'spark_feature_summary.json'}")
    spark.stop()


if __name__ == "__main__":
    main()
