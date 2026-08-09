"""
spark_session.py
-------------------
Shared SparkSession factory.

Two things here are not boilerplate and were driven by real failures on
this machine:

1. JAVA_HOME auto-detection. This machine has a Java 8 JRE first on PATH,
   and Spark 4.x requires Java 17+ - launching Spark without overriding
   JAVA_HOME fails with:
       UnsupportedClassVersionError: org/apache/spark/launcher/Main has
       been compiled by a more recent version of the Java Runtime (class
       file version 61.0) ... only recognizes class file versions up to 52.0
   Rather than requiring every user to hand-set JAVA_HOME, this searches
   the usual install locations for a JDK 17+ and sets it automatically,
   failing with an actionable message if none is found.

2. Deliberately modest local resources. This project targets a ~6GB-RAM
   machine (see README), so the driver is capped rather than left to
   Spark's defaults.

NOTE on pandas: PySpark 4.2 warns that it does not yet fully support
pandas >= 3.0 (this environment has pandas 3.0.3). Every handoff from
Spark to the rest of the pipeline therefore goes through Parquet on disk
rather than .toPandas(), which sidesteps the unsupported conversion path
entirely. That is a real constraint, documented rather than ignored.
"""

import os
import re
import sys
from pathlib import Path

MIN_JAVA_MAJOR = 17

_SEARCH_ROOTS = [
    Path(r"C:\Program Files\Microsoft"),
    Path(r"C:\Program Files\Eclipse Adoptium"),
    Path(r"C:\Program Files\Java"),
    Path(r"C:\Program Files\Amazon Corretto"),
    Path(r"C:\Program Files\Zulu"),
]


def _java_major(dir_name: str) -> int | None:
    """Pull the major version out of a JDK directory name."""
    m = re.search(r"jdk[-_]?(\d+)", dir_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_jdk() -> Path | None:
    """Return a JDK >= MIN_JAVA_MAJOR, preferring an already-set JAVA_HOME."""
    env_home = os.environ.get("JAVA_HOME")
    if env_home:
        p = Path(env_home)
        major = _java_major(p.name)
        if (p / "bin").exists() and (major is None or major >= MIN_JAVA_MAJOR):
            return p

    candidates = []
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir() or not (child / "bin" / "java.exe").exists():
                continue
            major = _java_major(child.name)
            if major is not None and major >= MIN_JAVA_MAJOR:
                candidates.append((major, child))
    if not candidates:
        return None
    # highest major version wins; sorted() for deterministic tie-breaking
    return sorted(candidates, key=lambda t: (t[0], str(t[1])))[-1][1]


def get_spark(app_name: str = "fair-lending", shuffle_partitions: int = 8):
    jdk = find_jdk()
    if jdk is None:
        sys.exit(
            f"No JDK {MIN_JAVA_MAJOR}+ found. PySpark 4.x requires Java {MIN_JAVA_MAJOR} or newer.\n"
            f"Install one with:  winget install Microsoft.OpenJDK.17\n"
            f"(searched: {', '.join(str(r) for r in _SEARCH_ROOTS)})"
        )
    os.environ["JAVA_HOME"] = str(jdk)
    os.environ["PATH"] = f"{jdk / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    # Make the Spark workers use this same interpreter
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "false")  # off, so the perf study measures what it claims to
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


if __name__ == "__main__":
    jdk = find_jdk()
    print(f"JDK detected: {jdk}")
    spark = get_spark("jdk-check")
    print(f"Spark version: {spark.version}")
    spark.stop()
