#!/usr/bin/env bash
# Installs PySpark into the WSL user site-packages (no sudo required).
#
# Java 17 must already be present:
#   sudo apt-get install -y openjdk-17-jdk-headless
#
# NOTE: TMPDIR is redirected to real disk on purpose. WSL mounts /tmp as a
# RAM-backed tmpfs (1.4G here), and pip builds the PySpark wheel in TMPDIR -
# which fails with "[Errno 28] No space left on device" even though the root
# filesystem has hundreds of GB free.
set -e
echo "python3: $(python3 --version)"
echo "java:    $(java -version 2>&1 | head -1)"
export TMPDIR="$HOME/.tmp_build"
mkdir -p "$TMPDIR"
python3 -m pip install --user --break-system-packages --no-cache-dir --quiet pyspark
python3 -c 'import pyspark; print("pyspark", pyspark.__version__)'
rm -rf "$TMPDIR"
