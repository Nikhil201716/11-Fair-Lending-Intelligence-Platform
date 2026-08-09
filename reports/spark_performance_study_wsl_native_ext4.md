# Spark Performance Study

Median of 5 timed runs after a warm-up run.

| Comparison | Variant | Median (s) | Speedup |
|---|---|---|---|
| Join strategy | shuffle join | 3.465 | - |
| Join strategy | broadcast join | 2.452 | **1.41x** |
| Partition pruning | non-partition filter (full scan) | 1.766 | - |
| Partition pruning | partition filter (pruned) | 1.433 | **1.23x** |
