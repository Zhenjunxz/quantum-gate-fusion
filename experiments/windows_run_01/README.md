# Windows run 01 — 2026-11-26

First Windows measurement supplied by the team. The report identifies local source commit `130bbfd2bb3b4e14d05fb2c3128b04fa06d3ecc4`; that commit was not found in the shared repository. Do not attribute these measurements to the later ABCDE integration commit.

## Currently archived
- [timing_summary.csv](timing_summary.csv): 9 configurations, 1,620 original timings, seeds 42/43/44, 30 repeats per seed, 5 warmups.
- [cpu_summary.csv](cpu_summary.csv): 192 valid CPU counter batches, cycles and retired instructions.
- [cache_summary.csv](cache_summary.csv): 192 valid LLC counter batches.
- [SHA256SUMS.txt](SHA256SUMS.txt): checksums of all 21 original files in the submitted ZIPs, including files pending upload.

All three CSV summaries are byte-for-byte identical to the provided originals (Git blob SHA verified). Full raw CSVs, supporting seed/batch tables and original PDF are pending upload from the prepared archive. Raw ETL files and original measurement scripts were not supplied.

## Interpretation and rerun
Hardware recordings were on battery power. Several configurations exceed the report's IQR thresholds; preserve these results as the first run and archive the controlled Windows rerun separately. The early hardware pilot is exploratory, not pooled with formal data. Do not combine Windows and Jetson measurements or infer DRAM bandwidth from LLC events.
