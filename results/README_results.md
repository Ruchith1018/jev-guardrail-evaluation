# Jev study outputs
raw_results.csv   one row per (test item x run). jev = Jev's probability of attack; protectai/deepseek = baseline scores; regex = 0/1.
                  label 1 = attack, 0 = safe. suite/variant/sub identify the test and slice (e.g. obfuscated / base64 / attack).
items.jsonl       the exact text of every test item.
summary_by_group.csv, thresholds.csv, calibration_bins.csv, threeway.csv, repeat_drift.csv   computed tables.
run_info.json, errors.json, log.txt   provenance and any failures. cache/ holds every raw Jev response.
