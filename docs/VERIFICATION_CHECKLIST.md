# Verification Checklist — What a Reviewer Should Eyeball Before Trusting a Batch

Before accepting `statistical_analysis/results_table.md` as a valid
reproduction of the v2 benchmark, walk through this list against the files
under `<batch_dir>/`.

## 1. Environment integrity

- [ ] `environment.json` exists and was written by `capture_environment.sh`
- [ ] `environment.json.environment_hash` matches the `environment_hash`
      field in **every** `*.json` run result
- [ ] `configs/versions.lock` in the repo at the batch's `git_sha` has a
      non-empty `container_image_digest` field
- [ ] `environment.json.container_image_digest` equals the lockfile value
- [ ] `environment.json.git_dirty_files == 0` (or, if non-zero, the uncommitted
      changes are noted in the batch's README)
- [ ] GPU clocks are locked (`nvidia_smi_query` shows
      `clocks.current.sm == clocks.max.sm`, or a value noted in batch notes)

## 2. Plan integrity

- [ ] `run_plan.json` exists and was NOT modified after `created_at`
      (check file mtime ≤ created_at + a few seconds)
- [ ] `run_plan.total_runs` equals the number of distinct `*.json` results
      + any entries in `exclusions.json`
- [ ] `run_plan.seed` is present and an integer
- [ ] Run ordinals in individual result files match `run_plan.runs[i].plan_ordinal`
- [ ] The first `warmup_reps` runs in the permutation have `warmup: true`
      in their result JSON
- [ ] No run result has `run_id` that isn't in the plan

## 3. Data completeness per run

For every non-warmup result:

- [ ] `metrics.kv_cache_bytes` is present and plausible (non-negative, < 128 GiB)
- [ ] `metrics.throughput_tokens_per_sec` is present and > 0
- [ ] `lifecycle_jsonl` file exists and contains at least 6 snapshots
      (baseline, post_dataset_build, pre_engine_build, post_engine_build,
      pre_bench, post_bench)
- [ ] `bench_ok == true` and `engine_build_ok == true`
- [ ] `trtllm_log` path exists and is non-empty

## 4. Exclusions are honest

- [ ] `exclusions.json` lists every failed run with an `exit_code` or `reason`
- [ ] No exclusions are based on a metric value (the exclusion rule
      in `TEST_PLAN.md` §10 is by run status, not by data)

## 5. Statistical analysis

- [ ] `statistical_analysis/results.json.preregistered` matches
      `TEST_PLAN.md` §10 exactly (bootstrap iterations, seed, CI level,
      falsification threshold, TOST margin, α)
- [ ] The **PRIMARY** section of the report contains an unambiguous
      verdict string: either
      `"REJECT H0 (real container overhead ≥2 GiB)"` or
      `"FAIL TO REJECT H0 (consistent with probe artifact)"`
- [ ] MDES is reported for every comparison with N < 30
- [ ] Throughput equivalence result is either `equivalent: YES` with a
      margin, or `NO / undetermined` — never "negligible" without a number

## 6. Cross-check the headline numbers against the raw data

Pick one cell × model at random. Verify by hand:

- [ ] Mean and SD of `kv_cache_bytes` computed from the per-run JSONs match
      what `results.json` reports for that cell
- [ ] The paired differences are computed against matched reps, not against
      arbitrary indices
- [ ] The bootstrap CI half-width is larger than `SD / sqrt(N)`
      (if it's smaller, something is wrong with the bootstrap)

## 7. Smell tests

- [ ] For **baseline** cells (E0C vs E0N), the 95% CI width is larger than
      the 2 GiB falsification threshold — if it's smaller, you have
      suspiciously low noise and should double-check that runs are
      independent (drop_caches actually succeeded, cooldowns were observed)
- [ ] For the **primary** cells (E4C vs E4N), the sign of the mean difference
      is reported honestly — don't let a negative mean turn into a positive
      "reduction" by flipping subtraction direction
- [ ] Throughput across all ok runs has CV < 5% (anything higher indicates
      thermal throttling or background contention)

## 8. Red flags — stop and investigate

- Any run where `lifecycle.jsonl` shows a large drop in `MemAvailable`
  between `baseline` and `pre_engine_build` **in the native case** but not
  the container case (suggests model weights are being loaded differently
  between paths — confounds the comparison)
- Any run where `nvidia_smi_pids` shows PIDs other than the inference
  process during the run window (indicates contention)
- Any run where GPU temperature at `pre_bench` > 55 °C (cooldown discipline
  was not actually enforced)
- Any cell where all 10 reps produce the **exact same** KV cache byte count
  to 6 digits (the dataset or config is deterministic in a way that hides
  real variation — may mask a bug)
