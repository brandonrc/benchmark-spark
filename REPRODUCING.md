# Reproducing the v2 Benchmark on a Clean DGX Spark

This guide walks a reviewer through rerunning the benchmark from scratch on a
clean NVIDIA DGX Spark (GB10). It assumes you have root on the box.

> **What v2 is.** v2 is a pre-registered redesign of Phase 1/Phase 2 that
> addresses peer-review findings on the original paper. The key change is a
> **falsifying experiment** (cells E4N and E4C) that tests whether the 50.6%
> "container KV-cache reduction" reported in Phase 1 is a real memory overhead
> or an artifact of TensorRT-LLM's free-memory probe under cgroups v2.
>
> Read `docs/TEST_PLAN.md` before running anything. It defines the hypotheses,
> the pre-registered statistical tests, and the decision rules — **do not** run
> the benchmark if you intend to change those rules after seeing the results.

---

## 0. Hardware & OS baseline

Expected:

| Component | Value |
|---|---|
| System | NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X) |
| OS | Ubuntu 24.04 LTS (kernel 6.11.x) |
| Cgroup | v2 (unified hierarchy — required) |
| Disk | ≥ 300 GB free (models + results) |
| Network | Internet access for model download and image pull |

Confirm:
```bash
uname -a
cat /sys/fs/cgroup/cgroup.controllers   # must show "memory cpu ..."
df -h /data
```

## 1. Install pinned software stack

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl jq sudo \
    docker.io nvidia-container-toolkit
sudo systemctl enable --now docker

# Verify versions match configs/versions.lock
docker --version
nvidia-container-cli --version
nvidia-smi
```

If any version differs from `configs/versions.lock`, either update the lockfile
**intentionally** and note it in your batch notes, or install the exact pinned
versions. Do **not** silently run against a drifted stack.

```bash
# Python deps for analysis
python3 -m venv ~/bench-venv
source ~/bench-venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. Clone and configure

```bash
git clone https://github.com/brandonrc/benchmark-spark
cd benchmark-spark
git checkout <sha-in-paper>     # use the exact commit the paper cites
```

## 3. Pull the TensorRT-LLM container (pinned by digest)

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

# Record the image digest and write it into configs/versions.lock
docker image inspect \
  --format '{{index .RepoDigests 0}}' \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
# → nvcr.io/nvidia/tensorrt-llm/release@sha256:...
#   Put this value into container_image_digest in configs/versions.lock
```

Commit the updated lockfile to your local branch so your batch is reproducible
by hash.

## 4. Extract container rootfs for native runs

The native side runs under chroot into the exact same rootfs as the container.
This is the only way to hold Python / CUDA / TensorRT-LLM versions **actually
identical** between the two environments.

```bash
./scripts/extract_container_rootfs.sh
# Produces ~/container-rootfs and ~/container-rootfs/run_in_rootfs.sh

# Record the rootfs hash (will be captured automatically by capture_environment.sh)
find ~/container-rootfs -type f -printf '%p %s %T@\n' | sort | sha256sum
```

## 5. Download models at pinned revisions

```bash
mkdir -p /data/models/huggingface
huggingface-cli login          # needs HF token for gated models

# Pin revisions — record the SHA in configs/experiment.yaml
huggingface-cli download \
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir-use-symlinks False

huggingface-cli download \
  Qwen/Qwen2.5-72B-Instruct \
  --local-dir /data/models/huggingface/Qwen/Qwen2.5-72B-Instruct \
  --local-dir-use-symlinks False
```

After downloading, edit `configs/experiment.yaml` and set `huggingface_revision`
for each model to the commit SHA you actually pulled. This closes the last
reproducibility gap.

## 6. Capture environment

```bash
mkdir -p results/$(date +%F)_clean_spark
./scripts/capture_environment.sh \
  results/$(date +%F)_clean_spark/environment.json \
  --verify configs/versions.lock
```

This will fail loudly if any lockfile field mismatches. **Do not bypass
verification by editing the lockfile after the fact.** Fix the stack, don't
fix the check.

## 7. Lock GPU clocks (reduces thermal-variance noise)

```bash
sudo nvidia-smi --lock-gpu-clocks=1950,1950
sudo nvidia-smi --persistence-mode=1
nvidia-smi --query-gpu=clocks.current.sm,persistence_mode --format=csv
```

Record the lock value in your batch notes.

## 8. Run the batch

### 8a. Full batch (220 runs, ~37 hours)

```bash
BATCH_DIR=results/$(date +%F)_clean_spark
./scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out "$BATCH_DIR"
```

This is **interactive only in its cooldown loop**. The rest is unattended.
Leave the machine alone, don't log in, don't run anything else on the GPU.

### 8b. Minimum-viable subset (40 runs, ~6.5 hours)

If you only want to verify the primary H0/H1 falsifier for DeepSeek-7B:

```bash
# Create a filtered config
python3 - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(open("configs/experiment.yaml"))
cfg["models"] = [m for m in cfg["models"] if m["short"] == "deepseek7b"]
cfg["cells"]  = [c for c in cfg["cells"] if c["id"] in {"E0N","E0C","E4N","E4C"}]
cfg["experiment"]["runs_per_cell"] = 10
pathlib.Path("configs/experiment_minimal.yaml").write_text(yaml.safe_dump(cfg))
PY

./scripts/orchestrate.py \
    --experiment configs/experiment_minimal.yaml \
    --out "$BATCH_DIR"
```

### 8c. Dry-run (writes plan, executes nothing)

Always do this first to eyeball the plan before committing to a 37-hour batch:

```bash
./scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out /tmp/dry \
    --dry-run --skip-env-capture
cat /tmp/dry/run_plan.json | jq '.total_runs, .runs[0], .runs[-1]'
rm -rf /tmp/dry
```

### 8d. Resume after interruption

Orchestrator state lives entirely in `$BATCH_DIR`. To continue after a crash,
network hiccup, or reboot:

```bash
./scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out "$BATCH_DIR" \
    --resume
```

`--resume` skips runs whose result JSON already exists. It refuses to run if
the on-disk plan doesn't match the yaml + seed.

## 9. Analyze

```bash
./analysis/statistics.py "$BATCH_DIR"
cat "$BATCH_DIR/statistical_analysis/results_table.md"
```

The report contains:

- Primary H0/H1 verdict (E4C vs E4N)
- Baseline replication (E0C vs E0N)
- Throughput TOST equivalence test (T0C vs T0N)
- Holm-corrected sweep over E1..E6
- Cohen's d and MDES for each comparison

## 10. What to ship in your reproduction report

Minimum artifacts a reviewer should receive:

1. `environment.json` — the controlled-state snapshot
2. `run_plan.json` — the immutable pre-shuffled run order
3. `*.json` — one result file per run
4. `*.lifecycle.jsonl` — the 6 lifecycle memory snapshots per run
5. `*.trtllm.log` — the raw TensorRT-LLM output
6. `exclusions.json` — any runs excluded (with reason)
7. `statistical_analysis/results_table.md` — the final report
8. `statistical_analysis/results.json` — same, machine-readable
9. Your `configs/versions.lock` **with the container image digest filled in**
10. Your notes: any deviations from this document, ambient conditions,
    and anything you noticed during the batch.

## 11. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `capture_environment.sh` exits 3 | lockfile mismatch | fix stack, don't edit lockfile |
| `sudo -n echo 3 > /proc/sys/vm/drop_caches` warnings | orchestrator running without passwordless sudo | add a sudoers rule for `drop_caches`, or accept noisy runs |
| OOM in E4N (pinned) | `pinned_kv_blocks` in yaml too high for the model | lower it; pinned count must fit post-weight memory |
| Container runs fail with `Failed to initialize NVML` | `--gpus all` not working | reinstall nvidia-container-toolkit + restart docker |
| Huge variance in `kv_cache_bytes` | thermal drift / background process | verify GPU clocks locked, verify nothing else is running, verify cooldowns observed |
| E4C and E4N KV bytes differ by < 2 GiB | **This is the H0 verdict** — report it | don't tune, don't re-run until it changes |

## 12. How long things take (observed on one GB10)

| Step | Time |
|---|---|
| Image pull | 15–30 min |
| Rootfs extract | 3–5 min |
| Model download (7B + 72B) | 45 min (bandwidth-limited) |
| Environment capture | <5 s |
| Single DeepSeek-7B run (W1) | ~8 min (build + bench + snapshots + cooldown) |
| Single Qwen-72B run (W1) | ~15 min |
| Single W2 run (short prompts) | ~6 min |
| Full 220-run batch | ~37 h |
| Minimal 40-run batch | ~6.5 h |
| Analysis | <30 s |

## 13. Going off-script

If you need to deviate — skip cells, change N, add a model — do it in a
**new yaml file**, not by editing `experiment.yaml`. Commit both the yaml
and the corresponding `run_plan.json` to your results directory so the
reproduction trail stays intact.
