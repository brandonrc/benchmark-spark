# Reproducing the v2 Benchmark on a Clean DGX Spark

This guide walks a reviewer — or you on a freshly-imaged box — through
running the v2 benchmark from scratch on an NVIDIA DGX Spark.

> **Read [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) before anything else.** It
> defines the pre-registered hypotheses, the falsification rule, and the
> statistical tests. Those must not change after results are inspected.

---

## TL;DR — the 10-command happy path

```bash
# 1. Clone + bootstrap the box (installs Docker, NCT, sudoers, pulls image,
#    extracts the chroot rootfs, creates dirs, sets up a Python venv).
git clone https://github.com/brandonrc/benchmark-spark
cd benchmark-spark
git checkout v2-reproducibility-redesign
bash scripts/bootstrap_clean_spark.sh

# 2. Activate the venv the bootstrap created
source ~/bench-venv/bin/activate

# 3. Log out / back in ONCE so your docker group membership takes effect
#    (or: newgrp docker)

# 4. Paste the container_image_digest printed by bootstrap into configs/versions.lock
${EDITOR:-vi} configs/versions.lock

# 5. Download models (needs an HF token)
huggingface-cli login
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir /data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /data/models/huggingface/Qwen/Qwen2.5-72B-Instruct

# 6. Lock GPU clocks so thermal drift can't leak into the measurements
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi --lock-gpu-clocks=1950,1950

# 7. Preflight — must pass before you touch the orchestrator
scripts/preflight.sh

# 8. Dry-run the orchestrator (no execution, just writes the plan for inspection)
scripts/orchestrate.py --experiment configs/experiment.yaml \
    --out /tmp/dry --dry-run --skip-env-capture
jq '.total_runs' /tmp/dry/run_plan.json
rm -rf /tmp/dry

# 9. Start the batch (unattended — roughly 37 h for the full 220-run plan)
export BATCH_DIR="results/$(date +%F)_clean_spark"
scripts/orchestrate.py --experiment configs/experiment.yaml --out "$BATCH_DIR"

# 10. Analyze
analysis/statistics.py "$BATCH_DIR"
cat "$BATCH_DIR/statistical_analysis/results_table.md"
```

That's the short version. The rest of this document explains what each step
is doing and how to deviate safely.

---

## 0. Hardware & OS baseline

| Component | Expected |
|---|---|
| System | NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X) |
| OS | Ubuntu 24.04 LTS, kernel 6.11.x |
| Cgroup | v2 unified hierarchy (**required**, v1 will not work) |
| Disk | ≥ 300 GB free on `/data` (models + results) |
| Network | Outbound to `nvcr.io` and `huggingface.co` |

Confirm:
```bash
uname -a
cat /sys/fs/cgroup/cgroup.controllers       # must list "memory"
df -BG /data
```

## 1. Bootstrap the box

```bash
git clone https://github.com/brandonrc/benchmark-spark
cd benchmark-spark
git checkout v2-reproducibility-redesign
bash scripts/bootstrap_clean_spark.sh
```

`bootstrap_clean_spark.sh` is **idempotent** — re-running it is safe. It
will:

1. Check you're on aarch64 + Ubuntu 24.04 + nvidia-smi works.
2. `apt-get install` docker.io, nvidia-container-toolkit, jq, python3-venv.
3. Enable and start the docker daemon; configure `nvidia-ctk runtime`.
4. Add you to the `docker` group (you must log out and back in for this to
   take effect — or run `newgrp docker` in your current shell).
5. Install a **sudoers drop-in** at `/etc/sudoers.d/99-benchmark-spark-drop-caches`
   so the orchestrator can drop page caches between runs without a password.
   This is **required** — without it, runs are not independent and the
   paired bootstrap CIs will be misleading.
6. `docker pull` the pinned TensorRT-LLM image and print its digest.
7. Run `scripts/extract_container_rootfs.sh` to populate
   `$HOME/container-rootfs` — the chroot target for native runs.
8. Create the results dir and `/data/models/huggingface`.
9. Create a Python venv at `~/bench-venv` and install `requirements.txt`.

Override any path by setting an env var before running:

| Variable | Default | What it is |
|---|---|---|
| `BENCH_CONTAINER_IMAGE` | `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` | Image to pull for container runs |
| `BENCH_CONTAINER_ROOTFS` | `$HOME/container-rootfs` | Extracted chroot root for native runs |
| `BENCH_MODELS_DIR` | `/data/models/huggingface` | Where model weights live |
| `BENCH_RESULTS_DIR` | `./results` | Where orchestrator writes batch output |
| `BENCH_VENV` | `$HOME/bench-venv` | Python venv for host-side orchestration |

These same variables are read by `experiment.yaml` via `${VAR:-default}`
expansion, so whatever you set at bootstrap time is what the orchestrator
uses at batch time. **Set them in your shell rc if you want to make them
sticky.**

Debug knobs (skip expensive steps while iterating):

```bash
SKIP_APT=1           bash scripts/bootstrap_clean_spark.sh
SKIP_IMAGE_PULL=1    bash scripts/bootstrap_clean_spark.sh
SKIP_ROOTFS=1        bash scripts/bootstrap_clean_spark.sh
SKIP_VENV=1          bash scripts/bootstrap_clean_spark.sh
```

## 2. After bootstrap: three manual steps

The bootstrap script intentionally does not do three things that require
your judgement:

### 2a. Paste the container image digest into `configs/versions.lock`

After bootstrap prints the digest, open `configs/versions.lock` and set:

```
container_image_digest = "nvcr.io/nvidia/tensorrt-llm/release@sha256:<whatever-was-printed>"
```

Commit the change on a local branch so your batch is reproducible by hash.

### 2b. Download model weights (needs HF token)

```bash
source ~/bench-venv/bin/activate
huggingface-cli login

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir /data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /data/models/huggingface/Qwen/Qwen2.5-72B-Instruct \
    --local-dir-use-symlinks False
```

After the downloads finish, **record the commit SHAs** into
`configs/experiment.yaml` under each model's `huggingface_revision:` field.
That closes the last reproducibility gap.

### 2c. Lock GPU clocks

Thermal drift and auto-boost are the biggest sources of run-to-run noise on
GB10. Lock clocks to a single value:

```bash
sudo nvidia-smi --persistence-mode=1
# Find the SM clock max for your board:
nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits
# Then lock both min and max:
sudo nvidia-smi --lock-gpu-clocks=1950,1950
```

Record whatever value you picked in your batch notes. If you want to
restore auto clocks after the batch: `sudo nvidia-smi --reset-gpu-clocks`.

## 3. Preflight check

**Always run preflight before starting a batch.** It takes ~10 seconds and
catches everything that would cause a 37-hour batch to fail in the first
run.

```bash
scripts/preflight.sh
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | All checks passed — you are ready to run the batch |
| 1 | Hard failures (fix them before running) |
| 2 | Warnings only (batch will run, but some validity guarantees are relaxed) |

What it verifies:

- aarch64 + Ubuntu 24.04 + cgroup v2 + memory controller
- `nvidia-smi` works, `/data` has ≥ 300 GB free
- All required tools are on PATH (`docker`, `nvidia-container-cli`,
  `python3 ≥ 3.10`, `git`, `jq`, `huggingface-cli`)
- Host Python venv has `numpy`, `scipy`, `pyyaml`, `pandas`
- Docker daemon reachable and `docker run --gpus all` works against the
  pinned image
- The TensorRT-LLM image is pulled and its digest is recorded
- The extracted chroot rootfs is populated and has bind-mount targets
- **Passwordless `drop_caches` works** (catches the #1 bootstrap omission)
- Model weights exist at the paths referenced by `experiment.yaml`
- GPU persistence mode is on and clocks are locked
- No other processes are using the GPU
- `capture_environment.sh --verify configs/versions.lock` passes

Fix every hard failure and re-run preflight until it exits 0.

## 4. Dry-run the orchestrator

Before committing to the full 37-hour run, inspect the randomized plan.

```bash
scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out /tmp/dry \
    --dry-run --skip-env-capture

jq '.total_runs' /tmp/dry/run_plan.json
jq '.runs[0:3]' /tmp/dry/run_plan.json
jq '.runs[-3:]' /tmp/dry/run_plan.json
rm -rf /tmp/dry
```

With the default yaml you should see 220 runs (11 cells × 2 models × 10
reps). If you want a different subset, see §6 below.

## 5. Run the batch

### 5a. Full batch — 220 runs, ~37 hours

```bash
export BATCH_DIR="results/$(date +%F)_clean_spark"
scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out "$BATCH_DIR"
```

This is unattended — the orchestrator sleeps through cooldowns and
dispatches each run. Leave the machine alone: don't log in, don't run
anything else on the GPU, don't let another workload touch `/data` (the
model directory).

### 5b. Resume after interruption

Batch state lives entirely in `$BATCH_DIR`. To continue after a crash,
reboot, or network hiccup:

```bash
scripts/orchestrate.py \
    --experiment configs/experiment.yaml \
    --out "$BATCH_DIR" \
    --resume
```

`--resume` skips runs whose result JSON already exists. It refuses to start
if the on-disk `run_plan.json` doesn't match the yaml + seed — you cannot
accidentally resume a different experiment into the same directory.

### 5c. Minimum-viable subset — 40 runs, ~6.5 hours

If you only want to verify the primary H0/H1 falsifier for DeepSeek-7B
(cells E0N, E0C, E4N, E4C on one model), generate a filtered yaml:

```bash
python3 - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(open("configs/experiment.yaml"))
cfg["models"] = [m for m in cfg["models"] if m["short"] == "deepseek7b"]
cfg["cells"]  = [c for c in cfg["cells"] if c["id"] in {"E0N","E0C","E4N","E4C"}]
pathlib.Path("configs/experiment_minimal.yaml").write_text(yaml.safe_dump(cfg))
PY

scripts/orchestrate.py \
    --experiment configs/experiment_minimal.yaml \
    --out "$BATCH_DIR"
```

## 6. Analyze

```bash
analysis/statistics.py "$BATCH_DIR"
cat "$BATCH_DIR/statistical_analysis/results_table.md"
```

The report contains, per model:

- **PRIMARY** falsification verdict on E4C vs E4N (either "REJECT H0" or
  "FAIL TO REJECT H0 — consistent with probe artifact")
- Baseline replication on E0C vs E0N with paired bootstrap 95% CI
- Cohen's d (paired) and minimum detectable effect size (MDES)
- Holm-Bonferroni corrected p-values for the container config sweep
- **TOST equivalence test** on throughput with a pre-registered ±5% margin

Cross-check against `docs/VERIFICATION_CHECKLIST.md` — walk through it
before accepting the result.

## 7. What to ship in a reproduction report

Minimum artifacts a reviewer should receive:

| File | Purpose |
|---|---|
| `environment.json` | Controlled-state snapshot |
| `run_plan.json` | Immutable pre-shuffled run order |
| `*.json` (one per run) | Per-run metrics, config, lifecycle jsonl path |
| `*.lifecycle.jsonl` | 6 memory snapshots per run |
| `*.trtllm.log` | Raw TensorRT-LLM output |
| `exclusions.json` | Excluded runs + reason |
| `statistical_analysis/results_table.md` | Human-readable report |
| `statistical_analysis/results.json` | Machine-readable report |
| Your `configs/versions.lock` | **With image digest filled in** |
| Your `configs/experiment.yaml` | **With HF revisions filled in** |
| Your batch notes | Any deviations from this document |

## 8. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `preflight.sh` says "passwordless drop_caches NOT configured" | sudoers drop-in missing | re-run `bootstrap_clean_spark.sh` |
| `preflight.sh` says "capture_environment.sh --verify" fails | stack drifted from `versions.lock` | fix the stack (don't edit the lockfile) |
| `preflight.sh` says "model missing" | you forgot step 2b | run `huggingface-cli download` |
| `docker run --gpus all` fails | NCT not installed or docker not restarted | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `orchestrate.py` aborts at step "rootfs missing" | `BENCH_CONTAINER_ROOTFS` not set / rootfs not extracted | re-run bootstrap, or set the env var to match where you actually extracted it |
| Native runs fail with "chroot: cannot change root directory" | the rootfs is incomplete — the extraction crashed | delete `$BENCH_CONTAINER_ROOTFS` and re-run `scripts/extract_container_rootfs.sh` |
| Native runs fail with "/workspace/benchmarks/run_benchmark.py: No such file" | `native_runner.sh` bind mounts didn't land — probably a stale manual mount | `sudo umount` any leftover mounts under `$BENCH_CONTAINER_ROOTFS/{workspace,results,models}` and re-run |
| E4N or E4C OOM | `pinned_kv_blocks: 4500` is too large for your weights + activations | lower the value in `experiment.yaml`, making sure both cells still pass |
| Huge run-to-run variance | GPU clocks not locked, or another process on the GPU | see step 2c; kill background processes |
| Container runs fail with "Failed to initialize NVML" | NVIDIA Container Toolkit not configured | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| orchestrator hangs in cooldown loop | GPU temp never reaches the threshold | lower `max_gpu_temp_c` in `experiment.yaml`, or fix the room AC |

## 9. How long things take (observed on one GB10)

| Step | Time |
|---|---|
| `bootstrap_clean_spark.sh` | ~25 min (dominated by `docker pull` + rootfs extract) |
| Model downloads (7B + 72B) | 30–60 min (bandwidth-limited) |
| `preflight.sh` | ~15 s |
| `capture_environment.sh` | ~5 s |
| Single DeepSeek-7B W1 run | ~8 min (build + bench + snapshots + cooldown) |
| Single Qwen-72B W1 run | ~15 min |
| Single W2 (short-prompt) run | ~6 min |
| **Full 220-run batch** | ~37 h |
| **Minimal 40-run batch** | ~6.5 h |
| `analysis/statistics.py` | < 30 s |

## 10. Going off-script

If you deviate (skip cells, change N, add a model), do it in a **new** yaml
file — don't edit `configs/experiment.yaml`. Commit both the yaml and the
corresponding `run_plan.json` to your results directory so your
reproduction trail stays intact.

Every cell ID must match `TEST_PLAN.md` §7 or the analysis will silently
skip comparisons. In particular: `E4N` and `E4C` are the **primary**
falsifier pair — you cannot replace one without the other.
