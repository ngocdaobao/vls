# VLS: Steering Pretrained Robot Policies via Vision–Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2602.03973"><img src="https://img.shields.io/badge/arXiv-2602.03973-b31b1b.svg" alt="arXiv"></a>
  <a href="https://vision-language-steering.github.io/webpage/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
</p>

<p align="center">
  <strong>Shuo Liu</strong><sup>1,2</sup> &nbsp;
  <strong>Ishneet Sukhvinder Singh</strong><sup>3</sup> &nbsp;
  <strong>Yiqing Xu</strong><sup>2,4</sup> &nbsp;
  <strong>Jiafei Duan</strong><sup>1,2*</sup> &nbsp;
  <strong>Ranjay Krishna</strong><sup>1,2*</sup>
</p>

<p align="center">
  <sup>1</sup>University of Washington &nbsp;
  <sup>2</sup>Allen Institute for AI &nbsp;
  <sup>3</sup>University of Oxford &nbsp;
  <sup>4</sup>National University of Singapore
</p>

<p align="center"><sup>*</sup>Co-advised</p>

Pretrained diffusion and flow-matching policies often fail under train–test
distribution shift. Rather than retraining, **VLS** adapts them at inference
time: a vision–language model synthesizes differentiable reward functions that
steer the policy's sampling toward the test-time spatial and task requirements.
It combines **gradient-based refinement**, **RBF diversity** and
**Feynman–Kac resampling**.

This README covers the **LIBERO-plus** evaluation with a **π0.5** policy, which
is the configuration `configs/` ships with. The CALVIN and LIBERO-PRO adapters
are still in `core/env_adapters/` but are not set up by these instructions.

---

## Requirements

- Linux x86_64 with an NVIDIA GPU (the lockfile pins CUDA 11.8 wheels of torch 2.7.1).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, `curl`.
  uv downloads Python 3.12 itself; no conda or system Python is needed.
- **No root is needed.** On Debian/Ubuntu the setup script fetches the one system
  library it needs (ImageMagick) into the project directory.
- A Hugging Face account with access to
  [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)
  (π0.5's tokenizer, a gated repo).
- Disk: ~9 GB for LIBERO-plus assets, ~7 GB for the policy, plus the VLM weights.

## Installation

Run everything from the repository root.

### 1. Clone

```bash
git clone https://github.com/Vision-Language-Steering/code.git vls
cd vls
```

### 2. Python environment

```bash
uv sync --extra libero
```

This creates `.venv/` with the exact versions in `uv.lock`, including the
[LeRobot fork](https://github.com/Treeeplanter/lerobot) and the MuJoCo /
robosuite / bddl stack. Commands below use `uv run`; activating the venv
(`source .venv/bin/activate`) works just as well.

### 3. LIBERO-plus benchmark, assets and ImageMagick

```bash
bash scripts/setup_libero_plus.sh
```

The script is idempotent. Each step is skipped when its result is already in place:

| Step | Result |
|---|---|
| Clone [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) at the tested commit | `third_party/libero_plus/` |
| Download and unpack the asset pack (6.4 GB zip) from [`Sylvest/LIBERO-plus`](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | `third_party/libero_plus/libero/libero/assets/` |
| Fetch ImageMagick if the system has none | `.deps/imagemagick/` |
| Check that wand, robosuite, MuJoCo and bddl load in the same order `main.py` loads them | — |

Why some of this is not a plain `pip install`:

- **LIBERO-plus is not installed into the venv.** The adapter puts
  `third_party/libero_plus` on `sys.path` and pins `LIBERO_CONFIG_PATH` to
  `third_party/libero_plus/.libero/`. That way `uv sync` cannot prune it, and
  LIBERO never reads or writes `~/.libero/config.yaml`.
- **ImageMagick comes from the distribution's own packages.** LIBERO-plus imports
  `wand` at module level, and wand needs `libMagickWand`.
  - A conda-forge ImageMagick does not work here. It links newer GLib, libstdc++
    and X11 than the system copies that torch and Mesa have already loaded into
    the process, so it fails to load.
  - The script downloads Ubuntu's `libmagickwand-6.q16-6` with `apt-get download`
    (no root) and unpacks it. `patches/imagemagick.py` then points wand at it.
  - If ImageMagick is already installed system-wide, or `MAGICK_HOME` is set,
    this step is skipped.

### 4. Hugging Face login

Accept the license on the
[PaliGemma model page](https://huggingface.co/google/paligemma-3b-pt-224), then:

```bash
uv run hf auth login
```

### 5. Policy checkpoint

`configs/policy.yaml` points at
[`baongocdao/pi05_libero_py`](https://huggingface.co/baongocdao/pi05_libero_py).
This is openpi's `pi05_libero` converted to LeRobot format. It is downloaded on
first run.

To use a local copy instead, pass `policy.pi05.pretrained_path=/path/to/dir`. To
convert an openpi PyTorch checkpoint yourself:

```bash
uv run python scripts/convert_openpi_pi05_to_lerobot.py \
    --checkpoint-dir pi05_libero_py \
    --norm-stats pi05_libero/assets/physical-intelligence/libero/norm_stats.json
```

### 6. VLM server

Guidance generation, grounding and stage recognition all query an
OpenAI-compatible endpoint. By default that is
`http://localhost:8000/v1/`, serving `Qwen/Qwen3-VL-30B-A3B-Instruct`
(`configs/perception.yaml`).

vLLM is not a dependency of this project; run it from its own environment,
ideally on its own GPU:

```bash
uv venv ~/vllm-env --python 3.12
uv pip install --python ~/vllm-env vllm
CUDA_VISIBLE_DEVICES=3 ~/vllm-env/bin/vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 --max-model-len 20000 --max-num-seqs 8
```

Keep `--max-num-seqs` equal to `perception.vlm_agent.max_num_seqs`. Check that
the server is up:

```bash
curl -s http://localhost:8000/v1/models
```

To use a hosted model instead, change `base_url`, `model` and `api_key` under
`vlm_agent`, `qwen_grounding` and `qwen` in `configs/perception.yaml`.
Alternatively, set `vlm_type: gemini` and export `GOOGLE_API_KEY`.

## Running

Smoke test, one LIBERO-plus task:

```bash
uv run python main.py backend.libero_plus.max_tasks=1
```

Everything is a [Hydra](https://hydra.cc) override of `configs/config.yaml`.
To list the full config, run `uv run python main.py --help`.

```bash
# One perturbation dimension, low difficulty only, in another suite
uv run python main.py backend.libero_plus.suite_name=libero_object \
    backend.libero_plus.perturbation_categories=[camera] \
    backend.libero_plus.difficulty_levels=[1,2]

# Explicit task ids, or an inclusive id range
uv run python main.py backend.libero_plus.task_ids_filter=[0,5,42]
uv run python main.py backend.libero_plus.start_id=0 backend.libero_plus.end_id=499

# Multi-GPU: one worker process per GPU (or main.num_workers of them),
# each on a disjoint stride of the selected tasks, results merged at the end
uv run python main.py main.gpus=[0,1,2] main.num_workers=6

# Re-run only the tasks a previous run failed on (see check.py)
uv run python main.py main.gpus=[0,1,2] main.rerun_error_dir=outputs/libero_spatial

# Steering off, as a baseline
uv run python main.py main.use_guidance=false
```

Useful selection keys (`configs/backend/libero_plus.yaml`):

| Key | Meaning |
|---|---|
| `suite_name` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_mix` |
| `perturbation_categories` | `all`, or a list of `camera`, `robot`, `language`, `light`, `background`, `noise`, `layout` |
| `difficulty_levels` | `all`, or a list of levels 1–5 |
| `task_ids_filter`, `start_id`/`end_id` | Restrict to specific task indices |
| `max_tasks`, `shuffle_tasks`, `task_seed` | Cap the run, optionally sampling across the suite |
| `episodes_per_task` | Rollouts per task variant (LIBERO-plus scores 1) |

With LIBERO-plus, the episode count is (selected tasks × `episodes_per_task`).
`main.episode_num` is ignored.

### Resuming from the result CSV

Each suite has a result file, `result/<suite_name>.csv`, with columns
`Task ID,Status,Category`.

- **Before a run:** every task already listed in the file is removed from the
  selection. Only the missing tasks run; if none are missing, `main.py` exits
  without loading anything.
- **During a run:** as each task finishes, its row is appended right away. Under
  `main.gpus`, all workers append to the same file under a file lock.
- **Crashed tasks:** a task that fails in preparation or execution is not written,
  so it stays pending for the next run.
- **Multiple trials:** with `episodes_per_task > 1`, the row is written after the
  task's last trial, and the task counts as `success` if any trial succeeded.

To re-run tasks that are already listed, delete their rows, or run with
`main.skip_completed=false`. `main.result_dir` changes the directory. To rebuild
a CSV from an output directory, run
`uv run python write_result.py --suite libero_spatial`.

Steering parameters live under `main` in `configs/config.yaml`:

| Key | Effect |
|---|---|
| `use_guidance`, `guide_scale` | Enable gradient steering and set its strength |
| `use_diversity`, `diversity_scale` | RBF repulsion between particles |
| `use_fkd`, `fkd.*` | Feynman–Kac resampling |
| `sample_batch_size` | Number of particles |
| `MCMC_steps` | Refinement steps per denoising step |
| `vlm_query_limit` | Max VLM queries per episode |

## Outputs

- `outputs/<suite_name>/`: per-task rollouts, meaning videos, VLM prompts and
  responses, and `episodes.json` with success per task. Multi-GPU runs write
  `shard_<i>/` subdirectories and a merged result.
- `outputs/libero_plus/<timestamp>/`: Hydra's run directory, holding the resolved
  config and `main.log`.
- Success rates are also broken down by perturbation category and difficulty
  level at the end of the run.
- `check.py` finds tasks with a missing or empty `Task_<id>` directory, no
  `.mp4`, or an error marker. `main.rerun_error_dir` uses it.
- `write_result.py` writes `result/<suite_name>.csv`, one row per finished task
  with its status and perturbation category.

## Code map

| Path | Role |
|---|---|
| `main.py` | Entry point: builds the env adapter, perception, policy; runs episodes; multi-GPU launcher |
| `core/env_adapters/` | Backend adapters (`libero_plus_adapter.py` is the one configured) |
| `core/pi05_steer.py`, `core/diffusion_policy_steer.py` | Steerable π0.5 / diffusion policies |
| `core/fkd_class.py` | Feynman–Kac particle resampling |
| `core/keypoint_detector.py`, `core/keypoint_tracker.py` | DINO-feature keypoint proposals and tracking |
| `core/qwen_grounder.py`, `core/gemini_grounder.py` | VLM object grounding and stage recognition |
| `vlm_query/` | VLM agent and prompt templates |
| `patches/` | Environment fixes applied at import time (EGL device selection, ImageMagick) |
| `scripts/setup_libero_plus.sh` | Non-Python setup for the LIBERO-plus backend |

## Troubleshooting

**`ImportError: MagickWand shared library not found`** or
**`ImageMagick (libMagickWand) not found`**. Run `bash scripts/setup_libero_plus.sh`.
On non-Debian systems, install ImageMagick with your package manager or set
`MAGICK_HOME`. Do not point it at a conda-forge ImageMagick (see step 3).

**`LIBERO-plus checkout not found`**. Run `bash scripts/setup_libero_plus.sh`.

**`401` / `GatedRepoError` for `google/paligemma-3b-pt-224`**. Accept the
license on the model page and run `uv run hf auth login` (step 4).

**`EGL is served by a software rasterizer ... rendering runs on CPU`**. The
machine or container has no NVIDIA EGL vendor library, so MuJoCo renders with
Mesa's llvmpipe. It works, but it is slow and does not scale with GPUs.
In Docker, start the container with `NVIDIA_DRIVER_CAPABILITIES=all` (or at
least `graphics`) so the driver's `libEGL_nvidia.so` is mounted. This cannot be
fixed from inside the container.

**`OverflowError: Python integer 256 out of bounds for uint8`** (in robosuite's
`read_pixels`, every task fails during preparation). This means robosuite 1.4's
segmentation decoding ran under NumPy 2 without the fix. `main.py` applies
`patches/robosuite_numpy2.py` at startup; if you build environments from your
own script, call `robosuite_numpy2.apply()` first. Delete the failed
`Task_<id>/` directories, or use `main.rerun_error_dir`, before re-running.

**`MUJOCO_GL=... MuJoCo would not render through EGL`**. Unset `MUJOCO_GL` or
set it to `egl`.

**Connection errors from the VLM agent / grounder**. The server from step 6 is
not reachable. Check it with `curl http://localhost:8000/v1/models` and make
sure `base_url` in `configs/perception.yaml` matches.

**`ModuleNotFoundError` for a package that is in `uv.lock`**. A stale
`$VIRTUAL_ENV` from another project can redirect `uv`. Run `deactivate`, or
`unset VIRTUAL_ENV`, then `uv sync --extra libero` again.

**DINOv3 feature extractors**. `dinov3_*` values need transformers ≥ 4.56 and
gated HF access. π0.5 pins LeRobot's 4.53 transformers fork, so those values
fall back to `dinov2_vits14`, which is the configured default.

## Citation

```bibtex
@article{liu2026vls,
  title     = {VLS: Steering Pretrained Robot Policies via Vision-Language Models},
  author    = {Shuo Liu and Ishneet Sukhvinder Singh and Yiqing Xu and Jiafei Duan and Ranjay Krishna},
  journal   = {arXiv preprint arXiv:2602.03973},
  year      = {2026}
}
```
