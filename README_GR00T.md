# VLS with GR00T N1.7

This document covers running the VLS steering algorithm and its LIBERO / LIBERO-plus
evaluation with **NVIDIA GR00T N1.7** instead of π0.5.

It is a companion to [`README.md`](README.md), which is the primary document: the
benchmark setup (LIBERO-plus checkout, assets, ImageMagick), the VLM server, the
Hydra override syntax, the result CSV, the output layout and most of the
troubleshooting are **identical** and are not repeated here. Read that first, then
come back for the GR00T-specific parts.

**Nothing in the π0.5 path was modified.** `main.py`, `core/pi05_steer.py`,
`configs/config.yaml` and `configs/policy.yaml` are untouched, and `.venv` keeps
working exactly as before. GR00T lives in parallel files and in its own virtual
environment, `.venv_gr00t`.

---

## What is new

| Path | Role |
|---|---|
| `main_gr00t.py` | Entry point for GR00T. Same evaluation loop and multi-GPU launcher as `main.py`; only the policy construction differs. |
| `core/gr00t_steer.py` | `GrootPolicySteer`, the steerable GR00T N1.7 policy. Same algorithm and same public API as `core/pi05_steer.py`. |
| `configs/config_gr00t.yaml` | Top-level config for `main_gr00t.py`. Identical to `configs/config.yaml` except it selects `policy_gr00t` instead of `policy`. |
| `configs/policy_gr00t.yaml` | The `groot` policy section (checkpoint, inference steps, chunk horizon, GrootConfig overrides). |
| `.venv_gr00t/` | The GR00T virtual environment (created below; git-ignored). |

`configs/backend/` and `configs/perception.yaml` are shared with the π0.5 run, as
are `core/env_adapters/`, `core/fkd_class.py`, the keypoint and grounding modules
and `vlm_query/`.

---

## Requirements: a newer LeRobot than `.venv` has

This targets **GR00T N1.7** — `lerobot.policies.groot.groot_n1_7`, the
Cosmos-Reason2 / Qwen3-VL backbone with the `GR00TN17ActionHead` flow-matching
head.

The LeRobot pinned in `uv.lock` (the `Treeeplanter` fork, LeRobot 0.4.2) is **too
old**: it ships only `groot_n1.py` / `GR00TN15`, and has no `groot_n1_7` module at
all. Conversely, upstream LeRobot **removed N1.5 outright** when N1.7 landed —
`groot_n1.py` and `GR00TN15` are gone, and pointing the policy at an N1.5
checkpoint now raises a deliberate error telling you so. There is no version that
has both, and `core/gr00t_steer.py` has no N1.5 fallback.

So `.venv_gr00t` needs upstream `huggingface/lerobot` at a commit that contains
`src/lerobot/policies/groot/groot_n1_7.py`. The installation below pins one.

Other GR00T-only runtime requirements, none of which are in `.venv`:

- **`dm-tree`** — `GR00TN17.prepare_input` calls `tree.map_structure` behind a
  `require_package("dm-tree", extra="groot")` guard. Without it every inference
  call fails.
- **`diffusers`** — `GR00TN17ActionHead.__init__` opens with
  `require_package("diffusers", extra="groot")`.
- **A `transformers` new enough for Qwen3-VL** (`Qwen3VLForConditionalGeneration`).
  This is the hard conflict with π0.5: `.venv` pins LeRobot's `transformers` fork
  at 4.53.x, because π0.5's PaliGemma tokenizer needs a file only that fork ships.
  N1.7's backbone needs a much newer `transformers`. **The two cannot coexist in
  one environment**, which is why `.venv_gr00t` is mandatory rather than tidy.
- Flash-Attention is optional and version-coupled to torch; installing it for
  GR00T must not be able to disturb `.venv`.

---

## Installation

Run everything from the repository root. Steps below **assume
[`README.md`](README.md) steps 1, 3, 4 and 6 are already done**: the repository is
cloned, `third_party/libero_plus` and its assets exist, you are logged in to
Hugging Face, and the VLM server is reachable. Those are shared,
environment-independent prerequisites and do not need repeating per venv.

### 1. Create `.venv_gr00t`

`uv` writes to `.venv` by default, so the environment directory must be named
explicitly. `UV_PROJECT_ENVIRONMENT` does that for every uv command in the shell:

```bash
export UV_PROJECT_ENVIRONMENT=.venv_gr00t
uv sync --extra libero
```

This builds `.venv_gr00t/` from the same `uv.lock` as `.venv` — same torch, same
MuJoCo/robosuite/bddl stack — without modifying `pyproject.toml` or `uv.lock`, and
without touching `.venv`. It also installs the *old* LeRobot, which the next step
replaces.

> Keep `UV_PROJECT_ENVIRONMENT=.venv_gr00t` exported for the rest of this document.
> **Unset it before going back to π0.5 work**, or `uv run python main.py` will
> also run out of `.venv_gr00t`:
> ```bash
> unset UV_PROJECT_ENVIRONMENT   # back to .venv / π0.5
> ```

### 2. Swap in a LeRobot that has N1.7, plus the GR00T dependencies

```bash
uv pip install --python .venv_gr00t \
    "lerobot[groot] @ git+https://github.com/huggingface/lerobot.git@5aa74557f84c54d4b458f8b9643c5aa2982acfed"
uv pip install --python .venv_gr00t dm-tree diffusers
```

`5aa7455` is a commit verified to contain `groot_n1_7.py`; a later upstream commit
is fine as long as that module is still there. The `[groot]` extra is what pulls
the newer `transformers` and the rest of the GR00T stack — if it resolves them,
the second line is redundant, but it is harmless and makes the two hard
requirements explicit.

This **overwrites the `lerobot` installed by step 1** inside `.venv_gr00t` only.
`.venv` is untouched.

Verify the environment before going further:

```bash
.venv_gr00t/bin/python -c "
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17ActionHead
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.groot_n1_7 import tree
import diffusers, transformers
assert tree is not None, 'dm-tree missing'
from transformers import Qwen3VLForConditionalGeneration
print('N1.7 OK | transformers', transformers.__version__, '| diffusers', diffusers.__version__)"
```

If that prints a version line, the environment is right. An `ImportError` on
`Qwen3VLForConditionalGeneration` means `transformers` is still the π0.5-era fork.

### 3. Backbone weights

N1.7's backbone is **`nvidia/Cosmos-Reason2-2B`** (Qwen3-VL), loaded with
`trust_remote_code=True`, and it is downloaded on first use. It needs network
access and the step 4 login from `README.md`. To pre-fetch:

```bash
.venv_gr00t/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/Cosmos-Reason2-2B'))"
```

### 4. Checkpoint

`configs/policy_gr00t.yaml` ships `pretrained_path: nvidia/GR00T-N1.7-3B`, the
public base model. Unlike the N1.5 integration, N1.7 accepts **either** form and
detects which it is:

- a **raw NVIDIA N1.7 checkpoint** (the base model, or one finetuned with
  Isaac-GR00T) — its processors are built from the checkpoint's own sidecar
  assets, no conversion needed;
- a **finetuned LeRobot `GrootPolicy` checkpoint** (a directory with
  `model.safetensors` and the serialized processor configs).

**The base model is not finetuned on LIBERO.** It loads and runs, so it is fine
for a smoke test, but it will not produce meaningful success rates. For real
numbers use a LIBERO-finetuned N1.7 checkpoint and tell it which embodiment to
use:

```bash
uv run python main_gr00t.py \
    policy.groot.pretrained_path=/path/to/groot_n1_7_libero \
    policy.groot.config_overrides.embodiment_tag=libero_sim
```

`embodiment_tag` matters more than it looks. It selects which of the checkpoint's
embodiments supplies the normalization statistics and the action horizon, and
setting it to `libero_sim` additionally switches on N1.7's LIBERO gripper action
decode (`action_decode_transform: libero`, which binarizes and flips the gripper
command) and caps the execution horizon at 8. Getting it wrong is the difference
between sensible rollouts and nonsense.

**Normalization statistics.** N1.7 normally reads these from the checkpoint's
sidecar `statistics.json` for the configured embodiment tag. If the checkpoint has
none for that tag, point the run at the LeRobot dataset it was finetuned on:

```bash
uv run python main_gr00t.py \
    policy.groot.pretrained_path=/path/to/ckpt \
    policy.groot.dataset_repo_id=<org>/<libero_dataset>
```

With neither, LeRobot raises rather than silently emitting normalized `[-1, 1]`
actions.

---

## Running

Everything below mirrors the "Running" section of `README.md`; only the script and
the config name change. Smoke test on one LIBERO-plus task:

```bash
export UV_PROJECT_ENVIRONMENT=.venv_gr00t
uv run python main_gr00t.py \
    policy.groot.pretrained_path=/path/to/ckpt \
    policy.groot.config_overrides.embodiment_tag=libero_sim \
    backend.libero_plus.max_tasks=1
```

Overrides are Hydra overrides of `configs/config_gr00t.yaml`:

```bash
# Another suite, one perturbation dimension
uv run python main_gr00t.py backend.libero_plus.suite_name=libero_object \
    backend.libero_plus.perturbation_categories=[camera]

# Multi-GPU: workers over disjoint task strides, merged at the end
uv run python main_gr00t.py main.gpus=[0,1,2] main.num_workers=6

# Re-run only the tasks a previous run failed on
uv run python main_gr00t.py main.gpus=[0,1,2] main.rerun_error_dir=outputs/libero_spatial

# Steering off, as a baseline
uv run python main_gr00t.py main.use_guidance=false
```

The multi-GPU launcher re-invokes `main_gr00t.py` (not `main.py`) for each shard,
so workers inherit the GR00T config automatically.

### Results collide with the π0.5 run

`outputs/<suite_name>/` and `result/<suite_name>.csv` are **not** namespaced by
policy. A GR00T run over a suite that a π0.5 run already covered will see those
rows in the CSV and skip every task as already completed.

Keep them apart before starting:

```bash
uv run python main_gr00t.py main.result_dir=result_gr00t \
    hydra.run.dir=outputs_gr00t/libero_plus/'${now:%Y-%m-%d_%H-%M-%S}'
```

`main.result_dir` moves the CSV. The per-task rollout directory is built from
`suite_name` alone, so to keep videos apart as well, move or rename
`outputs/<suite_name>/` between runs.

### GR00T-specific policy keys

`configs/policy_gr00t.yaml`, under `policy.groot`:

| Key | Effect |
|---|---|
| `pretrained_path` | Raw NVIDIA N1.7 checkpoint or finetuned LeRobot `GrootPolicy` checkpoint. |
| `num_inference_steps` | Flow-matching denoising steps. Default 10, matching the π0.5 setup. |
| `action_chunk_horizon` | Steps of the chunk that are guided and executed. Clamped to the action head's horizon (40 for N1.7); 8 matches `libero_sim`'s execution horizon. A warning is logged if you exceed it. |
| `config_overrides` | Forwarded to `GrootPolicy.from_pretrained` as `GrootConfig` overrides. `embodiment_tag` is pre-declared; null keys are not forwarded. Add others with Hydra's append syntax, e.g. `+policy.groot.config_overrides.use_flash_attention=true`. |
| `dataset_repo_id` | Optional. LeRobot dataset whose statistics normalize state and action, when the checkpoint carries none for the configured embodiment. |

Every steering key (`guide_scale`, `use_diversity`, `diversity_scale`, `use_fkd`,
`fkd.*`, `sample_batch_size`, `sigmoid_k`, `sigmoid_x0`, `start_ratio`,
`vlm_query_limit`) lives under `main` in `configs/config_gr00t.yaml` and means
exactly what it means for π0.5.

**The values were tuned for π0.5 and are not calibrated for GR00T.** The two
policies have different action scales and different velocity magnitudes, so
`guide_scale` in particular should be re-swept before any numbers are compared.
Start by confirming `main.use_guidance=false` reproduces plain GR00T, then raise
`guide_scale` until the trajectory visibly bends toward the keypoints without the
rollout becoming unstable.

---

## How the algorithm maps onto GR00T N1.7

The steering algorithm is unchanged: diversity (RBF repulsion) on the noisy end of
sampling, keypoint-gradient guidance with a reward-adaptive sigmoid scale on the
clean end, and optional FKD particle resampling. Three things had to be adapted,
and they are the only real differences between `core/gr00t_steer.py` and
`core/pi05_steer.py`.

**1. The flow runs the other way.** π0.5 integrates from `time = 1` (noise) down
to `0` (clean) with a negative `dt`. GR00T integrates from `t = 0` (noise) up to
`1` (clean) with a positive `dt`. `gr00t_steer.py` defines
`noise_level = 1 - t`, which plays exactly the role of π0.5's `time`, so
`start_ratio` keeps its meaning across both policies and the two phases split at
the same point in the schedule.

Because `dt` flips sign, the guidance terms flip sign with it. The guidance
functions return a *reward* (negative squared distance to the keypoint), so the
sample must ascend its gradient: `v -= scale * grad` when `dt < 0` (π0.5) is
`v += scale * grad` when `dt > 0` (GR00T). The diversity potential is a sum of
inverse pairwise distances and must be *descended*, so it flips the same way. FKD's
`resampling_t_start` moves from `int(start_ratio * num_steps)` to
`int((1 - start_ratio) * num_steps)` for the same reason.

**2. The action decode is not always differentiable.** Steering needs a gradient
through "normalized model action → environment action", because the guidance
functions score an end-effector trajectory integrated from environment actions.
N1.7 ships two decode steps and picks between them based on whether the checkpoint
carries sidecar statistics:

- `GrootActionUnpackUnnormalizeStep` (`groot_action_unpack_unnormalize_v2`) is
  pure torch — gradients flow straight through, and it is used as-is.
- `GrootN17ActionDecodeStep` (`groot_n1_7_action_decode_v1`) begins with
  `action.detach().cpu().float().numpy()`, which severs the graph.

For the second, `build_grad_postprocessor` reconstructs the decode. That step is
an **affine, index-preserving** map on the action — it slices contiguous
per-group blocks, applies the min-max inverse to each, and concatenates them back
in the same order — so it is recovered exactly from two probes:
`offset = f(0)`, `scale = f(1) - f(0)`, `f(x) = x * scale + offset`. The surrogate
therefore reproduces the real decode's numbers rather than approximating them
(verified to ~1e-7, with the gradient matching the true Jacobian). It is re-probed
per call, because with relative actions the offset depends on the state cached at
pack time.

Two decode features are not affine — `clip_normalized_action` (a clamp) and the
LIBERO gripper transform (a `sign`). Neither affects steering: the guidance
gradient is only ever read on dimensions `:3` (the end-effector xyz deltas), the
clamp is a no-op away from the boundary, and the gripper transform touches only
the last dimension. **Executed actions always go through the real pipeline**; only
the gradient path uses the surrogate.

**3. N1.7 head layout.** The sampling loop mirrors
`GR00TN17ActionHead.get_action_with_features`: no `future_tokens` between the
state and action embeddings (N1.5 had them), the `AlternateVLDiT` branch that
additionally takes `image_mask` and `backbone_attention_mask`, and state carrying
a history dimension folded in by `_encode_features`.

N1.7's real-time-chunking (RTC) path is deliberately not used: steering
regenerates a whole chunk each time, so there is no previous chunk to blend
against and `vel_strength` would be all ones anyway.

Also worth knowing: guidance gradients are taken in fp32 even though the backbone
and head run under `bfloat16` autocast, because the adapter's trajectory
integration and the VLM-written reward functions are numerically far too tight for
bf16. And `main_gr00t.py` builds the processors through
`make_groot_pre_post_processors_from_pretrained` rather than the generic
`make_pre_post_processors`, because that entry point handles the raw-checkpoint
case and re-links the pack/decode step pair that relative-action decoding needs.

---

## Troubleshooting

Everything in `README.md`'s troubleshooting section applies (ImageMagick, EGL,
robosuite/NumPy 2, VLM connection errors). GR00T adds these.

**`ModuleNotFoundError: lerobot.policies.groot.groot_n1_7`**. The environment
still has the old pinned LeRobot. Run installation step 2, and check you are on
`.venv_gr00t` (`uv run python -c "import sys; print(sys.prefix)"`).

**`ImportError: cannot import name 'Qwen3VLForConditionalGeneration'`**.
`transformers` is still the π0.5-era 4.53 fork. Reinstall with the `[groot]` extra
(step 2). Do not try to fix this in `.venv` — π0.5 needs that fork.

**`GR00T model_version 'n1.7' does not match base_model_path ...`**, mentioning
N1.5. You pointed the policy at a GR00T N1.5 checkpoint. Upstream LeRobot removed
N1.5; use an N1.7 checkpoint.

**`AttributeError: 'NoneType' object has no attribute 'map_structure'`**, or a
`require_package("dm-tree")` error. `dm-tree` is not installed (step 2).

**`require_package("diffusers")` error from the action head.** Same, for
`diffusers` (step 2).

**LeRobot raises about missing statistics for the embodiment tag.** The
checkpoint has no sidecar statistics for the tag you configured. Set
`policy.groot.config_overrides.embodiment_tag` to one the checkpoint actually has,
or pass `policy.groot.dataset_repo_id`.

**Rollouts look plausible but the gripper never closes** (or is always closed) on
LIBERO. The LIBERO gripper decode is off. Set
`policy.groot.config_overrides.embodiment_tag=libero_sim`.

**`Could not override 'policy.groot.config_overrides.<key>'`**. Only
`embodiment_tag` is pre-declared. Use Hydra's append syntax for anything else:
`+policy.groot.config_overrides.<key>=<value>`.

**Every task is skipped and the run exits immediately.** The result CSV already
lists them, most likely from a π0.5 run over the same suite. Use
`main.result_dir=result_gr00t`, or `main.skip_completed=false`.

**`action_chunk_horizon=N exceeds the GR00T action head's horizon`** in the log.
The value is clamped automatically; lower it in `configs/policy_gr00t.yaml` to
silence the warning.

**Flash-Attention `undefined symbol` on import.** It was built against a different
torch. `uv pip uninstall --python .venv_gr00t flash-attn`; GR00T falls back to
eager attention and still runs.

**`uv run` uses the wrong environment.** `UV_PROJECT_ENVIRONMENT` is exported per
shell. `echo $UV_PROJECT_ENVIRONMENT` should be `.venv_gr00t` for GR00T work and
unset for π0.5 work. A stale `$VIRTUAL_ENV` from an activated venv overrides both —
run `deactivate` first.
