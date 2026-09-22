# VLS with GR00T N1.7


## Installation

Run everything from the repository root. Steps below **assume
[`README.md`](README.md) steps 1, 3, 4 and 6 are already done**: the repository is
cloned, `third_party/libero_plus` and its assets exist, you are logged in to
Hugging Face, and the VLM server is reachable. Those are shared,
environment-independent prerequisites and do not need repeating per venv.

### 1. Create `.venv_gr00t`

```bash
export UV_PROJECT_ENVIRONMENT=.venv_gr00t
uv sync --extra libero

source .venv_gr00t/bin/activate
uv pip install  "lerobot[groot] @ git+https://github.com/huggingface/lerobot.git@5aa74557f84c54d4b458f8b9643c5aa2982acfed"
uv pip install dm-tree diffusers
```

### 2. Libero-plus installation

```bash
mkdir third_party
cd third_party
git clone https://github.com/sylvestf/LIBERO-plus.git
uv pip install -e .
bash scripts/setup_libero_plus.sh

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
# Another suite, one perturbation dimension, set perturbation_categories=null for running all perturbs
uv run python main_gr00t.py backend.libero_plus.suite_name=libero_object \
    backend.libero_plus.perturbation_categories=[camera]

# Multi-GPU: workers over disjoint task strides, merged at the end
uv run python main_gr00t.py main.gpus=[0,1,2] main.num_workers=6


# Steering off, as a baseline
uv run python main_gr00t.py main.use_guidance=false
```

The multi-GPU launcher re-invokes `main_gr00t.py` (not `main.py`) for each shard,
so workers inherit the GR00T config automatically.


```bash
uv run python main_gr00t.py main.result_dir=result_gr00t \
```

`main.result_dir` moves the CSV. The per-task rollout directory is built from
`suite_name` alone, so to keep videos apart as well, move or rename
`outputs/<suite_name>/` between runs.

### GR00T-specific policy keys

`configs/policy_gr00t.yaml`, under `policy.groot`

