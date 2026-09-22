# VLS with GR00T N1.7


## Installation

Run everything from the repository root. Steps below **assume
[`README.md`](README.md) steps 1, 3, 4 and 6 are already done**: the repository is
cloned, `third_party/libero_plus` and its assets exist, you are logged in to
Hugging Face, and the VLM server is reachable. Those are shared,
environment-independent prerequisites and do not need repeating per venv.

### VLM agent
```bash
uv venv
source .venv/bin/activate
uv pip install vllm
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --max-model-len 20000 \
    --max-num-seqs 16

```

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

```

---

## Running

```bash
# using default config
python main_gr00t.py
```

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

