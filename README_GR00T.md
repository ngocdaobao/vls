# VLS with GR00T N1.7

### VLM server
Set up VLM agent in a separate venv and process
```bash
uv venv
source .venv/bin/activate
uv pip install vllm
```

Run agent
```bash
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
```

### 2. Swap in a LeRobot that has N1.7, plus the GR00T dependencies

```bash
uv pip install "lerobot[groot] @ git+https://github.com/huggingface/lerobot.git@5aa74557f84c54d4b458f8b9643c5aa2982acfed"
uv pip install dm-tree diffusers
```

## Running

Everything below mirrors the "Running" section of `README.md`; only the script and
the config name change. Smoke test on one LIBERO-plus task:

```bash
# Running all task, config in configs/config_gr00t.yaml
python main_gr00t.py 
```

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

```bash
uv run python main_gr00t.py main.result_dir=result_gr00t \
    hydra.run.dir=outputs_gr00t/libero_plus/'${now:%Y-%m-%d_%H-%M-%S}'
```

