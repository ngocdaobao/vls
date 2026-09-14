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

## Abstract

Pretrained diffusion and flow-matching policies often fail under train-test distribution shifts. Rather than retraining, **VLS** performs **inference-time adaptation** by leveraging vision-language models to synthesize differentiable reward functions that steer the sampling process of pretrained policies toward satisfying test-time spatial and task requirements.

VLS introduces three steering mechanisms: **gradient-based refinement**, **RBF diversity**, and **Feynman–Kac resampling** — achieving **+31%** on CALVIN and **+13%** on LIBERO-PRO, with real-world Franka robot deployment.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The lockfile pins
CUDA 11.8 wheels, so the environment resolves for **Linux x86_64** only. uv
fetches the Python 3.12 interpreter itself — no system Python or conda needed.

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/Vision-Language-Steering/code.git
cd code
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create the Environment

```bash
uv sync --extra libero
```

This creates `.venv/` and installs every pinned version from `uv.lock`, including
`torch==2.7.1+cu118` and the [LeRobot fork](https://github.com/Treeeplanter/lerobot)
this project targets — the fork is fetched straight from git, so no separate
install step is required for it.

Drop `--extra libero` if you are not running the LIBERO-PRO benchmark. Other
optional groups:

| Command | Adds |
|---|---|
| `uv sync --extra libero` | LIBERO-PRO backend (robosuite, MuJoCo, bddl) |
| `uv sync --extra gemini` | Gemini grounding backend (`core/gemini_grounder.py`) |
| `uv sync --group dev` | pytest, IPython, jupytext |

Then either activate the venv or prefix commands with `uv run`:

```bash
source .venv/bin/activate
```

### 3. LIBERO-PRO (editable install)

LIBERO-PRO has no PyPI release and an SSH-only git remote, so it stays a manual
editable install. Run this **from the repository root**:

```bash
uv pip install --no-deps -e third_party/libero_plus -C editable_mode=compat
```

`-C editable_mode=compat` is required: the fork's `libero/` directory has no
`__init__.py`, so setuptools' `find_packages()` returns an empty list and a
standard PEP 660 editable install produces a package that cannot be imported.
Compat mode puts the source root on `sys.path` the way `setup.py develop` used
to, and `libero` then resolves as a namespace package.

Re-run this after any later `uv sync`, which prunes packages it does not manage
— or pass `uv sync --inexact` to leave it in place.

Verify the environment:
 
```bash
uv run python main.py --help
```

> **CALVIN backend:** `core/env_adapters/calvin_adapter.py` is still present but
> its dependencies (`pybullet`, `calvin_env`, `calvin_models`) are not part of
> this environment, and the adapter is imported lazily. To use it, check out the
> CALVIN submodule and install those packages yourself.

### 4. Download Model Checkpoints

Download or train your diffusion policy checkpoint and update the path in `config.yaml`:

```yaml
policy:
  pretrained_path: "/path/to/your/checkpoint/"
```

## Configuration

Main configuration is in `config.yaml`. Key sections:

### Main Settings

```yaml
main:
  episode_num: 1                    # Number of episodes to run
  instruction: "close the drawer"   # Task instruction
  use_guidance: true                # Enable steering
  guide_scale: 40.0                 # Guidance strength
  diversity_scale: 10.0             # Diversity weight for particle sampling
  sample_batch_size: 20             # Number of particles for FK steering
  action_horizon: 14                # Action sequence length
  start_step: 70                    # When to start guidance (diffusion step)
  MCMC_steps: 4                     # MCMC steps for each denoising step
```

### Environment Backend

```yaml
backend:
  backend: "calvin"  # Options: "calvin", "libero", "realworld"
```

**CALVIN-specific:**
```yaml
backend:
  calvin:
    id: "PlayTableSimEnv"
    show_gui: false               # Set true for visualization
    use_egl: true                 # EGL rendering (headless)
    vlm_camera: "static"          # Camera for VLM queries
    cubes_table_only: true        # Only spawn cubes on table
```

**LIBERO-specific:**
```yaml
backend:
  libero:
    suite_name: "libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10
    vlm_camera: "agentview"
```

### VLM Agent

```yaml
vlm_agent:
  model: "gpt-4"                    # or "gpt-4o", "claude-3.5-sonnet"
  temperature: 0.7
  max_completion_tokens: 2000
```

### Keypoint Detection

```yaml
keypoint_detector:
  num_candidates_per_mask: 5       # Keypoints per detected object
  min_dist_bt_keypoints: 0.02      # Minimum distance between keypoints
  max_mask_ratio: 0.5              # Ignore masks larger than this ratio
  bounds_min: [-1.0, -0.75, -0.1]  # Workspace bounds
  bounds_max: [0.10, 0.75, 1.2]
```

## Running the Pipeline

### Basic Usage

```bash
uv run python main.py --config config.yaml
```

### Pipeline Overview

```
1. Environment Setup
   └─> Load environment adapter (CALVIN/LIBERO/RealWorld)
   └─> Initialize observation space

2. VLM Query Stage
   └─> Capture scene image from vlm_camera
   └─> Send to VLM with task instruction
   └─> Extract guidance keypoints and stage information

3. Keypoint Detection & Tracking
   └─> Get VLM image and segmentation image from adapter
   └─> Extract keypoint candidates for each mask by clustering from DINO feature
   └─> Initialize KeypointTracker for online tracking

4. Policy Rollout Loop (each step):
   a) Get current observation from environment
   b) Update keypoint positions via tracker
   c) Compute guidance (if use_guidance=true):
      - Sample multiple action sequences (particles)
      - Transform delta_ee to 3D trajectories
      - Compute reward based on reward functions
      - FK resampling: weight and resample particles
      - Guided MCMC sampling
   d) Select best action from guided samples
   e) Execute action in environment
   f) Log trajectory and visualizations

5. Episode Termination
   └─> Save trajectory video
   └─> Save keypoint tracking video
   └─> Generate behavior heatmap
   └─> Log success metrics
```

### Key Components

**Environment Adapter** (`core/env_adapters/`):
- Unified interface across different backends
- Handles observation processing, action execution, camera access
- Each adapter implements: `reset()`, `step()`, `get_obs()`, `get_camera_image()`

**Keypoint Detector** (`core/keypoint_detector.py`):
- Grounding DINO for text-conditional object detection
- SAM for precise segmentation
- Extracts 3D keypoints from depth + segmentation masks

**Keypoint Tracker** (`core/keypoint_tracker.py`):
- Tracks keypoints across frames using optical flow
- Handles occlusion and reinitialization

**FK Steering** (`core/fkd_class.py`):
- Maintains particle swarm during diffusion sampling
- Resamples based on reward (keypoint proximity)
- Non-gradient particle filter approach

**Diffusion Policy** (`third_party/lerobot/.../modeling_diffusion_steer.py`):
- Modified diffusion policy that supports particle-based sampling
- Integrates FK steering into the denoising loop
- Returns multiple samples for reward evaluation

## Important Notes

### API Keys for VLM

Set your API key as environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### Checkpoint Compatibility

Make sure your policy checkpoint matches the observation space and action space:
- CALVIN: RGB (200x200) + Proprioception
- Action: 7-DOF delta pose + gripper

### Guidance Parameters Tuning

- `guide_scale`: Higher = stronger guidance, but may reduce diversity
- `diversity_scale`: Controls particle diversity during resampling
- `sample_batch_size`: More particles = better coverage but slower
- `start_step`: When to apply guidance in diffusion steps (0-100)
- `MCMC_steps`: More steps = better refinement but slower

Typical ranges:
- `guide_scale`: 10-100
- `diversity_scale`: 1-20
- `sample_batch_size`: 10-50
- `start_step`: 50-80

### Output Directory Structure

```
results/
└── TIMESTAMP/
    ├── episode_1/
    │   ├── vlm_agent/
    │   │   ├── query_img.png          # Scene image sent to VLM
    │   │   ├── prompt.txt              # Full prompt
    │   │   ├── output_raw.txt          # VLM response
    │   │   └── stage1_guidance.txt     # Parsed guidance
    │   ├── trajectory_*.png            # Trajectory visualization per step
    │   ├── episode_1_success.mp4       # Execution video
    │   └── keypoints_tracking.mp4      # Keypoint tracking video
    ├── episode_2/
    │   └── ...
    └── behavior_static.png             # Heatmap of end-effector positions
```

### Debugging

Enable visualizations for debugging:

```yaml
main:
  visualize_trajectory: true
  debug_draw_trajectory: true
  render: true  # Show GUI if supported
```

View logs:
```bash
tail -f results/TIMESTAMP/run.log
```

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'libero'`**
- Install it editable from the repo root, and keep the compat flag:
  `uv pip install --no-deps -e third_party/libero_pro -C editable_mode=compat`
- A plain `-e` install silently produces an unimportable package. See Installation step 3.

**Issue: `ModuleNotFoundError` for a package you know is in `uv.lock`**
- A bare `uv pip install` targets whatever `$VIRTUAL_ENV` points at, not the
  project directory. Check with `echo $VIRTUAL_ENV`, or pass `--python .venv/bin/python`.
- `uv sync` prunes anything not in the lockfile, including the editable `libero`.
  Use `uv sync --inexact` to keep it.

**Issue: `ModuleNotFoundError: No module named 'calvin_env'`**
- The CALVIN backend is not part of this environment. See the note at the end of Installation.

**Issue: VLM queries failing**
- Check API key is set: `echo $OPENAI_API_KEY`
- Check internet connection
- Try with a different model in config

**Issue: Keypoint detection finds nothing**
- Check VLM output in `results/.../vlm_agent/output_raw.txt`
- Make sure object names match what's in the scene
- Try adjusting `max_mask_ratio` in config

**Issue: Policy output is random/bad**
- Verify checkpoint path is correct
- Check if checkpoint is compatible with environment
- Try without guidance first (`use_guidance: false`)

**Issue: Slow execution**
- Reduce `sample_batch_size`
- Reduce `MCMC_steps`
- Set `visualize_trajectory: false`
- Use smaller image sizes in env config

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026vls,
  title     = {VLS: Steering Pretrained Robot Policies via Vision-Language Models},
  author    = {Shuo Liu and Ishneet Sukhvinder Singh and Yiqing Xu and Jiafei Duan and Ranjay Krishna},
  journal   = {arXiv preprint arXiv:2602.03973},
  year      = {2026}
}
```
