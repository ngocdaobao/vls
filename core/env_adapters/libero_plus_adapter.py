#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import sys
import json
import random
import yaml
import einops
import importlib.util
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


# import lerobot 
from lerobot.processor.pipeline import PolicyProcessorPipeline, ProcessorStep
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

# Import LIBERO-PLUS benchmark
# __file__ is in core/env_adapters/, so need .parent.parent.parent to get project root
LIBERO_PLUS_PATH = Path(__file__).parent.parent.parent / "third_party" / "libero_plus"
if not (LIBERO_PLUS_PATH / "libero" / "libero").is_dir():
    raise ImportError(
        f"LIBERO-plus checkout not found at {LIBERO_PLUS_PATH}. "
        "Run `bash scripts/setup_libero_plus.sh` from the repository root."
    )
if str(LIBERO_PLUS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS_PATH))

# Pin LIBERO's path config to this repo's libero_plus tree. Without this, LIBERO reads
# the machine-global ~/.libero/config.yaml, which is written once by whichever LIBERO
# checkout was imported first and may point at an unrelated one that lacks the
# LIBERO-plus perturbation bddl_files/init_files.
_LIBERO_CONFIG_DIR = LIBERO_PLUS_PATH / ".libero"
_LIBERO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_CONFIG_DIR)
_LIBERO_ROOT = LIBERO_PLUS_PATH / "libero" / "libero"
_LIBERO_CONFIG_FILE = _LIBERO_CONFIG_DIR / "config.yaml"
if not _LIBERO_CONFIG_FILE.exists():
    with _LIBERO_CONFIG_FILE.open("w") as _f:
        yaml.safe_dump(
            {
                "benchmark_root": str(_LIBERO_ROOT),
                "bddl_files": str(_LIBERO_ROOT / "bddl_files"),
                "init_states": str(_LIBERO_ROOT / "init_files"),
                "datasets": str(_LIBERO_ROOT.parent / "datasets"),
                "assets": str(_LIBERO_ROOT / "assets"),
            },
            _f,
        )

# LIBERO-plus's env wrapper imports wand, which needs libMagickWand at import time.
from patches import imagemagick
imagemagick.apply()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# vls
from .base_adapter import BaseEnvAdapter, Pose3D, CameraParams, TrackedObject, InteractableObject


# Import logging utility
from utils.logging_utils import SteerLogger
# Create logger instance
log = SteerLogger("LiberoAdapter")



def _convert_nested_dict(d, add_batch_dim: bool = True):
    """Convert nested dict with numpy arrays to torch tensors.
    
    Args:
        d: Nested dict with numpy arrays
        add_batch_dim: If True, add batch dimension to tensors (B=1)
    """
    result = {} 
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_nested_dict(v, add_batch_dim)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if add_batch_dim:
                t = t.unsqueeze(0)  # Add batch dim: (D,) -> (1, D)
            result[k] = t
        else:
            result[k] = v
    return result



# ============================================================
# LIBERO-plus perturbation task selection
# ============================================================
# LIBERO-plus ships its perturbations baked into the benchmark itself: each of
# libero_spatial / libero_object / libero_goal / libero_10 is expanded from 10
# tasks to ~2.5k perturbed variants (10030 in total), and the mapping from a
# variant to its perturbation category / difficulty level lives in
# benchmark/task_classification.json. Nothing has to be generated at runtime
# (unlike LIBERO-PRO) -- evaluation only has to pick which variants to roll out,
# with a single trial per task.

_TASK_CLASSIFICATION_FILE = _LIBERO_ROOT / "benchmark" / "task_classification.json"

# Short config aliases -> category strings used in task_classification.json
PERTURBATION_CATEGORIES: dict[str, str] = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}
_CATEGORY_ALIASES = {name.lower(): name for name in PERTURBATION_CATEGORIES.values()}
_CATEGORY_ALIASES.update(PERTURBATION_CATEGORIES)


@lru_cache(maxsize=None)
def _load_task_classification() -> dict[str, dict[str, dict]]:
    """Return suite_name -> task_name -> {"category", "difficulty_level"}."""
    if not _TASK_CLASSIFICATION_FILE.exists():
        log.warning(
            f"task_classification.json not found at {_TASK_CLASSIFICATION_FILE}; "
            "perturbation filtering is unavailable"
        )
        return {}
    with _TASK_CLASSIFICATION_FILE.open() as f:
        raw = json.load(f)
    return {
        suite: {
            entry["name"]: {
                "category": entry.get("category"),
                "difficulty_level": entry.get("difficulty_level"),
            }
            for entry in entries
        }
        for suite, entries in raw.items()
    }

 
def _normalize_categories(categories: Any) -> list[str] | None:
    """Resolve config category aliases to task_classification.json category names."""
    if categories is None or categories == "" or categories == "all":
        return None
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    resolved = []
    for category in categories:
        key = str(category).strip().lower()
        if key not in _CATEGORY_ALIASES:
            raise ValueError(
                f"Unknown perturbation category '{category}'. "
                f"Available: {', '.join(sorted(PERTURBATION_CATEGORIES))}"
            )
        resolved.append(_CATEGORY_ALIASES[key])
    return resolved


def _normalize_levels(levels: Any) -> list[int] | None:
    """Resolve a difficulty level config value to a list of ints."""
    if levels is None or levels == "" or levels == "all":
        return None
    if isinstance(levels, int):
        levels = [levels]
    elif isinstance(levels, str):
        levels = [lvl.strip() for lvl in levels.split(",") if lvl.strip()]
    return [int(lvl) for lvl in levels]


def select_perturbation_tasks(
    suite: Any,
    suite_name: str,
    categories: Any = None,
    difficulty_levels: Any = None,
    task_ids: Optional[Sequence[int]] = None,
    max_tasks: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[list[int], list[dict]]:
    """
    Pick which LIBERO-plus task variants to evaluate.

    Args:
        suite: Instantiated LIBERO-plus benchmark suite.
        suite_name: Name of that suite (used to look up its classification).
        categories: Perturbation categories to keep (aliases or full names),
            None/"all" keeps every category.
        difficulty_levels: Difficulty levels to keep, None/"all" keeps all.
        task_ids: Restrict the candidate pool to these task indices.
        max_tasks: Keep at most this many tasks after filtering.
        shuffle: Shuffle the selection before applying max_tasks, so that a
            capped run samples across the suite instead of its first tasks.
        seed: Seed for that shuffle.

    Returns:
        (task_indices, metadata) where metadata[i] describes task_indices[i].
    """
    classification = _load_task_classification().get(suite_name, {})
    categories = _normalize_categories(categories)
    difficulty_levels = _normalize_levels(difficulty_levels)
    filtering = categories is not None or difficulty_levels is not None

    if filtering and not classification:
        raise ValueError(
            f"Cannot filter suite '{suite_name}' by perturbation: no classification entries. "
            f"Expected them in {_TASK_CLASSIFICATION_FILE}."
        )

    if task_ids:
        candidates = [int(t) for t in task_ids]
        for tid in candidates:
            if not 0 <= tid < len(suite.tasks):
                raise ValueError(
                    f"task_ids entry {tid} is out of range for suite "
                    f"'{suite_name}' ({len(suite.tasks)} tasks)"
                )
    else:
        candidates = list(range(len(suite.tasks)))

    selected: list[int] = []
    meta: list[dict] = []
    unclassified = 0
    for tid in candidates:
        task = suite.tasks[tid]
        info = classification.get(task.name)
        if info is None:
            unclassified += 1
            # An unclassified task is an unperturbed original; it can only be
            # kept when no perturbation filter was requested.
            if filtering:
                continue
            info = {"category": None, "difficulty_level": None}
        if categories is not None and info["category"] not in categories:
            continue
        if difficulty_levels is not None and info["difficulty_level"] not in difficulty_levels:
            continue
        selected.append(tid)
        # `task.language` is derived from the file name and carries the
        # perturbation suffix as text, so it is deliberately not recorded here.
        # The instruction comes from the BDDL the env loads (LiberoPlusEnv).
        meta.append(
            {
                "task_id": tid,
                "task_name": task.name,
                "category": info["category"],
                "difficulty_level": info["difficulty_level"],
            }
        )

    if not selected:
        raise ValueError(
            f"No tasks selected for suite '{suite_name}' (categories={categories}, "
            f"difficulty_levels={difficulty_levels}, task_ids={task_ids})."
        )

    if shuffle:
        order = list(range(len(selected)))
        random.Random(seed).shuffle(order)
        selected = [selected[i] for i in order]
        meta = [meta[i] for i in order]

    if max_tasks is not None and int(max_tasks) > 0:
        selected = selected[: int(max_tasks)]
        meta = meta[: int(max_tasks)]

    if unclassified:
        log.info(f"Suite '{suite_name}': {unclassified} task(s) have no perturbation classification")
    log.info(
        f"Selected {len(selected)}/{len(suite.tasks)} LIBERO-plus tasks from '{suite_name}': "
        f"{dict(Counter(m['category'] for m in meta))}"
    )
    return selected, meta

def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    # LIBERO-plus variants do not each ship their own init file: a perturbed
    # task reuses the init states of the task it was derived from, and the
    # suite knows how to strip the perturbation suffix (_view_/_table_/_light_/
    # ...) to find it. Always go through the suite instead of joining the path
    # by hand.
    #
    # The init files are pickled numpy arrays, which torch>=2.6 refuses under its
    # default torch.load(weights_only=True), and LIBERO-plus calls torch.load
    # without the flag. Allow full unpickling for this call only; the files come
    # from the pinned LIBERO-plus checkout.
    original_load = torch.load
    torch.load = partial(original_load, weights_only=False)
    try:
        return task_suite.get_task_init_states(i)
    finally:
        torch.load = original_load


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


OBS_STATE_DIM = 8
ACTION_DIM = 7
AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

SPATIAL_TASK_MAX_STEPS = 280 #280,  # longest training demo has 193 steps
OBJECT_TASK_MAX_STEPS = 280 #280,  # longest training demo has 254 steps
GOAL_TASK_MAX_STEPS = 300
TEN_TASK_MAX_STEPS = 520 #505,  # longest training demo has 505 steps
NINETY_TASK_MAX_STEPS = 400

TASK_SUITE_MAX_STEPS: dict[str, int] = {
    # Original LIBERO suites
    "libero_spatial": SPATIAL_TASK_MAX_STEPS, 
    "libero_object": OBJECT_TASK_MAX_STEPS, # 280,  # longest training demo has 254 steps
    "libero_goal": GOAL_TASK_MAX_STEPS,  # longest training demo has 270 steps
    "libero_10": TEN_TASK_MAX_STEPS,  # longest training demo has 505 steps
    "libero_90": NINETY_TASK_MAX_STEPS,  # longest training demo has 373 steps
    # LIBERO-plus keeps the original suite names: the perturbations live in the
    # task variants inside each suite, not in separate suites.
    "libero_mix": TEN_TASK_MAX_STEPS,
}


class LiberoPlusEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        camera_name: str | Sequence[str] = "agentview_image, robot0_eye_in_hand_image",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
    ):
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",

            }
        self.camera_name_mapping = camera_name_mapping
        
        # libero_10 tasks need more stabilization steps due to more complex scenes
        if "libero_10" in task_suite_name.lower():
            self.num_steps_wait = max(num_steps_wait, 20)  # At least 20 steps for libero_10
            log.info(f"Using {self.num_steps_wait} stabilization steps for libero_10 suite")
        else:
            self.num_steps_wait = num_steps_wait
        
        self.episode_index = episode_index
        # Load once and keep
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        # The camera/robot perturbations are encoded in the task name rather than
        # in a file of their own (`..._view_<angles>_initstate_<n>[_noise_<n>]`);
        # LIBERO-plus's env wrapper parses those out and loads the base BDDL, so
        # the full variant name is what has to be handed to it.
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
            "camera_depths": True
        }
        env = OffScreenRenderEnv(**env_args)

        # Take the instruction from the BDDL the env actually loaded. For a
        # language-perturbed variant that is the rewritten instruction, and for
        # every other variant it is the original one; `task.language` is derived
        # from the file name instead and carries the perturbation suffix as text.
        self.task_description = getattr(env, "language_instruction", None) or task.language

        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert raw LIBERO observation to policy-compatible format."""
        observation = {}
        
        # Process images: camera_name -> mapped_name, convert to (B, C, H, W) float32 [0,1]
        for cam_name in self.camera_name:
            img = raw_obs[cam_name]
            mapped_name = self.camera_name_mapping[cam_name]
            
            # numpy (H, W, C) uint8 -> torch (1, C, H, W) float32
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W, C)
            img_tensor = img_tensor.permute(0, 3, 1, 2).contiguous().float() / 255.0  # (1, C, H, W)
            
            observation[f"{OBS_IMAGES}.{mapped_name}"] = img_tensor
        
        # Robot state
        observation[f"{OBS_STR}.robot_state"] = _convert_nested_dict({
            "eef": {
                "pos": raw_obs.get("robot0_eef_pos"),
                "quat": raw_obs.get("robot0_eef_quat"),
                "mat": self._env.robots[0].controller.ee_ori_mat,
            },
            "gripper": {
                "qpos": raw_obs.get("robot0_gripper_qpos"),
                "qvel": raw_obs.get("robot0_gripper_qvel"),
            },
            "joints": {
                "pos": raw_obs.get("robot0_joint_pos"),
                "vel": raw_obs.get("robot0_joint_vel"),
            },
        })

        
        
        return observation

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._env.seed(seed)
        if self.init_states and self._init_states is not None:
            self._env.set_init_state(self._init_states[self._init_state_id])
        raw_obs = self._env.reset()

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "success": bool(is_success),
            }
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


class LazyEnvSequence(Sequence):
    """
    Sequence of LIBERO-plus envs built on first access.

    A LIBERO-plus suite holds thousands of task variants, so instantiating a
    MuJoCo env for every one of them up front is not viable (LIBERO-PRO could
    get away with it at 10 tasks per suite). Only `cache_size` envs are kept
    alive at a time; the least recently used one is closed when that budget is
    exceeded. Evaluation walks the tasks in order, so a cache of 1 is enough to
    never rebuild an env twice in a row.
    """

    def __init__(
        self,
        factory: Callable[[int], "LiberoPlusEnv"],
        task_ids: Sequence[int],
        cache_size: int = 1,
    ):
        self._factory = factory
        self._task_ids = list(task_ids)
        self._cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[int, LiberoPlusEnv]" = OrderedDict()

    @property
    def task_ids(self) -> list[int]:
        return list(self._task_ids)

    def __len__(self) -> int:
        return len(self._task_ids)

    def __getitem__(self, index: int) -> "LiberoPlusEnv":
        if isinstance(index, slice):
            raise TypeError("LazyEnvSequence does not support slicing")
        if index < 0:
            index += len(self._task_ids)
        if not 0 <= index < len(self._task_ids):
            raise IndexError(index)

        env = self._cache.get(index)
        if env is not None:
            self._cache.move_to_end(index)
            return env

        env = self._factory(self._task_ids[index])
        self._cache[index] = env
        while len(self._cache) > self._cache_size:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception as e:  # closing is best effort, never fail a rollout over it
                log.warning(f"Failed to close cached LIBERO-plus env: {e}")
        return env

    def close(self):
        for env in self._cache.values():
            try:
                env.close()
            except Exception as e:
                log.warning(f"Failed to close cached LIBERO-plus env: {e}")
        self._cache.clear()


def create_libero_plus_envs(env_config: dict) -> tuple[LazyEnvSequence, list[dict]]:
    """
    Build the lazy env sequence for a LIBERO-plus perturbation evaluation.

    The suite names are the plain LIBERO ones (libero_spatial, libero_object,
    libero_goal, libero_10, libero_mix) -- LIBERO-plus expands each of them with
    its perturbed task variants instead of adding perturbation-specific suites.
    Which variants get rolled out is controlled by `perturbation_categories`,
    `difficulty_levels`, `task_ids_filter` and `max_tasks` in the backend config.

    Returns:
        (envs, task_metadata) with one metadata entry per env, in the same order.
    """
    suite_name = str(env_config["suite_name"]).strip()
    camera_names = _parse_camera_names(
        env_config.get("camera_name", "agentview_image,robot0_eye_in_hand_image")
    )

    suite = _get_suite(suite_name)
    log.info(f"Creating LIBERO-plus envs | suite={suite_name} | {len(suite.tasks)} task variants")

    start_task_id = env_config.get("start_id")
    end_task_id = env_config.get("end_id")
    if start_task_id is not None and end_task_id is not None:
        env_config["task_ids_filter"] = list(range(int(start_task_id), int(end_task_id) + 1))

    task_ids, task_metadata = select_perturbation_tasks(
        suite,
        suite_name,
        categories=env_config.get("perturbation_categories"),
        difficulty_levels=env_config.get("difficulty_levels"),
        task_ids=env_config.get("task_ids_filter"),
        max_tasks=env_config.get("max_tasks"),
        shuffle=env_config.get("shuffle_tasks", False),
        seed=env_config.get("task_seed", 0),
    )

    # Data-parallel evaluation: every worker performs the identical selection
    # above and then keeps only its own stride of it, so the workers cover
    # disjoint tasks (hence disjoint episodes) without talking to each other.
    # Striding (rather than contiguous chunking) keeps the perturbation
    # categories, which are grouped in task order, balanced across workers.
    shard_count = int(env_config.get("shard_count") or 1)
    shard_index = int(env_config.get("shard_index") or 0)
    if shard_count > 1:
        if not 0 <= shard_index < shard_count:
            raise ValueError(f"shard_index={shard_index} out of range for shard_count={shard_count}")
        total_before = len(task_ids)
        task_ids = task_ids[shard_index::shard_count]
        task_metadata = task_metadata[shard_index::shard_count]
        if not task_ids:
            raise ValueError(
                f"Shard {shard_index}/{shard_count} got no tasks: only {total_before} task(s) "
                f"selected. Use fewer workers."
            )
        log.info(
            f"Shard {shard_index}/{shard_count}: {len(task_ids)}/{total_before} tasks "
            f"(task_ids {task_ids[:5]}{'...' if len(task_ids) > 5 else ''})"
        )

    def factory(task_id: int) -> LiberoPlusEnv:
        task = suite.tasks[task_id]
        log.info(f"Building env | suite={suite_name} | task_id={task_id} | {task.name}")
        return LiberoPlusEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            obs_type=env_config.get("obs_type", "pixels_agent_pos"),
            render_mode=env_config.get("render_mode", "rgb_array"),
            observation_width=env_config.get("observation_width", 256),
            observation_height=env_config.get("observation_height", 256),
            visualization_width=env_config.get("visualization_width", 640),
            visualization_height=env_config.get("visualization_height", 640),
            init_states=env_config.get("init_states", True),
            episode_index=0,
            num_steps_wait=env_config.get("num_steps_wait", 10),
        )

    envs = LazyEnvSequence(factory, task_ids, cache_size=env_config.get("env_cache_size", 1))
    return envs, task_metadata


class LiberoPlusAdapter(BaseEnvAdapter):
    def __init__(self, env: None, env_config: dict, device: str = "cuda"):
        super().__init__(env, env_config, device)

        self._env, self.task_metadata = create_libero_plus_envs(env_config)

        # LIBERO-plus related attributes
        self.suite_name = env_config["suite_name"]
        self.task_num = len(self._env)
        self.current_task_idx = 0  # Start with first task

        # LIBERO-plus is evaluated with a single trial per task (its tasks are
        # already the perturbation samples), so the episode budget follows from
        # the task selection rather than from main.episode_num. Raise
        # `episodes_per_task` only to average over several rollouts per variant.
        self.episodes_per_task = max(1, int(env_config.get("episodes_per_task", 1)))
        self.total_episodes_num = self.task_num * self.episodes_per_task
        requested = env_config.get("episode_num")
        if requested is not None and int(requested) != self.total_episodes_num:
            log.info(
                f"Ignoring main.episode_num={requested}: running {self.total_episodes_num} episodes "
                f"({self.task_num} tasks x {self.episodes_per_task} trial(s) per task). "
                f"Use backend.libero_plus.max_tasks to shorten the run."
            )
        self.current_episode_idx = -1  # Will be incremented to 0 on first reset

        # Cache for robot state and observations
        self._last_obs = None
        self._robot_state = None

        env_preprocessor_steps: list[ProcessorStep] = []
        env_postprocessor_steps: list[ProcessorStep] = []
        env_preprocessor_steps.append(LiberoProcessorStep())
        self.env_preprocessor = PolicyProcessorPipeline(steps=env_preprocessor_steps)
        self.env_postprocessor = PolicyProcessorPipeline(steps=env_postprocessor_steps)


    def get_vlm_image(self) -> np.ndarray:
        """
        Get the current VLM image in (H, W, C) uint8 format for visualization.
        
        Returns image WITH flipud for correct human-viewable orientation.
        Uses visualization_width/height for higher resolution.
        """
        robosuite_env = self._get_current_robosuite_env()
        if robosuite_env is None:
            return None
        
        sim = robosuite_env.sim
        camera_name = self.vlm_camera or 'agentview'
        
        # Render at visualization resolution (e.g., 640x640)
        width = self._env_config.get('visualization_width', 640)
        height = self._env_config.get('visualization_height', 640)
        
        rgb = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False
        )
        
        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # Flip for correct human-viewable orientation
        rgb = np.flipud(rgb).copy()
        return rgb

    def get_vlm_image_raw(self) -> np.ndarray:
        """
        Get raw VLM image WITHOUT flipping for projection calculations.
        Used internally by draw_action_trajectory_on_vlm_image.
        Uses visualization_width/height for higher resolution.
        """
        robosuite_env = self._get_current_robosuite_env()
        if robosuite_env is None:
            return None
        
        sim = robosuite_env.sim
        camera_name = self.vlm_camera or 'agentview'
        
        # Render at visualization resolution (e.g., 640x640)
        width = self._env_config.get('visualization_width', 640)
        height = self._env_config.get('visualization_height', 640)
        
        rgb = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False
        )
        
        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # NO FLIP - return raw MuJoCo image for projection
        return rgb

    def get_policy_observation(self, sample_num: int = 1) -> Dict[str, torch.Tensor]:
        """
        Get observation in format expected by policy.
        
        Args:
            sample_num: Number of samples to expand batch dimension
        
        Returns:
            Dict with policy-expected keys (images already (B,C,H,W), state tensors)
        """
        if self._last_obs is None:
            return {}
        
        obs = self._last_obs.copy()
        
        # Task description needs to be replicated for each sample in batch
        task_desc = self._env[self.current_task_idx].task_description
        obs["task"] = [task_desc] * sample_num

        # Run preprocessor first (creates state tensor from robot_state)
        obs = self.env_preprocessor(obs)
        
        # Expand batch dimension for multi-sample inference if needed
        # This must happen AFTER env_preprocessor since it creates new tensors
        if sample_num > 1:
            for key, val in obs.items():
                if isinstance(val, torch.Tensor) and val.shape[0] == 1:
                    obs[key] = val.expand(sample_num, *val.shape[1:])

        return obs

    def get_task_description(self) -> str:
        """Get the task description for the current task."""
        return self._env[self.current_task_idx].task_description

    # ==================== Core Robot State ====================
    
    def _get_current_robosuite_env(self):
        """Get the current robosuite environment."""
        return self._env[self.current_task_idx]._env
    
    def _get_raw_obs(self) -> dict:
        """Get raw observations from robosuite environment."""
        robosuite_env = self._get_current_robosuite_env()
        # Step with zero action to get fresh observation
        action = np.zeros(7)
        raw_obs, _, _, _ = robosuite_env.step(action)
        # Reset back to maintain state
        # Note: This is a workaround since robosuite doesn't expose observations directly
        return raw_obs
    
    def get_ee_pose(self) -> Pose3D:
        """
        Get end-effector pose in robot base frame.
        For LIBERO, robot base is at world origin, so this is the same as world pose.
        """
        robosuite_env = self._get_current_robosuite_env()
        robot = robosuite_env.robots[0]
        controller = robot.controller
        
        pos = controller.ee_pos.copy()
        ori_mat = controller.ee_ori_mat.copy()
        
        # Convert rotation matrix to quaternion (wxyz format)
        from scipy.spatial.transform import Rotation as R
        rot = R.from_matrix(ori_mat)
        quat_xyzw = rot.as_quat()  # scipy returns xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return Pose3D(position=pos, quaternion=quat_wxyz)

    def get_ee_pose_world(self) -> Pose3D:
        """
        Get end-effector pose in world frame.
        For LIBERO, robot base is at world origin.
        """
        return self.get_ee_pose()
    
    def get_robot_base_pose(self) -> Pose3D:
        """
        Get robot base pose in world frame.
        For LIBERO, the robot base is at the world origin.
        """
        return Pose3D(
            position=np.array([0.0, 0.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (wxyz)
        )
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.
        
        Returns:
            (7,) array of joint positions in radians
        """
        robosuite_env = self._get_current_robosuite_env()
        robot = robosuite_env.robots[0]
        return robot._joint_positions.copy()
    
    def get_gripper_state(self) -> float:
        """
        Get current gripper state.
        
        Returns:
            Gripper opening width (0 = closed, positive = open)
        """
        robosuite_env = self._get_current_robosuite_env()
        robot = robosuite_env.robots[0]
        # Get gripper joints from simulation
        gripper = robot.gripper
        # Get gripper joint positions (typically has left_finger and right_finger)
        gripper_joint_ids = [robosuite_env.sim.model.joint_name2id(joint) for joint in gripper.joints]
        gripper_qpos = [robosuite_env.sim.data.qpos[jid] for jid in gripper_joint_ids]
        # Return sum as opening width (both fingers contribute)
        return sum(gripper_qpos)
    
    # ==================== Camera & Perception ====================
    
    def get_camera_params(self, camera_name: str) -> CameraParams:
        """
        Get camera intrinsic and extrinsic parameters from MuJoCo.
        
        Args:
            camera_name: Name of the camera ('agentview', 'robot0_eye_in_hand', etc.)
        
        Returns:
            CameraParams object with intrinsic, extrinsic, width, height
        """
        robosuite_env = self._get_current_robosuite_env()
        sim = robosuite_env.sim
        model = sim.model
        
        # Get camera ID from name
        try:
            cam_id = model.camera_name2id(camera_name)
        except:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model")
        
        # Get camera parameters from MuJoCo
        fovy_deg = model.cam_fovy[cam_id]  # FOV in degrees
        cam_pos = model.cam_pos[cam_id].copy()  # Camera position in world frame
        cam_quat_wxyz = model.cam_quat[cam_id].copy()  # Camera orientation (wxyz)
        
        # Use visualization resolution for VLM camera, observation resolution for others
        vlm_camera = self._env_config.get('vlm_camera', 'agentview')
        if camera_name == vlm_camera:
            width = self._env_config.get('visualization_width', 640)
            height = self._env_config.get('visualization_height', 640)
        else:
            width = self._env_config.get('observation_width', 256)
            height = self._env_config.get('observation_height', 256)
        
        # Compute intrinsic matrix from FOV
        fovy_rad = np.radians(fovy_deg)
        focal_length = (height / 2.0) / np.tan(fovy_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0
        
        intrinsic = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Compute extrinsic matrix for vis_utils.py (world-to-camera)
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        quat_xyzw = np.array([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw)
        R_cam_axes = rot.as_matrix()  # Camera axes in world frame
        
        # IMPORTANT: MuJoCo camera Z-axis points BACKWARD (opposite of OpenCV)
        # We need to flip Z-axis to match OpenCV convention (Z forward)
        R_cam_axes[:, 2] = -R_cam_axes[:, 2]
        
        # Build cam2world transform
        cam2world = np.eye(4, dtype=np.float32)
        cam2world[:3, :3] = R_cam_axes
        cam2world[:3, 3] = cam_pos
        
        # vis_utils.py and BaseAdapter.project_3d_to_2d() expect world2cam
        # So we return world2cam (inverse of cam2world)
        world2cam = np.linalg.inv(cam2world)

        extrinsic = world2cam
        
        return CameraParams(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=width,
            height=height
        )
    
    def get_camera_names(self) -> List[str]:
        """Get list of available camera names."""
        robosuite_env = self._get_current_robosuite_env()
        # Get camera names from the environment's camera configuration
        libero_env = self._env[self.current_task_idx]
        return list(libero_env.camera_name)
    
    # ==================== Scene Objects ====================
    
    def get_interactable_objects(self) -> List[InteractableObject]:
        """
        Get list of interactable/touchable objects in LIBERO scene.
        
        Filters out robot components and returns only task-relevant objects.
        """
        robosuite_env = self._get_current_robosuite_env()
        sim = robosuite_env.sim
        
        interactables = []
        segment_index = 1  # Start from 1 (0 reserved for background)
        
        # Method 1: Scan through all MuJoCo bodies and find task objects
        robot_keywords = ['panda', 'gripper', 'robot', 'mount', 'link', 'finger', 'eef', 'hand']
        env_keywords = ['world', 'floor', 'table', 'wall', 'arena']
        
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id)
            body_name_lower = body_name.lower()
            
            # Skip robot components
            if any(kw in body_name_lower for kw in robot_keywords):
                continue
            
            # Skip environment/arena
            if any(kw in body_name_lower for kw in env_keywords):
                continue
            
            # This is likely a task object!
            interactables.append(InteractableObject(
                name=body_name,
                object_id=body_id,
                link_index=-1,
                segment_index=segment_index
            ))
            segment_index += 1
        
        # Method 2: If no objects found, try mujoco_objects attribute
        if len(interactables) == 0 and hasattr(robosuite_env, 'env') and hasattr(robosuite_env.env, 'model'):
            if hasattr(robosuite_env.env.model, 'mujoco_objects'):
                for obj_name, mujoco_obj in robosuite_env.env.model.mujoco_objects.items():
                    try:
                        body_id = sim.model.body_name2id(mujoco_obj.root_body)
                        interactables.append(InteractableObject(
                            name=obj_name,
                            object_id=body_id,
                            link_index=-1,
                            segment_index=segment_index
                        ))
                        segment_index += 1
                    except:
                        pass
        
        return interactables
    
    def get_scene_objects(self) -> List[TrackedObject]:
        """
        Get trackable objects in LIBERO scene with their world positions.
        """
        interactables = self.get_interactable_objects()
        robosuite_env = self._get_current_robosuite_env()
        sim = robosuite_env.sim
        
        objects = []
        for interactable in interactables:
            # Try to get object position from simulation
            # LIBERO objects are typically named like "object_name_1"
            try:
                # Get body ID from object name in simulation
                body_id = sim.model.body_name2id(interactable.name)
                body_pos = sim.data.body_xpos[body_id].copy()
                body_quat_wxyz = sim.data.body_xquat[body_id].copy()  # MuJoCo uses wxyz
                
                pose = Pose3D(position=body_pos, quaternion=body_quat_wxyz)
                
                objects.append(TrackedObject(
                    name=interactable.name,
                    pose=pose,
                    obj_ref={
                        'object_id': interactable.object_id,
                        'segment_index': interactable.segment_index
                    }
                ))
            except:
                # If object not found in sim, skip it
                continue
        
        return objects
    
    def get_object_pose(self, object_name: str) -> Optional[Pose3D]:
        """Get pose of a specific object by name."""
        objects = self.get_scene_objects()
        for obj in objects:
            if obj.name == object_name:
                return obj.pose
        return None
    
    def get_object_pose_by_segment(self, segment_index: int) -> Optional[Pose3D]:
        """
        Get current world pose of an object by its segment index.
        """
        objects = self.get_scene_objects()
        for obj in objects:
            if obj.obj_ref.get('segment_index') == segment_index:
                return obj.pose
        return None
    
    # ==================== Segmentation Processing ====================
    
    def process_segmentation(
        self,
        seg_image: np.ndarray,
    ) -> Tuple[np.ndarray, List[InteractableObject], Dict[int, str]]:
        """
        Process raw segmentation image to show only interactable objects.
        
        Args:
            seg_image: Raw segmentation image from MuJoCo (instance IDs)
                      Can be (H, W) or (H, W, 2) where channel 1 contains geom IDs
        
        Returns:
            processed_seg: Segmentation image with interactable object indices (background = 0)
            interactable_list: List of InteractableObject found in the image
            segment_id_to_name: Dict mapping segment index to object name
        """
        # MuJoCo segmentation might be multi-channel: [body_id, geom_id]
        # Use the second channel (geom_id) if available, which has more detail
        if seg_image.ndim == 3 and seg_image.shape[2] > 1:
            seg_ids = seg_image[:, :, 1]  # Geom ID channel
        else:
            seg_ids = seg_image if seg_image.ndim == 2 else seg_image[:, :, 0]
        
        # Get interactable objects
        interactables = self.get_interactable_objects()
        
        # Build lookup table: object_id (body_id) -> InteractableObject
        body_id_to_obj = {obj.object_id: obj for obj in interactables}
        
        # Create processed segmentation image (everything starts as background)
        processed_seg = np.zeros_like(seg_ids, dtype=np.int32)
        segment_id_to_name = {0: "background"}
        found_objects = []
        
        # Get unique IDs in the segmentation
        unique_ids = np.unique(seg_ids)
        
        # Access simulation
        robosuite_env = self._get_current_robosuite_env()
        sim = robosuite_env.sim
        
        # For each geom in the segmentation, check if it belongs to an interactable object
        for seg_id in unique_ids:
            if seg_id == 0 or seg_id == -1:  # Background
                continue
            
            try:
                if seg_id >= sim.model.ngeom:
                    continue
                
                # Get the body that owns this geom
                geom_bodyid = sim.model.geom_bodyid[seg_id]
                
                # Check if this body is an interactable object
                if geom_bodyid in body_id_to_obj:
                    obj = body_id_to_obj[geom_bodyid]
                    
                    # Assign this geom's pixels to the object's segment index
                    mask = (seg_ids == seg_id)
                    processed_seg[mask] = obj.segment_index
                    
                    if obj not in found_objects:
                        found_objects.append(obj)
                        segment_id_to_name[obj.segment_index] = obj.name
                # else: keep as background (0)
                
            except:
                # If any error, keep as background
                pass
        
        return processed_seg, found_objects, segment_id_to_name
    
    def _depth_to_pointcloud(self, depth: np.ndarray, camera_params: CameraParams) -> np.ndarray:
        """
        Convert depth image to 3D point cloud in world coordinates.
        
        Args:
            depth: Depth image (H, W) in meters
            camera_params: CameraParams object
            
        Returns:
            points: Point cloud (H, W, 3) in world coordinates
        """
        H, W = depth.shape
        
        # Get camera intrinsics
        intrinsic = camera_params.intrinsic
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Get camera extrinsics and compute cam2world
        extrinsic = camera_params.extrinsic  # world2cam
        cam2world = np.linalg.inv(extrinsic)
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to camera space
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack to get points in camera space (H, W, 3)
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Filter invalid points
        valid_mask = (z > 0) & (z < 10.0)
        
        # Transform to world space
        points_cam_flat = points_cam.reshape(-1, 3)
        points_cam_homo = np.concatenate([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=-1)
        points_world_homo = (points_cam_homo @ cam2world.T)[:, :3]
        
        # Reshape back to image shape
        points_world = points_world_homo.reshape(H, W, 3)
        
        # Apply valid mask
        points_full = np.zeros((H, W, 3), dtype=np.float32)
        points_full[valid_mask] = points_world[valid_mask]
        
        return points_full
    
    def get_keypoint_detection_inputs(
        self,
        camera_name: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Get all inputs needed for keypoint detection.
        
        Args:
            camera_name: Name of the camera to use (default: use vlm_camera)
        
        Returns:
            rgb: RGB image (H, W, 3), uint8
            depth: Depth image (H, W), float32 in meters  
            points: Point cloud (H, W, 3) in world coordinates
            segmentation: Processed segmentation with interactable objects only
            segment_id_to_name: Dict mapping segment index to object name
        """
        if camera_name is None:
            camera_name = self.vlm_camera or 'agentview'
        
        robosuite_env = self._get_current_robosuite_env()
        sim = robosuite_env.sim
        
        # Render RGB, depth, and segmentation
        width = self._env_config.get('visualization_width', 640)
        height = self._env_config.get('visualization_height', 640)
        
        # Render with depth
        rgb, depth = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=True
        )
        
        # Render with segmentation  
        seg_raw = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            segmentation=True
        )
        
        # MuJoCo returns depth in range [0, 1] normalized by near/far planes
        # Use robosuite's official depth conversion formula
        # Reference: robosuite.utils.camera_utils.get_real_depth_map
        model = sim.model
        extent = model.stat.extent
        near = model.vis.map.znear * extent  # Scale by extent!
        far = model.vis.map.zfar * extent    # Scale by extent!
        
        # Convert normalized depth to actual meters
        # Formula: z = near / (1 - depth * (1 - near/far))
        depth_meters = near / (1.0 - depth * (1.0 - near / far))
        depth_meters = depth_meters.astype(np.float32)
        
        # Process segmentation to get only interactable objects
        segmentation, interactable_list, segment_id_to_name = self.process_segmentation(seg_raw)
        
        # Get camera parameters
        camera_params = self.get_camera_params(camera_name)
        
        # Convert depth to point cloud
        points = self._depth_to_pointcloud(depth_meters, camera_params)
        # points[:, 0] -= 0.15
        
        # Ensure RGB is uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # MuJoCo renders images with origin at bottom-left
        # OpenCV/NumPy uses origin at top-left, so flip Y-axis only
        rgb = np.flipud(rgb).copy()
        depth_meters = np.flipud(depth_meters).copy()
        segmentation = np.flipud(segmentation).copy()
        points = np.flipud(points).copy()
        
        return rgb, depth_meters, points, segmentation, segment_id_to_name
    
    # ==================== Action Processing ====================
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information for LIBERO."""
        robosuite_env = self._get_current_robosuite_env()
        robot = robosuite_env.robots[0]
        controller = robot.controller
        
        return {
            'dim': 7,  # 6D pose + 1D gripper
            'type': 'delta_ee_pose',  # OSC controller uses delta control
            'bounds': (np.array([-1.0] * 7), np.array([1.0] * 7)),
            'normalized': True,
            'control_type': controller.control_type if hasattr(controller, 'control_type') else 'OSC_POSE',
            'control_dim': controller.control_dim if hasattr(controller, 'control_dim') else 6,
        }
    
    def delta_actions_to_ee_trajectory(
        self,
        action_sequence: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Transform delta action sequence to 3D trajectory of end-effector.
        
        LIBERO uses OSC (Operational Space Control) with delta EE pose:
        - action[:3]: delta position (dx, dy, dz) in world frame
        - action[3:6]: delta orientation (axis-angle or euler)
        - action[6]: gripper action
        
        Args:
            action_sequence: (T, action_dim) action sequence (torch.Tensor or np.ndarray)
        
        Returns:
            trajectory_3d: (T+1, 3) 3D position trajectory (including start point)
        """
        # Convert to torch if needed
        if isinstance(action_sequence, np.ndarray):
            action_sequence = torch.from_numpy(action_sequence).float()
        
        device = action_sequence.device
        dtype = action_sequence.dtype
        
        # Get starting position as tensor
        start_pos = torch.tensor(
            self.get_ee_pose_world().position,
            device=device,
            dtype=dtype
        )
        
        # Extract delta positions (T, 3)
        # For robosuite OSC, actions are typically normalized to [-1, 1]
        # The actual scale depends on controller settings
        # Default OSC position scale is around 0.05 m per unit action
        ACTION_SCALE_POS = 0.01  # meters per normalized action unit
        
        delta_positions = action_sequence[:, :3] * ACTION_SCALE_POS
        
        # Cumulative sum of deltas - differentiable operation
        cumsum_deltas = torch.cumsum(delta_positions, dim=0)  # (T, 3)
        
        # Build trajectory: [start_pos, start_pos + cumsum[0], start_pos + cumsum[1], ...]
        trajectory = torch.cat([
            start_pos.unsqueeze(0),           # (1, 3)
            start_pos + cumsum_deltas         # (T, 3)
        ], dim=0)  # (T+1, 3)
        
        return trajectory
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action from normalized to actual delta values.
        
        For LIBERO/robosuite OSC, actions are already in normalized form [-1, 1]
        and the controller handles the scaling internally.
        """
        # LIBERO OSC controller handles scaling internally, so just return as-is
        return action

    # ==================== Environment Interface ====================
    def step(self, action: torch.Tensor) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step the LIBERO-plus environment."""

        action_transition = {"action": action}
        action_transition = self.env_postprocessor(action_transition)
        action = action_transition["action"]
        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy().copy()

        # Convert gripper action to binary (-1 or 1). This has to be written to
        # `action_numpy`, which is what gets stepped: `.to("cpu")` on a CUDA
        # tensor returns a copy, so mutating `action` after the conversion (as
        # this did) left the executed action untouched.
        action_numpy[-1] = 1.0 if action_numpy[-1] > 0 else -1.0

        current_env = self._env[self.current_task_idx]
        observation, reward, terminated, truncated, info = current_env.step(action_numpy)
        self.episode_step += 1

        # Cache observation and robot state
        self._last_obs = observation

        # Check for truncation based on max steps
        truncated = self.episode_step >= current_env._max_episode_steps

        # Add task info to info dict
        info.update(self.get_current_task_metadata())
        info['task_description'] = getattr(current_env, 'task_description', '')

        return observation, reward, terminated, truncated, info

    def get_current_task_metadata(self) -> Dict[str, Any]:
        """Suite/task identity plus the perturbation this task variant encodes."""
        meta = self.task_metadata[self.current_task_idx]
        return {
            'suite_name': self.suite_name,
            'task_idx': self.current_task_idx,
            'task_id': meta['task_id'],
            'task_name': meta['task_name'],
            'perturbation_category': meta['category'],
            'difficulty_level': meta['difficulty_level'],
        }

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the current LIBERO-PRO environment.
        
        Handles task switching logic:
        - After completing episodes_per_task episodes, switch to next task
        - Cycles through all tasks in order
        
        Returns:
            (observation, info) tuple
        """
        # Increment episode counter
        self.current_episode_idx += 1
        
        # Check if we need to switch to next task
        if self.current_episode_idx > 0 and self.current_episode_idx % self.episodes_per_task == 0:
            # Move to next task (cycle back to 0 if at end)
            self.current_task_idx = (self.current_task_idx + 1) % self.task_num
            log.info(f"Switching to task {self.current_task_idx} (episode {self.current_episode_idx})")
        
        # Reset episode step counter
        self.episode_step = 0
        
        # Get current environment and reset it
        current_env = self._env[self.current_task_idx]
        # With more than one trial per task, walk through the task's init states
        # so the repeats are not identical rollouts.
        if current_env._init_states is not None:
            trial_idx = self.current_episode_idx % self.episodes_per_task
            current_env._init_state_id = trial_idx % len(current_env._init_states)
        result = current_env.reset(**kwargs)
        
        # Cache robot state from observation
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        self._last_obs = obs
        
        # Add task info to info dict
        info.update(self.get_current_task_metadata())
        info['task_description'] = getattr(current_env, 'task_description', '')
        info['episode_idx'] = self.current_episode_idx
        
        return obs, info
    
    def get_obs(self) -> Dict:
        """Get current observation."""
        if self._last_obs is not None:
            return self._last_obs
        return {}
    
    # ==================== Task-Specific Information ====================
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information for guidance adjustment.
        
        Returns task-specific hints and metadata.
        """
        task_idx = self.current_task_idx
        # Get specific guide scale for this task if configured as a list
        task_guide_scales = self._env_config.get("task_guide_scales", [])
        recommended_scale = None
        if isinstance(task_guide_scales, (list, tuple)) and task_idx < len(task_guide_scales):
            recommended_scale = task_guide_scales[task_idx]
        
        if recommended_scale is None:
            recommended_scale = 3.0  # Default guide scale for LIBERO
            
        info = {
            'instruction': self.get_instruction(),
            'recommended_guide_scale': recommended_scale,
            'task_type': 'manipulation',
            'requires_precision': True,
        }
        info.update(self.get_current_task_metadata())
        return info
    
    def get_instruction(self) -> str:
        """Get the current task instruction/description."""
        return self.get_task_description()

    def close(self):
        """Close every env the lazy sequence still holds open."""
        self._env.close()





        






