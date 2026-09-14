#!/usr/bin/env python
"""
Finalize an openpi->pytorch converted pi05 checkpoint into a LeRobot pretrained dir.

The converted checkpoint (`model.safetensors` + openpi-style `config.json`) is missing
the two things LeRobot's loader needs:

  1. A LeRobot `PI05Config` `config.json`.
  2. The preprocessor / postprocessor pipeline files that
     `lerobot.policies.factory.make_pre_post_processors(pretrained_path=...)` loads.

This script writes both, taking normalization statistics from the original openpi
`assets/.../norm_stats.json` so the pipelines match how the checkpoint was trained.

Usage:
    python scripts/convert_openpi_pi05_to_lerobot.py \
        --checkpoint-dir pi05_libero_py \
        --norm-stats pi05_libero/assets/physical-intelligence/libero/norm_stats.json
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401  (registers "pi05")


# LeRobot PI05Config for the openpi `pi05_libero` checkpoint. Mirrors the reference
# conversion published as `lerobot/pi05_libero_base`, with device/dtype set for this repo.
LEROBOT_CONFIG = {
    "type": "pi05",
    "n_obs_steps": 1,
    "input_features": {
        "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.state": {"type": "STATE", "shape": [8]},
    },
    "output_features": {
        "action": {"type": "ACTION", "shape": [7]},
    },
    "empty_cameras": 1,
    "device": "cuda",
    "use_amp": False,
    "push_to_hub": False,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "bfloat16",
    "chunk_size": 50,
    "n_action_steps": 10,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "time_sampling_beta_alpha": 1.5,
    "time_sampling_beta_beta": 1.0,
    "min_period": 0.004,
    "max_period": 4.0,
    "image_resolution": [224, 224],
    "gradient_checkpointing": False,
    "compile_model": False,
    "compile_mode": "max-autotune",
    "optimizer_lr": 2.5e-05,
    "optimizer_betas": [0.9, 0.95],
    "optimizer_eps": 1e-08,
    "optimizer_weight_decay": 0.01,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-06,
    "tokenizer_max_length": 200,
}


def build_dataset_stats(norm_stats_path: Path, config: PreTrainedConfig) -> dict:
    """Translate openpi norm_stats.json into LeRobot's per-feature stats dict."""
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)["norm_stats"]

    # openpi key -> LeRobot feature key
    mapping = {"state": "observation.state", "actions": "action"}

    stats = {}
    for openpi_key, feature_key in mapping.items():
        if openpi_key not in norm_stats:
            raise KeyError(f"'{openpi_key}' missing from {norm_stats_path}")
        entry = norm_stats[openpi_key]
        stats[feature_key] = {
            name: torch.tensor(entry[name], dtype=torch.float32)
            for name in ("mean", "std", "q01", "q99")
            if name in entry
        }

    # QUANTILES normalization needs q01/q99; MEAN_STD needs mean/std. Fail loudly rather
    # than silently normalizing with the wrong statistics.
    for feature_key, feature in {**config.input_features, **config.output_features}.items():
        mode = config.normalization_mapping[feature.type.name]
        if mode.value == "IDENTITY":
            continue
        required = ("q01", "q99") if mode.value == "QUANTILES" else ("mean", "std")
        available = stats.get(feature_key, {})
        missing = [name for name in required if name not in available]
        if missing:
            raise ValueError(
                f"{feature_key} uses {mode.value} normalization but {norm_stats_path} "
                f"has no {missing} entry"
            )
        for name in required:
            expected = tuple(feature.shape)
            actual = tuple(available[name].shape)
            if actual != expected:
                raise ValueError(
                    f"{feature_key}.{name} has shape {actual}, config expects {expected}"
                )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", 
        type=Path,
        required=True,
        help="Directory holding the converted model.safetensors; written to in place.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        required=True,
        help="Path to the openpi assets norm_stats.json for this checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device recorded in config.json (default: cuda).",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"No model.safetensors in {checkpoint_dir}")
    if not args.norm_stats.exists():
        raise FileNotFoundError(f"No norm stats at {args.norm_stats}")

    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        backup = checkpoint_dir / "config.openpi.json"
        if not backup.exists():
            shutil.copy2(config_path, backup)
            print(f"Backed up openpi config to {backup}")

    config_dict = dict(LEROBOT_CONFIG, device=args.device)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Wrote LeRobot config to {config_path}")

    # Parse through the base class so the "type" discriminator is honored.
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    dataset_stats = build_dataset_stats(args.norm_stats, config)
    for feature_key, feature_stats in dataset_stats.items():
        print(f"  {feature_key}: {sorted(feature_stats)}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset_stats,
    )
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    print(f"Wrote processor pipelines to {checkpoint_dir}")

    for name in sorted(p.name for p in checkpoint_dir.iterdir()):
        print(f"  {name}")


if __name__ == "__main__":
    main()
