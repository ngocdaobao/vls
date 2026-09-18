from __future__ import annotations

import logging
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available, require_package

from .action_head.cross_attention_dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from .configuration_groot import N1_7_DEFAULT_IMAGE_CROP_SIZE, N1_7_DEFAULT_IMAGE_TARGET_SIZE

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoConfig,
        AutoModel,
        PretrainedConfig,
        PreTrainedModel,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    )
    from transformers.feature_extraction_utils import BatchFeature
else:
    AutoConfig = None
    AutoModel = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None
    Qwen3VLConfig = None
    Qwen3VLForConditionalGeneration = None

try:
    import tree
except ImportError:
    tree = None

logger = logging.getLogger(__name__)

from collections import deque
from collections.abc import Callable
from typing import List, Optional, Union
import numpy as np
import torch
from torch import Tensor

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from core.env_adapters import BaseEnvAdapter
from core.fkd_class import FKD
from utils.logging_utils import SteerLogger

log = SteerLogger("GR00T5PolicySteer")

class GrootPolicySteer(GrootPolicy):
    name = "groot_steer"
     
    def __init__(self, config):
        super().__init__(config)
        self._adapter = None
        self._postprocessor = None
        self._cached_action_chunk = None
        # Reward-based adaptive guidance
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
        self.action_head = SteerActionHead(config.action_head)

    def post_init(
        self,
        adapter: BaseEnvAdapter,
        postprocessor: Callable,
        sample_batch_size: int,
        policy_config: dict,
    ) -> None:
        """Initialize steering components after model loading."""
        self.action_head._adapter = adapter
        self._postprocessor = postprocessor
        self._sample_batch_size = sample_batch_size
        self._inference_steps = policy_config['num_inference_steps']
        self._action_chunk_horizon = policy_config['action_chunk_horizon']

    def reset(self):
        """Reset policy state for new episode."""
        self._cached_action_chunk = None
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
        self._action_queue = deque([], maxlen=self._action_queue_steps)
    
    def reset_stage(self):
        """Reset stage-specific state (call when stage changes)."""
        self._stage_init_reward = None
    
    def get_normalized_reward(self) -> float:
        """Get last normalized reward (0=start, 1=reached target)."""
        return self._last_normalized_reward
    
    def get_last_scale(self) -> float:
        """Get last guidance scale used."""
        return self._last_scale

    def select_action(self):
        return

class GROOTN17Steer(GR00TN17):
    def get_action(self, inputs):
        return

class SteerActionHead(GR00TN17ActionHead):
    def __init__(self, config):
        super().__init__(config)
        self._adapter = None
        self._original_action_dim = self.config.output_features[ACTION].shape[0]
        self._cached_action_chunk = None
        # Reward-based adaptive guidance
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0       
    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_ratio: Optional[float] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
    ) -> BatchFeature:
        
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        # dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            if options is None:
                raise ValueError("RTC options are required when action is provided to get_action.")
            action_horizon_before_padding = options["action_horizon"]
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)
            vel_strength[:, options["rtc_frozen_steps"] : options["rtc_overlap_steps"], :] = ramp[1:-1][
                None, :, None
            ].to(device)
        # Setup guidance
        start_time = start_ratio if start_ratio is None else 0.8
        dt = torch.tensor(-1.0 / self.num_inference_timesteps)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        bsize = vl_embeds.shape[0]
        step_idx=0

        keypoints_tensor = None
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, device=device, dtype=torch.float32)
        
        use_keypoint_guidance = guidance_fn is not None and keypoints_tensor is not None

        # Initialize FKD if enabled
        fkd = self._init_fkd(fkd_config, bsize, self.num_inference_timesteps, start_time, 
                             keypoints_tensor, guidance_fn, device) if use_fkd else None

        reward_history = []

        while time >= -dt/2:
            # t_cont = t_step / float(self.num_inference_timesteps)
            # t_discretized = int(t_cont * self.num_timestep_buckets)
            expanded_time = time.expand(bsize)
            # timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=expanded_time, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            if use_diversity and time > start_time and bsize > 1:
                div_grad = self._compute_diversity_gradient(actions, verbose=(verbose and step_idx == 0))
                if div_grad is not None:
                    pred += diversity_scale * div_grad

            elif use_keypoint_guidance and time <= start_time:
                kp_grad, reward_value = self._compute_keypoint_gradient(
                    actions, 
                    keypoints_tensor, 
                    guidance_fn, 
                    verbose=(verbose and step_idx == int(start_time * self.num_inference_steps))
                )
                
                if kp_grad is not None:
                    if self._stage_init_reward is not None and self._stage_init_reward < -1e-6:
                        normalized_reward = 1 - (reward_value / self._stage_init_reward)
                        normalized_reward = max(0.0, min(1.2, normalized_reward))
                    else:
                        normalized_reward = 0.0  # No baseline yet, treat as start
                    
                    self._last_normalized_reward = normalized_reward
                    
                    # Track reward for debugging
                    reward_history.append((step_idx, reward_value, normalized_reward))
                    guidance_strength = 1.0 / (1.0 + np.exp(sigmoid_k * (normalized_reward - sigmoid_x0)))
                    
                    scale = guide_scale * guidance_strength 
                    self._last_scale = scale
                    
                    if verbose and step_idx == int(start_time * self.num_inference_steps):
                        init_str = f"{self._stage_init_reward:.6f}" if self._stage_init_reward is not None else "None"
                        log.info(
                            f"[Step {global_step}] Stage {current_stage} | "
                            f"reward={reward_value:.6f}, init={init_str}, "
                            f"norm_r={normalized_reward:.3f}, sig_strength={guidance_strength:.3f}, scale={scale:.2f}"
                        )
                    
                    pred -= scale * kp_grad

            actions = actions + dt * pred[:, -self.action_horizon :] * vel_strength

            if fkd is not None and time <= start_time:
                actions, _ = fkd.resample(sampling_idx=step_idx, latents=actions, x0_preds=actions)

            time = time + dt
            step_idx += 1

        # Use the final reward of the first chunk as baseline for normalization
        if reward_history and self._stage_init_reward is None:
            final_reward = reward_history[-1][1]  # Last step's reward
            self._stage_init_reward = final_reward
            log.info(f"[Step {global_step}] Stage {current_stage} init_reward={final_reward:.6f} (from first chunk's final step)")

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_ratio: Optional[float] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
    ) -> BatchFeature:
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
            keypoints=keypoints,
            guidance_fn=guidance_fn,
            guide_scale=guide_scale,
            start_ratio=start_ratio,
            use_diversity=use_diversity,
            diversity_scale=diversity_scale,
            verbose=verbose,
            use_fkd=use_fkd,
            fkd_config=fkd_config,
            global_step=global_step,
            current_stage=current_stage,
            sigmoid_k=sigmoid_k,
            sigmoid_x0=sigmoid_x0
        )

    def _compute_diversity_gradient(self, sample: Tensor, verbose: bool = False) -> Optional[Tensor]:
        """Compute RBF diversity gradient."""
        batch_size = sample.shape[0]
        if batch_size < 2 or self._adapter is None:
            return None

        with torch.enable_grad():
            sample_grad = sample.detach().requires_grad_(True)
            traj = self._sample_to_trajectory_3d(sample_grad)[:, :self._action_chunk_horizon, :3]
            traj_flat = traj.reshape(batch_size, -1)

            # Pairwise distance potential
            diff = traj_flat.unsqueeze(1) - traj_flat.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sample.device)
            potential = (1.0 / (dist + 1e-6) * mask.float()).sum()

            grad = torch.autograd.grad(potential, sample_grad)[0]
            grad_norm = torch.norm(grad)
            # Normalize to unit norm for stability
            normalized_grad = grad / (grad_norm + 1e-8) if grad_norm > 1e-8 else grad

            return normalized_grad

    def _init_fkd(self, fkd_config, bsize, num_steps, start_time, 
                  keypoints_tensor, guidance_fn, device):
        """Initialize FKD for particle resampling."""
        if fkd_config is None or bsize <= 1 or keypoints_tensor is None:
            return None
        
        def reward_fn(x0_preds):
            traj = self._sample_to_trajectory_3d(x0_preds)[:, 1:self._action_chunk_horizon, :3]
            rewards = []
            for b in range(traj.shape[0]):
                try:
                    if isinstance(guidance_fn, list):
                        r = sum(fn(keypoints_tensor, traj[b:b+1]) for fn in guidance_fn)
                    else:
                        r = guidance_fn(keypoints_tensor, traj[b:b+1])
                    rewards.append(float(r.item()) if hasattr(r, 'item') else float(r))
                except Exception:
                    rewards.append(0.0)
            return torch.tensor(rewards, device=device, dtype=torch.float32)
        
        return FKD(
            potential_type=fkd_config.get('potential_type', 'max'),
            lmbda=fkd_config.get('lmbda', 10.0),
            num_particles=bsize,
            adaptive_resampling=fkd_config.get('adaptive_resampling', True),
            resample_frequency=fkd_config.get('resample_frequency', 5),
            resampling_t_start=int(start_time * num_steps),
            resampling_t_end=num_steps,
            timesteps=torch.linspace(1.0, 0.0, num_steps + 1, device=device),
            reward_fn=reward_fn,
            reward_min_value=float('-inf'),
            device=device,
        )

    def _compute_keypoint_gradient(
        self,
        sample: Tensor,
        keypoints_tensor: Tensor,
        guidance_fn: Union[Callable, List[Callable]],
        verbose: bool = False,
    ) -> tuple[Optional[Tensor], float]:
        """
        Compute gradient for keypoint guidance with statistics for adaptive scaling.
        
        Returns:
            normalized_grad: Normalized gradient (or None)
            reward_value: Raw reward value from guidance function
            grad_norm: Original gradient norm (before normalization) - key for adaptive scaling
        """
        if not guidance_fn or self._adapter is None:
            return None, 0.0

        try:
            with torch.enable_grad():
                sample_grad = sample.detach().requires_grad_(True)
                traj = self._sample_to_trajectory_3d(sample_grad)[:, :self._action_chunk_horizon, :3]
                
                # Keep batch dimension for consistency - guidance functions expect (B, T, 3)
                # Don't squeeze even when batch_size=1
                traj_input = traj
                
                if isinstance(guidance_fn, list):
                    reward = sum(fn(keypoints_tensor, traj_input) for fn in guidance_fn)
                else:
                    reward = guidance_fn(keypoints_tensor, traj_input)
                
                if reward is None or not reward.requires_grad:
                    return None, 0.0
                
                if reward.dim() > 0:
                    reward = reward.sum()

                reward_value = reward.item() if hasattr(reward, 'item') else float(reward)
                
                grad = torch.autograd.grad(reward, sample_grad)[0]
                grad_norm_value = torch.norm(grad).item()
                normalized_grad = grad / (grad_norm_value + 1e-8) if grad_norm_value > 1e-8 else grad
                
                if verbose:
                    log.info(f"Reward: {reward_value:.4f}")
                
                return normalized_grad, reward_value
        except Exception as e:
            if verbose or not hasattr(self, '_guidance_error_logged'):
                log.warning(f"Guidance function error: {e}")
                self._guidance_error_logged = True
            return None, 0.0




