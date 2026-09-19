#!/usr/bin/env python

"""
Steerable GR00T N1.7 Policy - gradient-based trajectory guidance for GR00T.

This is the GR00T counterpart of ``core/pi05_steer.py``. It keeps exactly the
same steering algorithm (diversity guidance on the noisy end of the trajectory,
keypoint-gradient guidance with a reward-adaptive sigmoid scale on the clean
end, plus optional FKD particle resampling) and the same public API, so
``main_gr00t.py`` drives it with the identical call sites used for PI0.5.

Targets **GR00T N1.7** (``lerobot.policies.groot.groot_n1_7``): the
Cosmos-Reason2 / Qwen3-VL backbone and the ``GR00TN17ActionHead``
flow-matching head. N1.7 replaced N1.5 outright in LeRobot — ``groot_n1.py``
and ``GR00TN15`` are gone — so this file has no N1.5 fallback.

Three things differ from ``pi05_steer.py``, and they are the only places this
file deviates from it:

1. Flow direction.
   PI0.5 integrates from ``time = 1`` (noise) down to ``time = 0`` (clean) with
   a *negative* ``dt``. GR00T integrates from ``t = 0`` (noise) up to ``t = 1``
   (clean) with a *positive* ``dt``. To keep the config semantics identical we
   define ``noise_level = 1 - t``, which plays exactly the role of PI0.5's
   ``time``: it starts at 1.0 and decays to 0.0, so ``start_ratio`` means the
   same thing for both policies.

   Because ``dt`` flips sign, the guidance terms flip sign too. The guidance
   functions return a *reward* (negative squared distance to the keypoint), so
   the sample must ascend the reward gradient, which for PI0.5 (``dt < 0``)
   means ``v_t -= scale * grad`` and for GR00T (``dt > 0``) means
   ``v_t += scale * grad``. Likewise the diversity potential (a sum of inverse
   pairwise distances) must be descended: ``+=`` for PI0.5, ``-=`` here.

2. The action decode is not always differentiable.
   Steering needs a gradient through "normalized model action -> environment
   action", because the guidance functions score an end-effector trajectory
   integrated from environment actions. N1.7 ships two decode steps, and which
   one a checkpoint gets depends on whether it carries sidecar statistics:

   - ``GrootActionUnpackUnnormalizeStep`` (``groot_action_unpack_unnormalize_v2``)
     is pure torch, so gradients flow straight through it.
   - ``GrootN17ActionDecodeStep`` (``groot_n1_7_action_decode_v1``) starts with
     ``action.detach().cpu().float().numpy()``, which severs the graph.

   ``build_grad_postprocessor`` below papers over that. It probes the pipeline
   and, when the graph is severed, rebuilds the decode as the affine map it
   actually is (see that function for why this is exact). Executed actions
   always go through the real pipeline; only the *gradient* path uses the
   surrogate.

3. N1.7 head layout.
   No ``future_tokens`` between state and action embeddings (N1.5 had them),
   an ``AlternateVLDiT`` branch that additionally takes ``image_mask`` and
   ``backbone_attention_mask``, and state carrying a history dimension. The
   sampling loop here mirrors ``GR00TN17ActionHead.get_action_with_features``.

   N1.7's real-time-chunking (RTC) path is deliberately not used: steering
   regenerates a whole chunk each time, so there is no previous chunk to blend
   against and ``vel_strength`` would be all ones anyway.
"""

from collections.abc import Callable
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.constants import ACTION

from core.env_adapters import BaseEnvAdapter
from core.fkd_class import FKD
from utils.logging_utils import SteerLogger

log = SteerLogger("GrootPolicySteer")

# Decode steps that sever the autograd graph, and so need the affine surrogate.
_NON_DIFFERENTIABLE_DECODE_STEPS = {"GrootN17ActionDecodeStep"}


# ----------------------------------------------------------------------------
# Differentiable action decode
# ----------------------------------------------------------------------------

def build_grad_postprocessor(postprocessor: Any) -> Callable:
    """Build a differentiable stand-in for GR00T's action postprocessor.

    The guidance gradient has to flow from the end-effector trajectory back to
    the normalized action sample, which means through the action decode. One of
    N1.7's two decode steps (``GrootN17ActionDecodeStep``) converts to NumPy and
    severs the graph.

    That step is nevertheless an **affine, index-preserving** map on the action:
    it walks the checkpoint's action groups in order, slices a contiguous
    ``[start_idx : start_idx + dim]`` block out of each, applies the min-max
    inverse ``x = (y + 1) / 2 * (max - min) + min`` to it, then concatenates the
    groups back in the same order. Output element ``d`` therefore depends only
    on input element ``d``, affinely. Any such map is recovered exactly from two
    probes::

        offset = f(0)
        scale  = f(1) - f(0)
        f(x)   = x * scale + offset

    so the surrogate reproduces the real decode's numbers rather than
    approximating them. It is re-probed on every call, because with relative
    actions the offset depends on the state cached at pack time, which changes
    from chunk to chunk.

    Two decode features are not affine: ``clip_normalized_action`` (a clamp) and
    the LIBERO gripper transform (a ``sign``). Neither affects steering — the
    guidance gradient is only ever read on dimensions ``:3`` (the end-effector
    xyz deltas), while the clamp is a no-op away from the boundary and the
    gripper transform touches only the last dimension. Executed actions are not
    affected at all, since those go through the real pipeline.

    Args:
        postprocessor: the ``PolicyProcessorPipeline`` for the GR00T policy.

    Returns:
        A callable ``(B, T, D_model) -> (B, T', D_env)`` that is differentiable
        with respect to its input.
    """
    decode_step_names = [
        type(step).__name__
        for step in getattr(postprocessor, "steps", [])
        if type(step).__name__.startswith("Groot")
    ]
    needs_surrogate = any(name in _NON_DIFFERENTIABLE_DECODE_STEPS for name in decode_step_names)

    if not needs_surrogate:
        log.info(
            f"GR00T action decode ({', '.join(decode_step_names) or 'no Groot step'}) is "
            "differentiable; guidance gradients use it directly"
        )
        return postprocessor

    log.info(
        f"GR00T action decode ({', '.join(decode_step_names)}) breaks the autograd graph; "
        "guidance gradients use an exact affine reconstruction of it"
    )

    def grad_postprocess(sample: Tensor) -> Tensor:
        with torch.no_grad():
            offset = postprocessor(torch.zeros_like(sample))
            scale = postprocessor(torch.ones_like(sample)) - offset

        offset = offset.to(device=sample.device, dtype=sample.dtype)
        scale = scale.to(device=sample.device, dtype=sample.dtype)

        # The decode may shorten the horizon (valid_horizon) and always narrows
        # the action dim, so follow whatever shape the probes came back with.
        horizon, width = scale.shape[-2], scale.shape[-1]
        return sample[..., :horizon, :width] * scale + offset

    return grad_postprocess


# ----------------------------------------------------------------------------
# Steerable policy
# ----------------------------------------------------------------------------

class GrootPolicySteer(GrootPolicy):
    """GR00T N1.7 policy with gradient-based trajectory steering."""

    name = "groot_steer"

    def __init__(self, config):
        super().__init__(config)
        self._adapter = None
        self._postprocessor = None
        self._grad_postprocessor = None
        self._original_action_dim = self.config.output_features[ACTION].shape[0]
        self._cached_action_chunk = None
        # Reward-based adaptive guidance
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0

    def post_init(
        self,
        adapter: BaseEnvAdapter,
        postprocessor: Callable,
        sample_batch_size: int,
        policy_config: dict,
    ) -> None:
        """Initialize steering components after model loading."""
        self._adapter = adapter
        self._postprocessor = postprocessor
        self._grad_postprocessor = build_grad_postprocessor(postprocessor)
        self._sample_batch_size = sample_batch_size
        self._inference_steps = policy_config['num_inference_steps']
        self._action_chunk_horizon = policy_config['action_chunk_horizon']

        # The action head's own horizon caps what we can ask for: GR00T's
        # pretrained DiT is built for a fixed chunk length.
        head_horizon = self._action_head.action_horizon
        if self._action_chunk_horizon > head_horizon:
            log.warning(
                f"action_chunk_horizon={self._action_chunk_horizon} exceeds the GR00T action head's "
                f"horizon ({head_horizon}); clamping to {head_horizon}"
            )
            self._action_chunk_horizon = head_horizon

        # Keep the head's denoising budget in sync with the configured one, so
        # FKD's step bookkeeping and the model agree on the step count.
        self._action_head.num_inference_timesteps = self._inference_steps

    @property
    def _action_head(self):
        return self._groot_model.action_head

    def reset(self):
        """Reset policy state for new episode."""
        super().reset()
        self._cached_action_chunk = None
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0

    def reset_stage(self):
        """Reset stage-specific state (call when stage changes)."""
        self._stage_init_reward = None

    def get_normalized_reward(self) -> float:
        """Get last normalized reward (0=start, 1=reached target)."""
        return self._last_normalized_reward

    def get_last_scale(self) -> float:
        """Get last guidance scale used."""
        return self._last_scale

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        generate_new_chunk: bool = False,
        use_guidance: bool = False,
        keypoints: Optional[np.ndarray] = None,
        guidance_fns: Optional[dict] = None,  # Dict of {stage_num: guidance_fn}
        guide_scale: float = 1.0,
        start_ratio: Optional[float] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        MCMC_steps: int = 4,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
        **kwargs  # Ignore other params for backward compatibility
    ) -> Tensor:

        self.eval()

        if ACTION in batch:
            batch.pop(ACTION)

        if generate_new_chunk:
            if use_guidance and guidance_fns:
                action_chunk = self._sample_actions_guided(
                    batch=batch,
                    keypoints=keypoints,
                    guidance_fn=guidance_fns,  # Just use the functions passed in
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
                    sigmoid_x0=sigmoid_x0,
                )
            else:
                action_chunk = self._sample_actions_guided(
                    batch=batch,
                    keypoints=None,
                    guidance_fn=None,
                    use_diversity=False,
                    verbose=verbose,
                    use_fkd=False,
                    global_step=global_step,
                    current_stage=current_stage,
                )

            self._cached_action_chunk = action_chunk
        else:
            action_chunk = self._cached_action_chunk

        return self._postprocessor(action_chunk) if self._postprocessor else action_chunk

    # ------------------------------------------------------------------
    # Core guided sampling
    # ------------------------------------------------------------------

    def _sample_actions_guided(
        self,
        batch: dict[str, Tensor],
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
    ) -> Tensor:
        """
        Core guided sampling with reward-based adaptive guidance.

        Mirrors ``GR00TN17ActionHead.get_action_with_features`` step for step,
        with guidance folded into the velocity before each Euler update.

        Guidance strategy (identical to PI0.5, expressed on ``noise_level``,
        which runs 1 -> 0 as denoising progresses):
        1. noise_level > start_time: Diversity guidance (spread trajectories)
        2. noise_level <= start_time: Keypoint gradient guidance with adaptive
           scale - scale adapts based on reward value (higher reward = softer
           guidance)
        """
        head = self._action_head
        device = next(self.parameters()).device

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            # Preprocessing is done by the processor pipeline; just filter the
            # batch down to what GR00T consumes. No action/action_mask at
            # inference time - that is what the model predicts.
            groot_inputs = self._filter_groot_inputs(batch, include_action=False)
            backbone_inputs, action_inputs = self._groot_model.prepare_input(groot_inputs)
            backbone_outputs = self._groot_model.backbone(backbone_inputs)

            # Handles vlln + VL self-attention and the state history reshape.
            features = head._encode_features(backbone_outputs, action_inputs)
            vl_embeds = features.backbone_features
            state_features = features.state_features
            embodiment_id = action_inputs.embodiment_id

            bsize = vl_embeds.shape[0]
            num_steps = self._inference_steps

            # Sample initial noise
            actions = torch.randn(
                size=(bsize, head.config.action_horizon, head.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

            # Setup guidance. ``start_ratio`` arrives from the config as None or
            # the literal string "None" when unset, so treat anything that is
            # not a real number as "use the default".
            start_time = start_ratio if isinstance(start_ratio, (int, float)) else 0.8

            keypoints_tensor = None
            if keypoints is not None:
                keypoints_tensor = torch.tensor(keypoints, device=device, dtype=torch.float32)

            use_keypoint_guidance = guidance_fn is not None and keypoints_tensor is not None

            # Initialize FKD if enabled
            fkd = self._init_fkd(fkd_config, bsize, num_steps, start_time,
                                 keypoints_tensor, guidance_fn, device) if use_fkd else None

            # Flow matching loop with guidance. GR00T integrates forwards:
            # t goes 0 -> 1 with a positive dt, so noise_level = 1 - t is the
            # quantity that matches PI0.5's `time`.
            dt = 1.0 / num_steps
            H = self._action_chunk_horizon

            # Reward tracking for debugging
            reward_history = []
            guidance_start_step = int((1.0 - start_time) * num_steps)

            for step_idx in range(num_steps):
                t_cont = step_idx / float(num_steps)
                noise_level = 1.0 - t_cont
                t_discretized = int(t_cont * head.num_timestep_buckets)

                # Get velocity from model
                timesteps_tensor = torch.full(size=(bsize,), fill_value=t_discretized, device=device)
                action_features = head.action_encoder(actions, timesteps_tensor, embodiment_id)
                if head.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)

                # N1.7 concatenates state and action embeddings directly; there
                # are no future tokens between them.
                sa_embs = torch.cat((state_features, action_features), dim=1)

                if head.config.use_alternate_vl_dit:
                    model_output = head.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_outputs.image_mask,
                        backbone_attention_mask=backbone_outputs.backbone_attention_mask,
                    )
                else:
                    model_output = head.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )
                pred = head.action_decoder(model_output, embodiment_id)
                v_t = pred[:, -head.action_horizon:]

                # Apply guidance
                if use_diversity and noise_level > start_time and bsize > 1:
                    div_grad = self._compute_diversity_gradient(
                        actions[:, :, :self._original_action_dim], verbose=(verbose and step_idx == 0)
                    )
                    if div_grad is not None:
                        # dt > 0, so descend the inverse-distance potential with a minus.
                        v_t[:, :H, :3] -= diversity_scale * div_grad[:, :H, :3].to(v_t.dtype)

                elif use_keypoint_guidance and noise_level <= start_time:
                    kp_grad, reward_value = self._compute_keypoint_gradient(
                        actions[:, :, :self._original_action_dim],
                        keypoints_tensor,
                        guidance_fn,
                        verbose=(verbose and step_idx == guidance_start_step),
                    )

                    if kp_grad is not None:
                        # Reward-based adaptive scaling.
                        # Normalized reward: 0 (stage start) -> 1 (target reached).
                        if self._stage_init_reward is not None and self._stage_init_reward < -1e-6:
                            normalized_reward = 1 - (reward_value / self._stage_init_reward)
                            normalized_reward = max(0.0, min(1.2, normalized_reward))
                        else:
                            normalized_reward = 0.0  # No baseline yet, treat as start

                        self._last_normalized_reward = normalized_reward

                        # Track reward for debugging
                        reward_history.append((step_idx, reward_value, normalized_reward))

                        # Sigmoid map from "full guidance" to "graceful release":
                        # strength ~1.0 well below sigmoid_x0, exactly 0.5 at it.
                        guidance_strength = 1.0 / (1.0 + np.exp(sigmoid_k * (normalized_reward - sigmoid_x0)))

                        scale = guide_scale * guidance_strength
                        self._last_scale = scale

                        if verbose and step_idx == guidance_start_step:
                            init_str = f"{self._stage_init_reward:.6f}" if self._stage_init_reward is not None else "None"
                            log.info(
                                f"[Step {global_step}] Stage {current_stage} | "
                                f"reward={reward_value:.6f}, init={init_str}, "
                                f"norm_r={normalized_reward:.3f}, sig_strength={guidance_strength:.3f}, scale={scale:.2f}"
                            )

                        # dt > 0, so ascend the reward gradient with a plus.
                        v_t[:, :H, :3] += scale * kp_grad[:, :H, :3].to(v_t.dtype)

                # Euler step
                actions = actions + dt * v_t

                # FKD resampling
                if fkd is not None and noise_level <= start_time:
                    actions, _ = fkd.resample(sampling_idx=step_idx, latents=actions, x0_preds=actions)

            # Use the final reward of the first chunk as baseline for normalization
            if reward_history and self._stage_init_reward is None:
                final_reward = reward_history[-1][1]  # Last step's reward
                self._stage_init_reward = final_reward
                log.info(f"[Step {global_step}] Stage {current_stage} init_reward={final_reward:.6f} (from first chunk's final step)")

        # Unpad and return
        return actions[:, :self._action_chunk_horizon, :self._original_action_dim]

    def _init_fkd(self, fkd_config, bsize, num_steps, start_time,
                  keypoints_tensor, guidance_fn, device):
        """Initialize FKD for particle resampling."""
        if fkd_config is None or bsize <= 1 or keypoints_tensor is None:
            return None

        def reward_fn(x0_preds):
            traj = self._sample_to_trajectory_3d(
                x0_preds[:, :, :self._original_action_dim]
            )[:, 1:self._action_chunk_horizon, :3]
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

        # GR00T denoises noise -> clean, so the guided (clean) phase starts at
        # step int((1 - start_time) * num_steps), not int(start_time * num_steps).
        return FKD(
            potential_type=fkd_config.get('potential_type', 'max'),
            lmbda=fkd_config.get('lmbda', 10.0),
            num_particles=bsize,
            adaptive_resampling=fkd_config.get('adaptive_resampling', True),
            resample_frequency=fkd_config.get('resample_frequency', 5),
            resampling_t_start=int((1.0 - start_time) * num_steps),
            resampling_t_end=num_steps,
            timesteps=torch.linspace(1.0, 0.0, num_steps + 1, device=device),
            reward_fn=reward_fn,
            reward_min_value=float('-inf'),
            device=device,
        )

    def _sample_to_trajectory_3d(self, sample: Tensor) -> Tensor:
        """Convert action sample to 3D EE trajectory.

        Uses the differentiable decode, since this is what the guidance
        gradient is taken through.
        """
        if self._adapter is None or self._grad_postprocessor is None:
            raise RuntimeError("Call post_init() first")

        device, dtype = sample.device, sample.dtype
        actions = self._grad_postprocessor(sample).to(device, dtype)
        batch_size = sample.shape[0]

        action_transition = {"action": actions}
        action_transition = self._adapter.env_postprocessor(action_transition)
        actions = action_transition["action"]

        if batch_size == 1:
            traj = self._adapter.delta_actions_to_ee_trajectory(
                actions.squeeze(0)[:self._action_chunk_horizon]
            )
            return traj.unsqueeze(0).to(device, dtype)

        trajs = [
            self._adapter.delta_actions_to_ee_trajectory(actions[b, :self._action_chunk_horizon])
            for b in range(batch_size)
        ]
        return torch.stack(trajs, dim=0).to(device, dtype)

    def _compute_keypoint_gradient(
        self,
        sample: Tensor,
        keypoints_tensor: Tensor,
        guidance_fn: Union[Callable, List[Callable]],
        verbose: bool = False,
    ) -> tuple[Optional[Tensor], float]:
        """
        Compute gradient for keypoint guidance with statistics for adaptive scaling.

        The gradient is taken in fp32 even when the model runs in bf16: the
        adapter's trajectory math and the VLM-written guidance functions are
        numerically far too tight for bf16.

        Returns:
            normalized_grad: Normalized gradient (or None)
            reward_value: Raw reward value from guidance function
        """
        if not guidance_fn or self._adapter is None:
            return None, 0.0

        try:
            with torch.enable_grad(), torch.autocast(device_type=sample.device.type, enabled=False):
                sample_grad = sample.detach().float().requires_grad_(True)
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

    def _compute_diversity_gradient(self, sample: Tensor, verbose: bool = False) -> Optional[Tensor]:
        """Compute RBF diversity gradient."""
        batch_size = sample.shape[0]
        if batch_size < 2 or self._adapter is None:
            return None

        with torch.enable_grad(), torch.autocast(device_type=sample.device.type, enabled=False):
            sample_grad = sample.detach().float().requires_grad_(True)
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
