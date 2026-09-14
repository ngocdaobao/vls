"""
VLS utilities for multi-backend support.

This package provides:
- env_adapters: Unified interface for CALVIN, LIBERO, and real-world robots
- keypoint_tracker: Keypoint tracking using adapters
- sam_segmenter: Independent SAM segmentation component
"""

from .env_adapters import (
    BaseEnvAdapter,
    Pose3D,
    CameraParams,
    TrackedObject,
    create_adapter,
)
from .keypoint_tracker import KeypointTracker


def __getattr__(name: str):
    """Forward adapter classes to env_adapters, which imports them lazily."""
    if name in ("LiberoAdapter", "LiberoPlusAdapter", "CalvinAdapter"):
        from . import env_adapters

        return getattr(env_adapters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# SAM segmentation (lazy import to avoid loading heavy models)
def get_sam_segmenter(*args, **kwargs):
    """Lazy import SAMSegmenter to avoid loading heavy models on import."""
    from .sam_segmenter import SAMSegmenter
    return SAMSegmenter(*args, **kwargs)

def get_vlm_sam_pipeline(*args, **kwargs):
    """Lazy import VLMSAMPipeline to avoid loading heavy models on import."""
    from .sam_segmenter import VLMSAMPipeline
    return VLMSAMPipeline(*args, **kwargs)

__all__ = [
    # Adapters
    "BaseEnvAdapter",
    "Pose3D",
    "CameraParams", 
    "TrackedObject",
    "LiberoAdapter",
    "LiberoPlusAdapter",
    "create_adapter",
    # Keypoint tracking
    "KeypointTracker",
    # SAM segmentation
    "get_sam_segmenter",
    "get_vlm_sam_pipeline",
]

