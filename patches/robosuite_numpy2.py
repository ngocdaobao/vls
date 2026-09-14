"""Make robosuite 1.4's segmentation rendering work under NumPy 2.

``MjRenderContext.read_pixels`` decodes MuJoCo's ID-color segmentation image as

    rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2**16)

on a ``uint8`` array. NumPy 1 silently promoted that to a wider integer; NumPy 2
(NEP 50) keeps the array's dtype for Python-int operands and raises
``OverflowError: Python integer 256 out of bounds for uint8``. Every
``sim.render(..., segmentation=True)`` call therefore fails, which is what the
keypoint detection of each episode does first.

NumPy cannot be pinned below 2 here (rerun-sdk, required by the LeRobot fork,
needs NumPy 2), so `apply()` swaps in the same method with the decode done in
int32. Must run before any robosuite environment renders.
"""

import mujoco
import numpy as np

_applied = False


def read_pixels(self, width, height, depth=False, segmentation=False):
    """robosuite 1.4.0 ``MjRenderContext.read_pixels``, with an int32 segmentation decode."""
    viewport = mujoco.MjrRect(0, 0, width, height)
    rgb_img = np.empty((height, width, 3), dtype=np.uint8)
    depth_img = np.empty((height, width), dtype=np.float32) if depth else None

    mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

    ret_img = rgb_img
    if segmentation:
        channels = rgb_img.astype(np.int32)
        seg_img = channels[:, :, 0] + channels[:, :, 1] * (2**8) + channels[:, :, 2] * (2**16)
        seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
        seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

        for i in range(self.scn.ngeom):
            geom = self.scn.geoms[i]
            if geom.segid != -1:
                seg_ids[geom.segid + 1, 0] = geom.objtype
                seg_ids[geom.segid + 1, 1] = geom.objid
        ret_img = seg_ids[seg_img]

    if depth:
        return (ret_img, depth_img)
    return ret_img


def apply() -> None:
    """Replace MjRenderContext.read_pixels. Idempotent."""
    global _applied
    if _applied:
        return

    import robosuite
    from robosuite.utils import binding_utils

    if robosuite.__version__ != "1.4.0":
        raise RuntimeError(
            f"patches/robosuite_numpy2.py replaces robosuite 1.4.0's read_pixels, "
            f"but robosuite {robosuite.__version__} is installed; re-check the patch."
        )
    binding_utils.MjRenderContext.read_pixels = read_pixels
    _applied = True
