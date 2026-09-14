"""Make the ImageMagick C library findable before LIBERO-plus imports Wand.

LIBERO-plus's env wrapper imports ``wand`` at module level for its sensor-noise
perturbations, and Wand is only a ctypes binding: the ``pip``/``uv`` package
does not ship ``libMagickWand``. Without it, ``import libero.libero`` fails for
*every* task, not just the noise ones.

On machines without root, scripts/setup_libero_plus.sh unpacks the distribution's
own ImageMagick packages into ``.deps/imagemagick``. It deliberately does not use
a conda-forge build: those link against newer GLib/libstdc++/X11 than the system
copies that torch and Mesa's EGL have already loaded into this process by the
time LIBERO is imported, so ``dlopen`` fails with undefined symbols.

`apply()` points Wand (``MAGICK_HOME``) and ImageMagick's coder/config lookup at
that prefix. Wand reads the environment when ``wand.api`` is first imported, so
this must run before LIBERO-plus is imported.
"""

import ctypes.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PREFIX = PROJECT_ROOT / ".deps" / "imagemagick"

# The sonames Wand itself probes, newest first.
_SYSTEM_LIBRARY_NAMES = (
    "MagickWand-7.Q16HDRI",
    "MagickWand-7.Q16",
    "MagickWand-6.Q16HDRI",
    "MagickWand-6.Q16",
    "MagickWand",
)


def _has_local_prefix() -> bool:
    return any((LOCAL_PREFIX / "lib").glob("libMagickWand*.so*"))


def _point_at_local_prefix() -> None:
    lib = LOCAL_PREFIX / "lib"
    os.environ["MAGICK_HOME"] = str(LOCAL_PREFIX)
    # The .deb build has /usr/lib/... compiled in as its module and config
    # directories; the PNG coder LIBERO-plus's motion blur uses lives there.
    modules = next(lib.glob("ImageMagick-*/modules-Q16"), None)
    if modules is not None:
        os.environ.setdefault("MAGICK_CODER_MODULE_PATH", str(modules / "coders"))
        os.environ.setdefault("MAGICK_CODER_FILTER_PATH", str(modules / "filters"))
    config = next((LOCAL_PREFIX / "etc").glob("ImageMagick-*"), None)
    if config is not None:
        os.environ.setdefault("MAGICK_CONFIGURE_PATH", str(config))


def apply() -> None:
    """Point Wand at an ImageMagick library, or fail with setup instructions."""
    if os.environ.get("MAGICK_HOME"):
        return
    if _has_local_prefix():
        _point_at_local_prefix()
        return
    if any(ctypes.util.find_library(name) for name in _SYSTEM_LIBRARY_NAMES):
        return
    raise ImportError(
        "ImageMagick (libMagickWand) not found, but LIBERO-plus needs it. Either run "
        "`bash scripts/setup_libero_plus.sh` to install it into .deps/imagemagick, "
        "install it system-wide (`apt install libmagickwand-6.q16-6`), or set MAGICK_HOME."
    )
