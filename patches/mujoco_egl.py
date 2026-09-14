"""Pick a valid EGL render device for this process.

robosuite resolves its offscreen EGL display by treating ``MUJOCO_EGL_DEVICE_ID``
(or, when that is unset, ``CUDA_VISIBLE_DEVICES``) as an *index into the EGL
device list*, and additionally asserts at import time that
``MUJOCO_EGL_DEVICE_ID`` is a substring of ``CUDA_VISIBLE_DEVICES``. CUDA
ordinals and EGL device indices are unrelated numberings: this node, for
instance, exposes 4 CUDA GPUs but a single EGL device, so a worker pinned to GPU
2 dies with

    RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an
    integer between 0 and 0 (inclusive), got 2

`apply()` takes the EGL device selection over: it maps the requested GPU onto an
EGL device that actually exists (by CUDA ordinal when the driver reports one,
otherwise by index, clamped), and hides ``MUJOCO_EGL_DEVICE_ID`` from robosuite
so its import-time assert cannot fire.

Must run before robosuite/LIBERO is imported.
"""

import os

from utils.logging_utils import SteerLogger

log = SteerLogger("MujocoEGL")

# EGL_CUDA_DEVICE_NV: the CUDA ordinal an EGL device belongs to (NVIDIA only).
_EGL_CUDA_DEVICE_NV = 0x323A

_applied = False


def _requested_gpu() -> int:
    """The GPU this process should render on, as a CUDA ordinal.

    VLS_EGL_DEVICE_ID is set by the multi-GPU launcher and holds the *physical*
    id, which is what the EGL driver reports; MUJOCO_EGL_DEVICE_ID is honoured
    for hand-run processes. Both fall back to the first visible CUDA device.
    """
    for var in ("VLS_EGL_DEVICE_ID", "MUJOCO_EGL_DEVICE_ID"):
        value = os.environ.get(var, "").strip()
        if value.isdigit():
            return int(value)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


def _egl_device_index(requested_gpu: int, num_devices: int) -> int:
    """Map a CUDA ordinal to an index into eglQueryDevicesEXT()."""
    try:
        from OpenGL import EGL
        from OpenGL.EGL.EXT.device_query import eglQueryDeviceAttribEXT
        from mujoco.egl import egl_ext as EGL_EXT

        import ctypes
        for index, device in enumerate(EGL_EXT.eglQueryDevicesEXT()):
            value = EGL.EGLAttrib()
            if not eglQueryDeviceAttribEXT(device, _EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
                continue
            if int(value.value) == requested_gpu:
                return index
    except Exception as exc:  # driver without EGL_NV_device_cuda, older PyOpenGL, ...
        log.debug(f"Cannot query EGL device CUDA ordinals ({exc}); falling back to index mapping")

    if requested_gpu < num_devices:
        return requested_gpu
    # Fewer EGL devices than GPUs (common: one headless device for the node).
    # Rendering then shares that device; it is a slowdown, not a failure.
    log.warning(
        f"GPU {requested_gpu} has no EGL device of its own ({num_devices} EGL device(s) available); "
        f"rendering on EGL device 0"
    )
    return 0


_SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "lavapipe")


def _abort(message: str) -> None:
    """Refuse to run rather than render through a backend nobody asked for."""
    log.error(message)
    raise SystemExit(1)


def _probe_gl(egl_context, device_index: int) -> str:
    """Open a throwaway context on the resolved device and report what drives it."""
    from OpenGL import GL

    context = egl_context.EGLGLContext(max_width=1, max_height=1, device_id=device_index)
    try:
        context.make_current()
        parts = []
        for name in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION"):
            value = GL.glGetString(getattr(GL, name))
            parts.append(value.decode() if isinstance(value, bytes) else str(value))
        return " | ".join(parts)
    finally:
        context.free()


def _silence_teardown_errors(egl_context) -> None:
    """Make EGL/MuJoCo context teardown at interpreter shutdown non-fatal.

    ``MjRenderContext.__del__`` frees its MuJoCo and EGL contexts. When it runs
    during interpreter shutdown the EGL display has usually been terminated
    already, so ``eglDestroyContext`` raises ``EGL_NOT_INITIALIZED`` and Python
    prints an "Exception ignored in" traceback per render context. The contexts
    are released by the driver either way, so the traceback is noise.
    """
    import robosuite.utils.binding_utils as binding_utils

    original_free = egl_context.EGLGLContext.free

    def free(self):
        try:
            original_free(self)
        except Exception as exc:  # display already terminated, thread gone, ...
            log.debug(f"Ignoring EGL context teardown error: {exc}")
        finally:
            # Clear the handle either way, so __del__ cannot retry a destroy
            # that already failed.
            self._context = None

    egl_context.EGLGLContext.free = free

    original_del = binding_utils.MjRenderContext.__del__

    def __del__(self):
        try:
            original_del(self)
        except Exception as exc:
            log.debug(f"Ignoring render context teardown error: {exc}")

    binding_utils.MjRenderContext.__del__ = __del__


def _preload_triton() -> None:
    """Load libtriton before Mesa pulls the system LLVM into the process.

    A software-rasterized EGL context loads ``dri/swrast_dri.so``, which in turn
    loads the distribution's ``libLLVM-15.so.1`` with global symbol visibility.
    ``triton._C.libtriton`` carries its own, much newer LLVM; when it is loaded
    *after* Mesa, its LLVM symbols bind to Mesa's copy and the interpreter dies
    with a segfault the moment torch imports triton (via ``torch._dynamo``,
    which torchvision pulls in). Loading triton first makes its symbols win.
    """
    try:
        import triton  # noqa: F401
    except Exception as exc:  # triton is optional (CPU-only installs, ...)
        log.debug(f"Could not preload triton before EGL init: {exc}")


def apply() -> None:
    """Install the device-aware EGL display selection. Idempotent."""
    global _applied
    if _applied:
        return

    _preload_triton()

    mujoco_gl = os.environ.get("MUJOCO_GL", "").lower().strip()
    if mujoco_gl and mujoco_gl != "egl":
        _abort(f"MUJOCO_GL={mujoco_gl!r}: MuJoCo would not render through EGL. Unset it or set MUJOCO_GL=egl.")

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    requested_gpu = _requested_gpu()

    try:
        import robosuite.renderers.context.egl_context as egl_context
        from mujoco.egl import egl_ext as EGL_EXT
    except ImportError as exc:
        _abort(f"Cannot import the robosuite EGL context ({exc}); no EGL rendering available.")

    devices = EGL_EXT.eglQueryDevicesEXT()
    device_index = _egl_device_index(requested_gpu, len(devices))

    def create_initialized_egl_device_display(device_id=0):
        """Open the EGL display on the resolved device.

        Same body as robosuite's, minus its environment-variable guessing:
        the device was already resolved above.
        """
        from OpenGL import error

        for device in EGL_EXT.eglQueryDevicesEXT()[device_index:device_index + 1]:
            display = EGL_EXT.eglGetPlatformDisplayEXT(EGL_EXT.EGL_PLATFORM_DEVICE_EXT, device, None)
            if display == EGL_EXT.EGL_NO_DISPLAY or EGL_EXT.eglGetError() != EGL_EXT.EGL_SUCCESS:
                continue
            try:
                initialized = EGL_EXT.eglInitialize(display, None, None)
            except error.GLError:
                continue
            if initialized == EGL_EXT.EGL_TRUE and EGL_EXT.eglGetError() == EGL_EXT.EGL_SUCCESS:
                return display
        return EGL_EXT.EGL_NO_DISPLAY

    egl_context.create_initialized_egl_device_display = create_initialized_egl_device_display

    # robosuite asserts MUJOCO_EGL_DEVICE_ID is a substring of
    # CUDA_VISIBLE_DEVICES at import time, and its own selection is now unused.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ["VLS_EGL_DEVICE_ID"] = str(requested_gpu)

    if not devices:
        _abort("No EGL devices found; MuJoCo cannot render through EGL in this environment.")

    # robosuite picks its GL backend at import time from MUJOCO_GL; make sure the
    # choice really is EGL before any environment is built.
    import robosuite.utils.binding_utils as binding_utils

    backend = binding_utils.GLContext.__name__
    if backend != "EGLGLContext":
        _abort(f"MuJoCo render backend is {backend}, not EGL (MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}).")

    _silence_teardown_errors(egl_context)

    _applied = True
    driver = _probe_gl(egl_context, device_index)
    log.info(
        f"MuJoCo render device: EGL device {device_index} of {len(devices)} (GPU {requested_gpu}) | {driver}"
    )
    if any(name in driver.lower() for name in _SOFTWARE_RENDERERS):
        log.warning(
            "EGL is served by a software rasterizer (no NVIDIA EGL vendor ICD installed): rendering runs "
            "on CPU, not on the GPU. Rendering throughput will not scale with more GPUs."
        )
