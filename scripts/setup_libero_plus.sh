#!/usr/bin/env bash
# Fetch everything the libero_plus backend needs that `uv sync` cannot install:
#
#   1. LIBERO-plus benchmark code  -> third_party/libero_plus (pinned commit)
#   2. LIBERO-plus asset pack      -> third_party/libero_plus/libero/libero/assets (~6 GB zip)
#   3. ImageMagick C library       -> .deps/imagemagick (skipped if the system has it)
#
# No root needed. Idempotent: each step is skipped when its result is in place.
# Run `uv sync --extra libero` first; step 2 uses the venv's `hf` CLI.
#
# Usage: bash scripts/setup_libero_plus.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_PLUS_DIR="$ROOT/third_party/libero_plus"
LIBERO_PLUS_REPO="https://github.com/sylvestf/LIBERO-plus.git"
LIBERO_PLUS_COMMIT="4976dc30028e805ff8094b55501d532c48fec182"
ASSETS_DIR="$LIBERO_PLUS_DIR/libero/libero/assets"
DEPS_DIR="$ROOT/.deps"
MAGICK_PREFIX="$DEPS_DIR/imagemagick"

step() { printf '\n==> %s\n' "$*"; }

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "No .venv found. Run 'uv sync --extra libero' from $ROOT first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
step "LIBERO-plus code ($LIBERO_PLUS_DIR)"
if [ ! -d "$LIBERO_PLUS_DIR/.git" ]; then
    git clone "$LIBERO_PLUS_REPO" "$LIBERO_PLUS_DIR"
    git -C "$LIBERO_PLUS_DIR" checkout --quiet "$LIBERO_PLUS_COMMIT"
fi
head_commit="$(git -C "$LIBERO_PLUS_DIR" rev-parse HEAD)"
if [ "$head_commit" = "$LIBERO_PLUS_COMMIT" ]; then
    echo "at pinned commit $LIBERO_PLUS_COMMIT"
else
    echo "WARNING: checkout is at $head_commit, not the tested $LIBERO_PLUS_COMMIT; leaving it as is." >&2
fi

# ---------------------------------------------------------------------------
step "LIBERO-plus assets ($ASSETS_DIR)"
if [ -d "$ASSETS_DIR/scenes" ] && [ -d "$ASSETS_DIR/new_objects" ]; then
    echo "already present"
else
    download_dir="$DEPS_DIR/downloads"
    mkdir -p "$download_dir"
    "$ROOT/.venv/bin/hf" download Sylvest/LIBERO-plus assets.zip \
        --repo-type dataset --local-dir "$download_dir"

    # The zip does not unpack to assets/ directly: every entry sits under a long
    # build-machine prefix (inspire/hdd/.../LIBERO-plus-0/assets/), so locate the
    # assets dir after extraction instead of trusting the archive layout.
    unpack_dir="$download_dir/assets_unpacked"
    rm -rf "$unpack_dir"
    mkdir -p "$unpack_dir"
    if command -v unzip >/dev/null; then
        unzip -q "$download_dir/assets.zip" -d "$unpack_dir"
    else
        "$ROOT/.venv/bin/python" -m zipfile -e "$download_dir/assets.zip" "$unpack_dir"
    fi
    extracted="$(find "$unpack_dir" -type d -name assets -exec test -d '{}/scenes' \; -print -quit)"
    if [ -z "$extracted" ]; then
        echo "Could not find an assets/ directory with scenes/ inside assets.zip" >&2
        exit 1
    fi
    mkdir -p "$ASSETS_DIR"
    cp -a "$extracted/." "$ASSETS_DIR/"
    rm -rf "$unpack_dir" "$download_dir/assets.zip"
    echo "installed"
fi

# ---------------------------------------------------------------------------
step "ImageMagick for Wand"
if [ -n "${MAGICK_HOME:-}" ]; then
    echo "using MAGICK_HOME=$MAGICK_HOME"
elif compgen -G "$MAGICK_PREFIX/lib/libMagickWand*.so*" >/dev/null; then
    echo "already installed in $MAGICK_PREFIX"
elif ldconfig -p 2>/dev/null | grep -q libMagickWand; then
    echo "found system libMagickWand"
else
    # Unpack the distribution's own packages rather than a conda-forge build:
    # conda's ImageMagick links newer GLib/libstdc++/X11 than the system copies
    # torch and Mesa have already loaded when LIBERO imports wand, and dlopen
    # then fails with undefined symbols. apt can fetch without root when given
    # private state/cache dirs; only packages not already installed are fetched.
    if ! command -v apt-get >/dev/null || ! command -v dpkg-deb >/dev/null; then
        echo "No libMagickWand found and this is not a Debian/Ubuntu system." >&2
        echo "Install ImageMagick with your package manager, or set MAGICK_HOME." >&2
        exit 1
    fi
    apt_dir="$DEPS_DIR/apt"
    rm -rf "$apt_dir"
    mkdir -p "$apt_dir/lists/partial" "$apt_dir/cache/archives/partial" "$apt_dir/debs" "$apt_dir/root"
    apt_opts=(-o "Dir::State::Lists=$apt_dir/lists" -o "Dir::Cache=$apt_dir/cache" -o Debug::NoLocking=1)
    # Docker base images ship an apt post-update hook that cleans the system
    # cache; it cannot without root and only prints noise, so drop that line.
    apt-get "${apt_opts[@]}" update -qq 2> >(grep -v '/var/cache/apt' >&2)

    wanted=()
    for pkg in $(apt-cache "${apt_opts[@]}" depends --recurse --no-recommends --no-suggests \
                   --no-conflicts --no-breaks --no-replaces --no-enhances libmagickwand-6.q16-6 \
                 | grep -v '^[ <]' | sort -u); do
        case "$pkg" in fonts-*|ttf-*) continue ;; esac  # only needed to render text
        dpkg -s "$pkg" 2>/dev/null | grep -q '^Status: install ok installed' || wanted+=("$pkg")
    done
    echo "fetching: ${wanted[*]}"
    (cd "$apt_dir/debs" && apt-get "${apt_opts[@]}" download "${wanted[@]}")
    for deb in "$apt_dir"/debs/*.deb; do
        dpkg-deb -x "$deb" "$apt_dir/root"
    done

    wand_lib="$(find "$apt_dir/root/usr/lib" -name 'libMagickWand-*.so.*' -print -quit)"
    rm -rf "$MAGICK_PREFIX"
    mkdir -p "$MAGICK_PREFIX/lib" "$MAGICK_PREFIX/etc"
    cp -a "$(dirname "$wand_lib")/." "$MAGICK_PREFIX/lib/"
    cp -a "$apt_dir"/root/etc/ImageMagick-* "$MAGICK_PREFIX/etc/"

    # The .deb libraries expect to live in /usr/lib and carry no RPATH; let them
    # find their unpacked siblings (libMagickCore, liblqr, libfftw3, ...).
    find "$MAGICK_PREFIX/lib" -maxdepth 1 -type f -name '*.so*' \
        -exec uvx --quiet patchelf --set-rpath '$ORIGIN' {} \;
    rm -rf "$apt_dir"
    echo "installed into $MAGICK_PREFIX"
fi

# ---------------------------------------------------------------------------
step "Verifying"
cd "$ROOT"
# Same import order as main.py: torch and the EGL renderer load their system
# libraries before LIBERO-plus pulls in wand.
.venv/bin/python - <<'EOF'
import numpy as np
import torch  # noqa: F401
from patches import imagemagick, mujoco_egl
mujoco_egl.apply()
imagemagick.apply()
from wand.image import Image
with Image.from_array(np.zeros((16, 16, 3), dtype=np.uint8)) as image:
    Image(blob=image.make_blob("png"), format="png").close()
import robosuite, mujoco, bddl  # noqa: F401
print("wand (with PNG coder), robosuite, mujoco and bddl load fine")
EOF
echo "LIBERO-plus setup complete."
