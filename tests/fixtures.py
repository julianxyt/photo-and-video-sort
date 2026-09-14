"""Build a synthetic photo library so the tests never need real photos.

Two visually distinct populations stand in for "photos you keep" and "photos
you bin": bright, sharp, colourful frames versus dark, soft, low-contrast ones.
Plus the awkward cases the sorter has to handle - burst near-duplicates, exact
byte duplicates, a screenshot, a dated filename, and a non-media file.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(path, quality=92)
    return path


def _scene(rng: np.random.Generator, *, brightness: float, texture: float,
           blobs: int = 5) -> np.ndarray:
    """A smooth, photo-like scene: a few soft colour blobs plus fine detail.

    Real photographs are mostly low frequency with texture on top, which is
    what perceptual hashing and the sharpness features are tuned for. White
    noise would be neither.
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    image = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for _ in range(blobs):
        cx, cy = rng.random(2)
        sx, sy = rng.random(2) * 0.3 + 0.12
        blob = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
        image += blob[..., None].astype(np.float32) * rng.random(3).astype(np.float32)
    image /= max(float(image.max()), 1e-6)
    image *= brightness
    if texture:
        image += rng.normal(0, texture, image.shape).astype(np.float32)
    return np.clip(image, 0, 1).astype(np.float32)


def keeper(rng: np.random.Generator) -> np.ndarray:
    """Bright, saturated, plenty of fine detail - a photo worth keeping."""
    return _scene(rng, brightness=0.92, texture=0.055)


def junk(rng: np.random.Generator) -> np.ndarray:
    """Dark, soft and flat - an underexposed blurry mistake."""
    scene = _scene(rng, brightness=0.13, texture=0.0, blobs=2)
    # Block-average it so there is no high-frequency detail left at all.
    blocks = scene.reshape(SIZE // 16, 16, SIZE // 16, 16, 3).mean(axis=(1, 3))
    return np.repeat(np.repeat(blocks, 16, axis=0), 16, axis=1)


def burst(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """One scene shot n times, a hair apart - the travel backlog special."""
    base = keeper(rng)
    frames = [base]
    for _ in range(n - 1):
        shifted = np.roll(base, int(rng.integers(1, 4)), axis=1)
        frames.append(
            np.clip(shifted + rng.normal(0, 0.006, base.shape), 0, 1).astype(np.float32)
        )
    return frames


def build_library(root: Path, *, keepers: int = 30, junk_count: int = 30, seed: int = 7) -> dict:
    """Populate ``root`` and return a manifest of what was written."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {"keep": [], "drop": [], "burst": [], "other": []}

    for i in range(keepers):
        path = _save(keeper(rng), root / "Camera" / f"IMG_2019{(i % 12) + 1:02d}15_{i:04d}.jpg")
        manifest["keep"].append(str(path))

    for i in range(junk_count):
        path = _save(junk(rng), root / "Camera" / f"IMG_2021{(i % 12) + 1:02d}03_{i:04d}.jpg")
        manifest["drop"].append(str(path))

    for i, frame in enumerate(burst(rng, 5)):
        path = _save(frame, root / "Trip" / f"DSC_{9000 + i}.jpg")
        manifest["burst"].append(str(path))

    # An exact duplicate: same bytes, different name and folder.
    source = Path(manifest["keep"][0])
    copy = root / "Backup" / "copy-of-first.jpg"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, copy)
    manifest["other"].append(str(copy))

    # Special-cased names and a non-media file.
    manifest["other"].append(str(_save(junk(rng), root / "Screenshot_20220104_112233.png")))
    manifest["other"].append(str(_save(keeper(rng), root / "received_1234.jpg")))
    sidecar = root / "Camera" / "notes.txt"
    sidecar.write_text("not media", encoding="utf-8")
    manifest["other"].append(str(sidecar))

    return manifest
