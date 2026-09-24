"""Turn an image into a vector the preference model can learn from.

Two backends:

``classic`` (default)
    ~100 hand-built numbers - colour distribution, exposure, sharpness, edge
    density, and a 3x3 composition grid. Needs only numpy and Pillow, runs at
    a few hundred files a second, and is very good at what most storage
    cleaning actually is: finding the blurry, dark, badly framed ones.

``clip``
    OpenAI CLIP ViT-B/32 image embeddings, if torch and open_clip happen to be
    installed. Semantic rather than statistical, so it can learn "I keep
    mountains and street food, I bin whiteboards and parking receipts".

The backend name is stored alongside every vector, so switching backends
re-extracts rather than mixing incompatible feature spaces.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import ffmpeg_path

CLASSIC_KIND = "classic-v1"
CLIP_KIND = "clip-vitb32"


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def _register_heif() -> bool:
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


HEIF_AVAILABLE = _register_heif()


def rawpy_available() -> bool:
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


# EXIF orientations that turn the picture a quarter turn, swapping width/height.
_QUARTER_TURNS = {5, 6, 7, 8}


def open_media_image(path: Path, max_edge: int | None = None):
    """Open any supported file as ``(upright RGB Pillow image, (width, height))``.

    Returns None when the file cannot be turned into pixels. This is the single
    place every file becomes pixels, so thumbnails, hashes and features always
    agree on what a file looks like:

    * videos - a frame from ffmpeg (None without it)
    * RAW    - the camera's embedded JPEG preview via rawpy, else a full decode
    * others - Pillow, with HEIC support when pillow-heif is installed

    ``max_edge`` lets JPEG decoding skip resolution that will be thrown away.
    The returned size is the file's true upright resolution regardless - the
    image itself may be smaller. EXIF orientation is applied, so a portrait
    phone photo stored sideways comes out the right way up.
    """
    from PIL import Image, ImageOps

    from .classify import RAW_EXTS, VIDEO_EXTS

    ext = path.suffix.lower()
    try:
        if ext in VIDEO_EXTS:
            image = _video_frame(path)
            if image is None:
                return None
            return ImageOps.exif_transpose(image).convert("RGB"), image.size
        if ext in RAW_EXTS:
            opened = _open_raw(path, max_edge)
            if opened is None:
                return None
            image, size = opened
            return ImageOps.exif_transpose(image).convert("RGB"), size
        # Closed deterministically: on Windows an open handle stops `apply`
        # from moving the file, and garbage collection is not a guarantee.
        with Image.open(path) as image:
            size = image.size  # before draft() shrinks it
            if max_edge:
                image.draft("RGB", (max_edge, max_edge))
            if image.getexif().get(0x0112) in _QUARTER_TURNS:
                size = (size[1], size[0])
            return ImageOps.exif_transpose(image).convert("RGB"), size
    except Exception:
        return None


def raw_preview_jpeg(path: Path) -> bytes | None:
    """The JPEG a camera embeds inside its RAW file, if it has one.

    Almost every camera writes one (Fuji RAFs always do) - it is what the
    camera's own screen shows, with the film simulation applied. It is far
    cheaper than demosaicing the sensor data, and it carries the camera's EXIF.
    """
    try:
        import rawpy
    except ImportError:
        return None
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
    except Exception:
        return None
    if thumb.format == rawpy.ThumbFormat.JPEG:
        return bytes(thumb.data)
    return None


def _open_raw(path: Path, max_edge: int | None):
    """Decode a RAW file to ``(image, true upright size)``.

    Embedded preview first, half-size demosaic as the fallback. Returns None
    when rawpy is not installed: Pillow can open some TIFF-based RAW formats,
    but what it sees is the undemosaiced sensor mosaic, and features computed
    from that would be confidently wrong - worse than none.
    """
    import io

    from PIL import Image

    try:
        import rawpy
    except ImportError:
        return None

    with rawpy.imread(str(path)) as raw:
        # LibRaw reports the sensor's output size and how the camera was held;
        # flips 5 and 6 are the quarter turns.
        sizes = raw.sizes
        size = (sizes.width, sizes.height)
        if sizes.flip in (5, 6):
            size = (size[1], size[0])
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            image = Image.open(io.BytesIO(bytes(thumb.data)))
            if max_edge:
                image.draft("RGB", (max_edge, max_edge))
            image.load()
            return image, size
        if thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
            return Image.fromarray(thumb.data), size
        # No usable preview: demosaic at half resolution, which is plenty for a
        # 720px thumbnail and several times faster than a full-size decode.
        # postprocess() applies the camera orientation itself.
        rgb = raw.postprocess(half_size=True, use_camera_wb=True)
    return Image.fromarray(rgb), size


def _video_frame(path: Path):
    """Grab a representative frame with ffmpeg; None when ffmpeg is absent."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return None
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.jpg"
        for seek in ("00:00:02", "00:00:00"):
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", seek,
                 "-i", str(path), "-frames:v", "1", "-q:v", "4", "-y", str(out)],
                capture_output=True,
                timeout=60,
                check=False,
            )
            if proc.returncode == 0 and out.exists() and out.stat().st_size:
                with Image.open(out) as img:
                    return img.copy()
    return None


# --------------------------------------------------------------------------- #
# Small numpy image ops (kept local so the package has no OpenCV dependency)
# --------------------------------------------------------------------------- #

def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV. H in [0,1), S and V in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    span = maxc - minc
    value = maxc
    sat = np.where(maxc > 0, span / np.maximum(maxc, 1e-8), 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rc = (maxc - r) / np.maximum(span, 1e-8)
        gc = (maxc - g) / np.maximum(span, 1e-8)
        bc = (maxc - b) / np.maximum(span, 1e-8)
    hue = np.where(maxc == r, bc - gc, np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = (hue / 6.0) % 1.0
    hue = np.where(span < 1e-8, 0.0, hue)
    return np.stack([hue, sat, value], axis=-1).astype(np.float32)


def laplacian_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap blur detector."""
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    return np.sqrt(gx * gx + gy * gy)


# --------------------------------------------------------------------------- #
# Classic backend
# --------------------------------------------------------------------------- #

class ClassicExtractor:
    """Statistical description of colour, exposure, sharpness and composition."""

    kind = CLASSIC_KIND

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        gray = to_gray(rgb)
        hsv = rgb_to_hsv(rgb)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        edges = sobel_magnitude(gray)

        parts: list[np.ndarray] = []

        # Colour distribution: marginal histograms plus a coarse joint one.
        parts.append(_hist(hue, 12, weights=sat))       # hue weighted by saturation
        parts.append(_hist(sat, 8))
        parts.append(_hist(val, 8))
        joint, _ = np.histogramdd(
            rgb.reshape(-1, 3), bins=(4, 3, 3), range=((0, 1), (0, 1), (0, 1))
        )
        parts.append((joint / max(joint.sum(), 1)).flatten().astype(np.float32))

        # Exposure and contrast.
        parts.append(np.array([
            gray.mean(), gray.std(),
            np.percentile(gray, 5), np.percentile(gray, 95),
            float(np.percentile(gray, 95) - np.percentile(gray, 5)),
            float((gray < 0.06).mean()),   # crushed shadows
            float((gray > 0.97).mean()),   # blown highlights
        ], dtype=np.float32))

        # Saturation and colourfulness (Hasler-Susstrunk).
        rg = rgb[..., 0] - rgb[..., 1]
        yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
        colourfulness = float(
            np.sqrt(rg.std() ** 2 + yb.std() ** 2)
            + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        )
        warm = float((rgb[..., 0] > rgb[..., 2]).mean())
        parts.append(np.array([sat.mean(), sat.std(), colourfulness, warm], dtype=np.float32))

        # Sharpness overall and in the centre (a sharp subject on a soft
        # background is a keeper; uniformly soft is a blurry mistake).
        centre = gray[gray.shape[0] // 4: -gray.shape[0] // 4,
                      gray.shape[1] // 4: -gray.shape[1] // 4]
        parts.append(np.array([
            np.log1p(laplacian_var(gray) * 1000.0),
            np.log1p(laplacian_var(centre) * 1000.0),
            float(edges.mean()),
            float((edges > 0.1).mean()),
            _entropy(gray),
        ], dtype=np.float32))

        # Composition: brightness and edge energy over a 3x3 grid, which is
        # what separates "subject on the thirds" from "sky and a thumb".
        parts.append(_grid_stats(gray))
        parts.append(_grid_stats(edges))

        vector = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def dim(self) -> int:
        return self.extract(np.zeros((32, 32, 3), dtype=np.float32)).size


def _hist(channel: np.ndarray, bins: int, weights: np.ndarray | None = None) -> np.ndarray:
    hist, _ = np.histogram(
        channel, bins=bins, range=(0.0, 1.0),
        weights=None if weights is None else weights,
    )
    total = hist.sum()
    return (hist / total if total else hist).astype(np.float32)


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _grid_stats(plane: np.ndarray) -> np.ndarray:
    """Mean of each cell of a 3x3 grid, as a flat 9-vector."""
    rows = np.array_split(plane, 3, axis=0)
    cells = [block.mean() for row in rows for block in np.array_split(row, 3, axis=1)]
    return np.array(cells, dtype=np.float32)


# --------------------------------------------------------------------------- #
# CLIP backend (optional)
# --------------------------------------------------------------------------- #

class ClipExtractor:
    """CLIP ViT-B/32 image embeddings. Requires torch + open_clip."""

    kind = CLIP_KIND

    def __init__(self) -> None:
        import open_clip  # noqa: F401  (raises ImportError when unavailable)
        import torch

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self._model.eval()
        self.dim = 512

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        batch = self._preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)


def get_extractor(name: str = "auto"):
    """Build a feature extractor. ``auto`` prefers CLIP when it is installed."""
    if name in ("auto", "clip"):
        try:
            return ClipExtractor()
        except Exception:
            if name == "clip":
                raise RuntimeError(
                    "CLIP backend requested but unavailable. "
                    "Install it with: pip install open_clip_torch torch"
                ) from None
    return ClassicExtractor()


def pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
