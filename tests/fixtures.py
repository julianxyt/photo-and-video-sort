"""Build a synthetic photo library so the tests never need real photos.

Two visually distinct populations stand in for "photos you keep" and "photos
you bin": bright, sharp, colourful frames versus dark, soft, low-contrast ones.
Plus the awkward cases the sorter has to handle - burst near-duplicates, exact
byte duplicates, a screenshot, a dated filename, and a non-media file.
"""
from __future__ import annotations

import io
import shutil
import struct
from datetime import datetime
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


# --------------------------------------------------------------------------- #
# Files the default library does not contain: RAW and video
# --------------------------------------------------------------------------- #

def make_mp4(path: Path, *, taken: datetime | None = None, padding: int = 0) -> Path:
    """A minimal but valid MP4: ftyp + moov/mvhd, and optionally a ``free`` box.

    It has no video stream, so nothing can extract a frame from it - exactly
    what a video looks like on a machine without ffmpeg. ``padding`` grows the
    file, so tests can give the model a size signal to learn from.
    """
    taken = taken or datetime(2023, 5, 1, 10, 0, 0)
    created = int((taken - datetime(1904, 1, 1)).total_seconds())
    mvhd = b"mvhd" + bytes(4) + struct.pack(">I", created) + b"\0" * 80
    mvhd = struct.pack(">I", len(mvhd) + 4) + mvhd
    moov = struct.pack(">I", len(mvhd) + 8) + b"moov" + mvhd
    ftyp = struct.pack(">I", 16) + b"ftypisom" + bytes(4)
    free = struct.pack(">I", padding + 8) + b"free" + bytes(padding) if padding else b""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ftyp + moov + free)
    return path


def make_dng(path: Path, rgb: np.ndarray, *, preview: bool, taken: str | None = None) -> Path:
    """Write a real, minimal DNG that LibRaw (and so rawpy) can decode.

    DNG is a RAW format like Fuji's RAF, and swipesort handles both through the
    same rawpy calls, so this exercises the genuine RAW path without shipping a
    multi-megabyte camera file in the repository.

    The sensor data is an RGGB Bayer mosaic sampled from ``rgb``. With
    ``preview=True`` the file also embeds a JPEG preview the way cameras do,
    carrying EXIF DateTimeOriginal = ``taken``; the date lives *only* there, so
    a test can tell that it was read from the preview.
    """
    h, w, _ = rgb.shape
    scaled = rgb.astype(np.float32) / 255.0 * 60000.0
    bayer = np.zeros((h, w), dtype=np.uint16)
    bayer[0::2, 0::2] = scaled[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = scaled[0::2, 1::2, 1]   # G
    bayer[1::2, 0::2] = scaled[1::2, 0::2, 1]   # G
    bayer[1::2, 1::2] = scaled[1::2, 1::2, 2]   # B
    sensor = bayer.astype("<u2").tobytes()

    identity = [(1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1)]
    camera = {
        271: ("A", "swipesort"),                # Make
        272: ("A", "fixture"),                  # Model
        50706: ("B", [1, 4, 0, 0]),             # DNGVersion
        50708: ("A", "swipesort fixture"),      # UniqueCameraModel
        50721: ("SR", identity),                # ColorMatrix1
        50778: ("H", [21]),                     # CalibrationIlluminant1 = D65
    }
    raw_ifd = {
        254: ("I", [0]), 256: ("I", [w]), 257: ("I", [h]), 258: ("H", [16]),
        259: ("H", [1]),                        # uncompressed
        262: ("H", [32803]),                    # PhotometricInterpretation = CFA
        273: ("I", ["blob:sensor"]), 277: ("H", [1]), 278: ("I", [h]),
        279: ("I", [len(sensor)]), 284: ("H", [1]),
        33421: ("H", [2, 2]),                   # CFARepeatPatternDim
        33422: ("B", [0, 1, 1, 2]),             # CFAPattern = RGGB
        50717: ("I", [60000]),                  # WhiteLevel
    }
    blobs = {"sensor": sensor}

    if not preview:
        ifds = [{**raw_ifd, **camera}]
    else:
        exif = Image.Exif()
        if taken:
            exif.get_ifd(0x8769)[36867] = taken
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, "JPEG", quality=90, exif=exif)
        blobs["preview"] = buffer.getvalue()
        preview_ifd = {
            254: ("I", [1]),                    # a reduced-resolution preview
            256: ("I", [w]), 257: ("I", [h]), 258: ("H", [8, 8, 8]),
            259: ("H", [7]),                    # JPEG
            262: ("H", [6]),                    # YCbCr
            273: ("I", ["blob:preview"]), 277: ("H", [3]), 278: ("I", [h]),
            279: ("I", [len(blobs["preview"])]),
            330: ("I", ["ifd:1"]),              # SubIFDs -> the sensor data
            **camera,
        }
        ifds = [preview_ifd, raw_ifd]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_tiff(ifds, blobs))
    return path


def _encode(kind: str, values) -> tuple[int, int, bytes]:
    """TIFF (type, count, bytes) for a tag value."""
    if kind == "A":
        data = values.encode("ascii") + b"\0"
        return 2, len(data), data
    if kind == "B":
        return 1, len(values), bytes(values)
    if kind == "H":
        return 3, len(values), struct.pack(f"<{len(values)}H", *values)
    if kind == "I":
        return 4, len(values), struct.pack(f"<{len(values)}I", *values)
    if kind == "SR":
        return 10, len(values), b"".join(struct.pack("<ii", n, d) for n, d in values)
    raise ValueError(kind)


def _tiff(ifds: list[dict], blobs: dict[str, bytes]) -> bytes:
    """Little-endian TIFF from IFD dicts ``{tag: (kind, values)}``.

    LONG values may be the placeholders ``"blob:<name>"`` or ``"ifd:<n>"``,
    which resolve to that blob's or IFD's byte offset.
    """
    out = bytearray(8)
    offsets: dict[str, int] = {}
    for name, data in blobs.items():
        out += b"\0" * (len(out) % 2)
        offsets[f"blob:{name}"] = len(out)
        out += data

    def ifd_size(ifd: dict) -> int:
        extra = 0
        for kind, values in ifd.values():
            probe = values if kind == "A" else [0 if isinstance(v, str) else v for v in values]
            data = _encode(kind, probe)[2]
            if len(data) > 4:
                extra += len(data) + len(data) % 2
        return 2 + 12 * len(ifd) + 4 + extra

    position = len(out) + len(out) % 2
    for index, ifd in enumerate(ifds):
        offsets[f"ifd:{index}"] = position
        position += ifd_size(ifd)

    for index, ifd in enumerate(ifds):
        base = offsets[f"ifd:{index}"]
        out += b"\0" * (base - len(out))
        entries = bytearray(struct.pack("<H", len(ifd)))
        spill = bytearray()
        spill_at = base + 2 + 12 * len(ifd) + 4
        for tag in sorted(ifd):
            kind, values = ifd[tag]
            if kind != "A":
                values = [offsets[v] if isinstance(v, str) else v for v in values]
            typ, count, data = _encode(kind, values)
            if len(data) <= 4:
                entries += struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0")
            else:
                entries += struct.pack("<HHII", tag, typ, count, spill_at + len(spill))
                spill += data + b"\0" * (len(data) % 2)
        entries += struct.pack("<I", 0)
        out += entries + spill

    out[0:8] = b"II*\0" + struct.pack("<I", offsets["ifd:0"])
    return bytes(out)
