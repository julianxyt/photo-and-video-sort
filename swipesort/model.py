"""The preference model: a regularised logistic regression over image features.

Deliberately small. A few hundred swipes is all the training data this will
ever have, so the model has to work from cold, train in milliseconds on every
retrain, and never be so confident that it hides a photo you wanted. Fitting is
done with Newton-Raphson (IRLS), which converges in a handful of iterations at
these sizes and needs no learning-rate tuning.

Labels: ``love`` and ``keep`` are positives (love counts double), ``drop`` is
the negative, ``later`` is not a label at all.

Files with no image features - a video without ffmpeg, a HEIC without
pillow-heif, anything that would not decode - are still trained on and scored,
using their metadata alone. Their image columns are filled with the training
mean, which standardises to exactly zero, so those columns contribute nothing
for that row; a ``has_image`` indicator lets the model learn how such files
differ on their own. This is ordinary mean imputation with a missingness flag.
"""
from __future__ import annotations

import json
import math
import sqlite3
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from . import db
from .config import MIN_LABELS_TO_PREDICT
from .features import unpack

LABEL_WEIGHTS = {"love": 2.0, "keep": 1.0, "drop": 1.0}
POSITIVE_ACTIONS = {"love", "keep"}

META_FEATURE_NAMES = (
    "log_size", "is_video", "is_raw", "is_live", "is_screenshot", "is_facebook",
    "is_metadata", "log_duration", "hour_sin", "hour_cos", "month_sin",
    "month_cos", "has_date", "log_dup_group", "in_dup_group", "log_pixels",
    "aspect", "has_image",
)


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #

def meta_features(row: Any, dup_sizes: dict[int, int] | None = None) -> np.ndarray:
    """Non-image signals that matter for triage, drawn from the index row."""
    get = row.__getitem__ if isinstance(row, sqlite3.Row) else row.get
    def field(name, default=None):
        try:
            value = get(name)
        except (KeyError, IndexError):
            return default
        return default if value is None else value

    bucket = field("bucket", "") or ""
    kind = field("kind", "") or ""
    size = float(field("size_bytes", 0) or 0)
    duration = float(field("duration_s", 0) or 0)
    width = float(field("width", 0) or 0)
    height = float(field("height", 0) or 0)

    taken = field("taken_at")
    hour = month = None
    if taken:
        try:
            stamp = datetime.fromisoformat(str(taken))
            hour, month = stamp.hour, stamp.month
        except ValueError:
            pass

    group = field("dup_group")
    group_size = (dup_sizes or {}).get(group, 1) if group is not None else 1

    return np.array([
        math.log1p(size) / 20.0,
        1.0 if kind == "video" else 0.0,
        1.0 if kind == "raw" else 0.0,
        1.0 if kind == "live" else 0.0,
        1.0 if bucket == "Screenshots" else 0.0,
        1.0 if bucket == "Facebook" else 0.0,
        1.0 if bucket == "Metadata" else 0.0,
        math.log1p(duration) / 10.0,
        math.sin(2 * math.pi * hour / 24) if hour is not None else 0.0,
        math.cos(2 * math.pi * hour / 24) if hour is not None else 0.0,
        math.sin(2 * math.pi * month / 12) if month is not None else 0.0,
        math.cos(2 * math.pi * month / 12) if month is not None else 0.0,
        1.0 if taken else 0.0,
        math.log1p(group_size) / 3.0,
        1.0 if group_size > 1 else 0.0,
        math.log1p(width * height) / 20.0,
        (width / height) if height else 0.0,
        1.0 if field("features") is not None else 0.0,
    ], dtype=np.float32)


def row_vector(row: Any, dup_sizes: dict[int, int] | None = None,
               image_dim: int = 0) -> np.ndarray:
    """Image features (``image_dim`` wide) followed by metadata features.

    A row with no image features - or ones of a different width, left over
    from another backend - gets NaNs in the image columns. The model imputes
    them; see the module docstring.
    """
    blob = row["features"] if isinstance(row, sqlite3.Row) else row.get("features")
    image = unpack(blob) if blob is not None else None
    usable = image is not None and image.size == image_dim and image_dim > 0
    if not usable:
        image = np.full(image_dim, np.nan, dtype=np.float32)
    meta = meta_features(row, dup_sizes)
    # has_image (the last metadata column) must say whether the image columns
    # are real, which only this function knows: a vector from another backend
    # is present in the database but not usable here.
    meta[-1] = 1.0 if usable else 0.0
    return np.concatenate([image, meta])


def image_width(conn: sqlite3.Connection) -> int:
    """Width of the library's image feature vectors; 0 if none have any yet.

    If two backends' vectors are both present - say a partial
    ``ingest --backend clip`` over a classic library - the more common width
    wins and the rest are treated as having no preview until the next ingest
    re-extracts them. Refusing to train instead would fail every retrain, and
    retraining happens inside the swipe request.
    """
    # Not aliased "width": media has a width column (pixels), and SQLite
    # resolves GROUP BY names against table columns before result aliases.
    top = conn.execute(
        "SELECT length(features) / 4 AS dims, COUNT(*) AS n FROM media "
        "WHERE features IS NOT NULL AND missing=0 "
        "GROUP BY dims ORDER BY n DESC, dims DESC LIMIT 1"
    ).fetchone()
    return int(top["dims"]) if top else 0


def dup_group_sizes(conn: sqlite3.Connection) -> dict[int, int]:
    return {
        r["dup_group"]: r["n"]
        for r in conn.execute(
            "SELECT dup_group, COUNT(*) AS n FROM media "
            "WHERE dup_group IS NOT NULL AND missing=0 GROUP BY dup_group"
        )
    }


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

@dataclass
class TrainReport:
    n_labels: int
    n_positive: int
    n_without_image: int
    n_features: int
    feat_kind: str
    holdout_accuracy: float | None
    holdout_auc: float | None
    baseline_accuracy: float | None
    trained_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PreferenceModel:
    """L2-regularised logistic regression with feature standardisation."""

    def __init__(self, weights: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray,
                 feat_kind: str) -> None:
        self.weights = weights
        self.bias = bias
        self.mean = mean
        self.scale = scale
        self.feat_kind = feat_kind

    # -- fitting ---------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        feat_kind: str = "unknown",
        l2: float = 1.0,
        iterations: int = 30,
    ) -> "PreferenceModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)

        # Columns that are NaN for some rows (image features of files with no
        # preview) are imputed with the mean of the rows that do have them. A
        # column with no observed values at all has nothing to learn from.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(X, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        X = np.where(np.isnan(X), mean, X)
        scale = X.std(axis=0)
        scale[scale < 1e-6] = 1.0
        Z = np.hstack([(X - mean) / scale, np.ones((X.shape[0], 1))])

        theta = np.zeros(Z.shape[1])
        penalty = np.eye(Z.shape[1]) * l2
        penalty[-1, -1] = 0.0  # never regularise the intercept

        for _ in range(iterations):
            p = _sigmoid(Z @ theta)
            gradient = Z.T @ (w * (p - y)) + penalty @ theta
            s = np.clip(w * p * (1 - p), 1e-6, None)
            hessian = (Z * s[:, None]).T @ Z + penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            theta -= step
            if np.max(np.abs(step)) < 1e-7:
                break

        return cls(theta[:-1], float(theta[-1]), mean, scale, feat_kind)

    # -- inference -------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(keep) for each row of X."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X = np.where(np.isnan(X), self.mean, X)
        Z = (X - self.mean) / self.scale
        return _sigmoid(Z @ self.weights + self.bias)

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x)[0])

    # -- persistence ------------------------------------------------------ #

    def to_json(self) -> str:
        return json.dumps({
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feat_kind": self.feat_kind,
        })

    @classmethod
    def from_json(cls, payload: str) -> "PreferenceModel":
        data = json.loads(payload)
        return cls(
            np.array(data["weights"], dtype=np.float64),
            float(data["bias"]),
            np.array(data["mean"], dtype=np.float64),
            np.array(data["scale"], dtype=np.float64),
            data.get("feat_kind", "unknown"),
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -60, 60)))


def auc_score(y: Sequence[float], p: Sequence[float]) -> float | None:
    """Rank-based AUC; None when one class is missing."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Training against the index
# --------------------------------------------------------------------------- #

def train(conn: sqlite3.Connection, *, l2: float = 1.0, seed: int = 0) -> TrainReport | None:
    """Fit on every labelled row, store the model, and report holdout quality.

    Returns None while there are too few labels to fit anything meaningful.
    """
    rows = db.labelled_rows(conn)
    if len(rows) < MIN_LABELS_TO_PREDICT:
        return None

    sizes = dup_group_sizes(conn)
    image_dim = image_width(conn)
    feat_kinds = {r["feat_kind"] for r in rows if r["feat_kind"]}
    vectors, labels, weights = [], [], []
    without_image = 0
    for row in rows:
        vectors.append(row_vector(row, sizes, image_dim))
        labels.append(1.0 if row["action"] in POSITIVE_ACTIONS else 0.0)
        weights.append(LABEL_WEIGHTS.get(row["action"], 1.0))
        without_image += row["features"] is None

    X = np.vstack(vectors)
    y = np.array(labels)
    w = np.array(weights)

    if len(set(y.tolist())) < 2:
        return None

    holdout_acc = holdout_auc = baseline = None
    if len(y) >= 40:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = max(8, int(len(y) * 0.2))
        test, train_idx = idx[:cut], idx[cut:]
        if len(set(y[train_idx].tolist())) == 2 and len(set(y[test].tolist())) == 2:
            probe = PreferenceModel.fit(X[train_idx], y[train_idx], w[train_idx], l2=l2)
            p = probe.predict(X[test])
            holdout_acc = float(((p >= 0.5).astype(float) == y[test]).mean())
            holdout_auc = auc_score(y[test], p)
            majority = 1.0 if y[train_idx].mean() >= 0.5 else 0.0
            baseline = float((y[test] == majority).mean())

    if not feat_kinds:
        feat_kind = "metadata-only"
    else:
        feat_kind = next(iter(feat_kinds)) if len(feat_kinds) == 1 else "mixed"
    model = PreferenceModel.fit(X, y, w, feat_kind=feat_kind, l2=l2)
    db.set_meta(conn, "model", model.to_json())

    report = TrainReport(
        n_labels=len(y),
        n_positive=int(y.sum()),
        n_without_image=without_image,
        n_features=X.shape[1],
        feat_kind=feat_kind,
        holdout_accuracy=holdout_acc,
        holdout_auc=holdout_auc,
        baseline_accuracy=baseline,
        trained_at=db.utcnow(),
    )
    db.set_meta(conn, "model_report", report.as_dict())
    conn.commit()
    return report


def load(conn: sqlite3.Connection) -> PreferenceModel | None:
    payload = db.get_meta(conn, "model")
    if not payload:
        return None
    try:
        return PreferenceModel.from_json(
            payload if isinstance(payload, str) else json.dumps(payload)
        )
    except Exception:
        return None


def score_all(conn: sqlite3.Connection, model: PreferenceModel | None = None) -> int:
    """Refresh ``media.score`` for every indexed file. Returns the count."""
    model = model or load(conn)
    if model is None:
        return 0
    image_dim = model.weights.size - len(META_FEATURE_NAMES)
    if image_dim < 0:
        return 0  # a model saved by an older version; the next retrain replaces it
    sizes = dup_group_sizes(conn)
    rows = conn.execute("SELECT * FROM media WHERE missing=0").fetchall()
    if not rows:
        return 0
    vectors = np.vstack([row_vector(row, sizes, image_dim) for row in rows])
    probabilities = model.predict(vectors)
    conn.executemany(
        "UPDATE media SET score=? WHERE id=?",
        [(float(p), row["id"]) for p, row in zip(probabilities, rows)],
    )
    conn.commit()
    return len(rows)
