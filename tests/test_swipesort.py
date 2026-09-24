"""End-to-end and unit tests. Run with: python -m unittest discover -s tests"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from swipesort import apply as apply_mod, capture_date, classify, db, features, ingest, model, phash
from swipesort import queue as queue_mod
from swipesort.config import Library

from fixtures import build_library, make_dng, make_mp4


class ClassifyTests(unittest.TestCase):
    def test_buckets_follow_the_powershell_guard_order(self):
        self.assertEqual(classify.bucket_for("Screenshot_2019.png", ".png"), "Screenshots")
        self.assertEqual(classify.bucket_for("FB_IMG_1.jpg", ".jpg"), "Facebook")
        self.assertEqual(classify.bucket_for("received_1.jpg", ".jpg"), "Facebook")
        self.assertEqual(classify.bucket_for("clip.MP", ".mp"), "Live Photos")
        self.assertEqual(classify.bucket_for("notes.pdf", ".pdf"), "Metadata")
        self.assertEqual(classify.bucket_for("holiday.jpg", ".jpg"), "Photos")
        self.assertEqual(classify.bucket_for("holiday.mov", ".mov"), "Videos")
        self.assertEqual(classify.bucket_for("DSCF1.raf", ".raf"), "RAWs")

    def test_screenshot_rule_beats_extension(self):
        # A screenshot is a screenshot even though .png is also a photo type.
        self.assertEqual(classify.bucket_for("Screenshot.png", ".png"), "Screenshots")

    def test_nef_is_raw(self):
        self.assertEqual(classify.media_kind(".nef"), "raw")

    def test_year_from_filename(self):
        self.assertEqual(classify.year_from_filename("IMG_20190304_121314.jpg"), "2019")
        self.assertEqual(classify.year_from_filename("20211231.mp4"), "2021")
        self.assertIsNone(classify.year_from_filename("IMG_1234.jpg"))
        self.assertIsNone(classify.year_from_filename("19990101.jpg"))
        # Month 13 is not a date.
        self.assertIsNone(classify.year_from_filename("20191301.jpg"))

    def test_target_folder(self):
        self.assertEqual(classify.target_folder("Photos", "2019"), "2019 Photos")
        self.assertEqual(classify.target_folder("Photos", None), "Unclassified Photos")
        self.assertEqual(classify.target_folder("Screenshots", "2019"), "Screenshots")


class HashTests(unittest.TestCase):
    def test_near_duplicates_cluster_and_distinct_images_do_not(self):
        rng = np.random.default_rng(3)
        base = rng.random((64, 64))
        a = phash.dhash(base)
        b = phash.dhash(base + rng.normal(0, 0.003, base.shape))
        c = phash.dhash(rng.random((64, 64)))
        self.assertLessEqual(phash.hamming(a, b), 6)
        self.assertGreater(phash.hamming(a, c), 15)

        groups = phash.pack_groups(phash.group_near_duplicates([(1, a), (2, b), (3, c)], 6))
        self.assertEqual(groups.get(1), groups.get(2))
        self.assertNotIn(3, groups)

    def test_threshold_controls_strictness(self):
        rng = np.random.default_rng(5)
        base = rng.random((64, 64))
        a = phash.dhash(base)
        b = phash.dhash(base + rng.normal(0, 0.02, base.shape))
        distance = phash.hamming(a, b)
        loose = phash.group_near_duplicates([(1, a), (2, b)], distance)
        strict = phash.group_near_duplicates([(1, a), (2, b)], max(distance - 1, 0))
        self.assertEqual(len(loose), 2)
        self.assertEqual(strict, {})

    def test_items_without_a_hash_are_ignored(self):
        self.assertEqual(phash.group_near_duplicates([(1, None), (2, None)], 6), {})


class CaptureDateTests(unittest.TestCase):
    def test_quicktime_mvhd_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mp4(Path(tmp) / "clip.mp4", taken=datetime(2019, 6, 1))
            found, source = capture_date.capture_datetime(path)
        self.assertIsNotNone(found)
        self.assertEqual(found.year, 2019)
        self.assertEqual(source, "quicktime")

    def test_date_taken_beats_date_modified(self):
        # Cameras put DateTimeOriginal in the Exif sub-IFD; IFD0's DateTime is
        # when the file was last modified. An edited photo must still file
        # under the day it was taken, not the day it was edited.
        from PIL import Image

        exif = Image.Exif()
        exif[306] = "2024:12:25 18:00:00"
        exif.get_ifd(0x8769)[36867] = "2023:05:01 10:00:00"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edited.jpg"
            Image.new("RGB", (16, 16)).save(path, exif=exif)
            found, source = capture_date.capture_datetime(path)
        self.assertEqual(found, datetime(2023, 5, 1, 10, 0, 0))
        self.assertEqual(source, "exif")

    def test_modified_date_is_only_a_last_resort(self):
        from PIL import Image

        exif = Image.Exif()
        exif[306] = "2024:12:25 18:00:00"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.jpg"
            Image.new("RGB", (16, 16)).save(path, exif=exif)
            self.assertEqual(capture_date.from_pillow(path).year, 2024)

    def test_exif_datetime_variants(self):
        parse = capture_date._parse_exif_datetime
        self.assertEqual(parse("2023:05:01 10:00:00"), datetime(2023, 5, 1, 10))
        self.assertEqual(parse("2023:05:01 10:00:00.250").microsecond, 250000)
        self.assertEqual(parse("2023:05:01 10:00:00+05:30"), datetime(2023, 5, 1, 10))
        self.assertEqual(parse("2023-05-01T10:00:00Z"), datetime(2023, 5, 1, 10))
        self.assertEqual(parse("2023:05:01"), datetime(2023, 5, 1))
        for blank in ("", "0000:00:00 00:00:00", "    :  :     :  :  ", "garbage"):
            self.assertIsNone(parse(blank), blank)

    def test_unreadable_file_reports_no_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            found, source = capture_date.capture_datetime(path)
        self.assertIsNone(found)
        self.assertEqual(source, "none")

    def test_mtime_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            self.assertEqual(capture_date.capture_datetime(path)[1], "none")
            self.assertEqual(capture_date.capture_datetime(path, use_mtime=True)[1], "mtime")


def _two_tone(height: int = 96, width: int = 128) -> np.ndarray:
    """Left half red, right half noise - easy to recognise after a decode."""
    rgb = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    rgb[:, : width // 2] = [230, 40, 40]
    return rgb


class ImageLoadingTests(unittest.TestCase):
    def test_exif_orientation_is_applied(self):
        # Phones store portraits sideways and set Orientation=6 ("rotate 90").
        from PIL import Image

        exif = Image.Exif()
        exif[0x0112] = 6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portrait.jpg"
            Image.new("RGB", (64, 32), "white").save(path, exif=exif)
            image, size = features.open_media_image(path)
        self.assertEqual(image.size, (32, 64), "the pixels are turned upright")
        self.assertEqual(size, (32, 64), "and so is the reported size")

    def test_true_size_survives_draft_decoding(self):
        # draft() decodes a JPEG at reduced scale for speed. The recorded size
        # must still be the photo's real resolution, not the reduced one.
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.jpg"
            Image.new("RGB", (2400, 1600), "gray").save(path)
            image, size = features.open_media_image(path, max_edge=300)
        self.assertEqual(size, (2400, 1600))
        self.assertLess(image.size[0], 2400, "while decoding smaller")

    def test_unreadable_file_gives_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            self.assertIsNone(features.open_media_image(path))


class RawTests(unittest.TestCase):
    """RAW decoding through real LibRaw, on a generated DNG.

    Fuji RAFs take exactly the same code path - rawpy.imread, extract_thumb,
    postprocess - so a DNG proves the path without a camera file in the repo.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_rawpy_api_contract(self):
        # Everything swipesort calls. A rawpy release that renames any of these
        # should fail here, loudly, rather than silently skip every RAW file.
        import rawpy

        for name in ("imread", "ThumbFormat", "LibRawNoThumbnailError",
                     "LibRawUnsupportedThumbnailError"):
            self.assertTrue(hasattr(rawpy, name), name)
        self.assertTrue(hasattr(rawpy.ThumbFormat, "JPEG"))
        self.assertTrue(hasattr(rawpy.ThumbFormat, "BITMAP"))

    def test_embedded_preview_is_used(self):
        path = make_dng(self.tmp / "a.dng", _two_tone(), preview=True)
        image, size = features.open_media_image(path)
        self.assertEqual(size, (128, 96), "the sensor's size, from LibRaw")
        self.assertEqual(image.size, (128, 96), "the full-size preview JPEG")
        red = np.asarray(image)[:, :32].mean(axis=(0, 1))
        self.assertGreater(red[0], 180)
        self.assertLess(red[1], 90)

    def test_full_decode_when_there_is_no_preview(self):
        path = make_dng(self.tmp / "b.dng", _two_tone(), preview=False)
        image, size = features.open_media_image(path)
        self.assertEqual(size, (128, 96))
        self.assertEqual(image.size, (64, 48), "half-size demosaic")
        red = np.asarray(image)[:, :16].mean(axis=(0, 1))
        self.assertGreater(red[0], 180, "the colour survives the demosaic")
        self.assertLess(red[1], 90)

    def test_capture_date_comes_from_the_embedded_preview(self):
        path = make_dng(self.tmp / "c.dng", _two_tone(), preview=True,
                        taken="2021:07:14 09:30:00")
        with mock.patch("swipesort.capture_date.exiftool_path", return_value=None):
            found, source = capture_date.capture_datetime(path)
        self.assertEqual(found, datetime(2021, 7, 14, 9, 30))
        self.assertEqual(source, "raw-preview")

    def test_libraw_placeholder_is_not_mistaken_for_a_date(self):
        # With no date to find, LibRaw's own EXIF block says 1970-01-01. That
        # must read as "undated", not as a photo taken in 1970.
        path = make_dng(self.tmp / "undated.dng", _two_tone(), preview=True, taken=None)
        self.assertIsNotNone(features.raw_preview_jpeg(path), "a preview is there")
        self.assertIsNone(capture_date.from_raw_preview(path))

    def test_without_rawpy_raw_files_are_skipped_not_misread(self):
        # Pillow can open some TIFF-based RAWs, but it sees the raw sensor
        # mosaic. Features from that would be confidently wrong.
        path = make_dng(self.tmp / "d.dng", _two_tone(), preview=True)
        with mock.patch.dict(sys.modules, {"rawpy": None}):
            self.assertFalse(features.rawpy_available())
            self.assertIsNone(features.open_media_image(path))

    def test_corrupt_raw_gives_none(self):
        path = self.tmp / "broken.raf"
        path.write_bytes(b"FUJIFILMCCD-RAW " + bytes(512))
        self.assertIsNone(features.open_media_image(path))

    def test_ingest_previews_and_dates_raw_files(self):
        make_dng(self.tmp / "Fuji" / "DSCF0001.dng", _two_tone(), preview=True,
                 taken="2021:07:14 09:30:00")
        make_dng(self.tmp / "Fuji" / "DSCF0002.dng", _two_tone(), preview=False)
        library = Library.resolve(self.tmp)
        with mock.patch("swipesort.capture_date.exiftool_path", return_value=None):
            report = ingest.scan(library)
        self.assertEqual(report.featured, 2)
        self.assertEqual(report.gaps, [])

        conn = db.connect(library.db_path)
        self.addCleanup(conn.close)
        rows = {r["filename"]: r for r in conn.execute("SELECT * FROM media")}
        dated = rows["DSCF0001.dng"]
        self.assertEqual(dated["bucket"], "RAWs")
        self.assertEqual(dated["year"], "2021")
        self.assertEqual(dated["date_source"], "raw-preview")
        self.assertEqual((dated["width"], dated["height"]), (128, 96))
        self.assertIsNone(rows["DSCF0002.dng"]["year"], "no preview, so no date")
        for row in rows.values():
            self.assertIsNotNone(row["features"])
            self.assertTrue((library.thumb_dir / row["thumb"]).exists())


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = features.ClassicExtractor()

    def test_vectors_are_finite_and_fixed_width(self):
        rng = np.random.default_rng(0)
        cases = [
            rng.random((128, 128, 3)).astype(np.float32),
            np.zeros((128, 128, 3), dtype=np.float32),
            np.ones((128, 128, 3), dtype=np.float32),
        ]
        widths = set()
        for image in cases:
            vector = self.extractor.extract(image)
            widths.add(vector.size)
            self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(len(widths), 1)

    def test_sharpness_separates_sharp_from_blurry(self):
        rng = np.random.default_rng(1)
        sharp = rng.random((128, 128)).astype(np.float32)
        blurry = np.repeat(np.repeat(rng.random((8, 8)).astype(np.float32), 16, 0), 16, 1)
        self.assertGreater(features.laplacian_var(sharp), features.laplacian_var(blurry) * 10)

    def test_pack_roundtrip(self):
        vector = self.extractor.extract(np.random.default_rng(2).random((64, 64, 3)).astype(np.float32))
        self.assertTrue(np.allclose(features.unpack(features.pack(vector)), vector))


class ModelTests(unittest.TestCase):
    def test_learns_a_separable_boundary(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(1.2, 1, (120, 8)), rng.normal(-1.2, 1, (120, 8))])
        y = np.r_[np.ones(120), np.zeros(120)]
        fitted = model.PreferenceModel.fit(X[::2], y[::2])
        accuracy = ((fitted.predict(X[1::2]) >= 0.5) == y[1::2]).mean()
        self.assertGreater(accuracy, 0.85)

    def test_regularisation_keeps_separable_data_finite(self):
        X = np.vstack([np.full((20, 3), 5.0), np.full((20, 3), -5.0)])
        y = np.r_[np.ones(20), np.zeros(20)]
        fitted = model.PreferenceModel.fit(X, y, l2=1.0)
        self.assertTrue(np.isfinite(fitted.weights).all())
        self.assertLess(np.abs(fitted.weights).max(), 50)

    def test_json_roundtrip(self):
        rng = np.random.default_rng(4)
        X, y = rng.normal(size=(60, 5)), (rng.random(60) > 0.5).astype(float)
        fitted = model.PreferenceModel.fit(X, y)
        revived = model.PreferenceModel.from_json(fitted.to_json())
        self.assertTrue(np.allclose(fitted.predict(X), revived.predict(X)))

    def test_missing_image_columns_are_imputed(self):
        # Rows without image features carry NaN there; the model fills them with
        # the column mean, which standardises to zero, so they contribute nothing.
        rng = np.random.default_rng(6)
        X = rng.normal(size=(80, 6))
        y = (X[:, 0] + X[:, 4] > 0).astype(float)
        X[::3, :3] = np.nan          # a third of rows have no "image"
        fitted = model.PreferenceModel.fit(X, y)
        p = fitted.predict(X)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(fitted.mean).all())

        row = X[0].copy()            # a NaN row...
        filled = row.copy()
        filled[:3] = fitted.mean[:3]  # ...predicts exactly as if at the mean
        self.assertAlmostEqual(fitted.predict_one(row), fitted.predict_one(filled))

    def test_a_column_with_no_observations_is_inert(self):
        X = np.column_stack([np.full(40, np.nan), np.r_[np.ones(20), -np.ones(20)]])
        y = np.r_[np.ones(20), np.zeros(20)]
        fitted = model.PreferenceModel.fit(X, y)
        self.assertEqual(fitted.weights[0], 0.0)
        self.assertTrue(np.isfinite(fitted.predict(X)).all())

    def test_auc_edges(self):
        self.assertIsNone(model.auc_score([1, 1, 1], [0.1, 0.2, 0.3]))
        self.assertEqual(model.auc_score([0, 1], [0.1, 0.9]), 1.0)
        self.assertEqual(model.auc_score([1, 0], [0.1, 0.9]), 0.0)
        self.assertEqual(model.auc_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]), 0.5)


class LibraryTestCase(unittest.TestCase):
    """Shared setup: a real synthetic library, ingested once per test."""

    keepers = 26
    junk = 26

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manifest = build_library(self.tmp, keepers=self.keepers, junk_count=self.junk)
        self.library = Library.resolve(self.tmp)
        self.report = ingest.scan(self.library, progress=None)
        self.conn = db.connect(self.library.db_path)
        self.addCleanup(self.conn.close)

    def duplicate_pair(self):
        """The byte-identical pair, as (flagged_as_duplicate, kept)."""
        flagged = self.conn.execute(
            "SELECT * FROM media WHERE exact_dup_of IS NOT NULL"
        ).fetchall()
        self.assertEqual(len(flagged), 1, "fixture has exactly one identical pair")
        kept = self.conn.execute(
            "SELECT * FROM media WHERE id=?", (flagged[0]["exact_dup_of"],)
        ).fetchone()
        self.assertIsNotNone(kept)
        return flagged[0], kept

    def a_plain_keeper(self):
        """A keeper row that is not part of the byte-identical pair."""
        flagged, kept = self.duplicate_pair()
        row = self.conn.execute(
            "SELECT * FROM media WHERE filename LIKE 'IMG_2019%' AND id NOT IN (?, ?) "
            "ORDER BY rel_path LIMIT 1",
            (flagged["id"], kept["id"]),
        ).fetchone()
        self.assertIsNotNone(row)
        return row

    def swipe_all(self):
        """Label the fixture population the way a person would."""
        keep_paths = set(self.manifest["keep"])
        drop_paths = set(self.manifest["drop"])
        for row in self.conn.execute("SELECT id, path FROM media"):
            if row["path"] in keep_paths:
                db.record_decision(self.conn, row["id"], "keep")
            elif row["path"] in drop_paths:
                db.record_decision(self.conn, row["id"], "drop")


class IngestTests(LibraryTestCase):
    def test_indexes_media_and_skips_non_media(self):
        rows = self.conn.execute("SELECT * FROM media").fetchall()
        paths = {r["path"] for r in rows}
        self.assertNotIn(str(self.tmp / "Camera" / "notes.txt"), paths)
        self.assertEqual(len(rows), self.report.scanned)
        self.assertGreater(self.report.thumbs, 0)
        self.assertEqual(self.report.featured, self.report.thumbs)

    def test_every_row_has_features_and_a_thumbnail_on_disk(self):
        for row in self.conn.execute("SELECT * FROM media"):
            self.assertIsNotNone(row["features"], row["path"])
            self.assertEqual(row["feat_kind"], features.CLASSIC_KIND)
            self.assertTrue((self.library.thumb_dir / row["thumb"]).exists())

    def test_exact_duplicate_is_linked_not_deleted(self):
        flagged, kept = self.duplicate_pair()
        self.assertNotEqual(flagged["id"], kept["id"])
        self.assertEqual(flagged["sha256"], kept["sha256"])
        self.assertIsNone(kept["exact_dup_of"], "the survivor points at nobody")
        for row in (flagged, kept):
            self.assertTrue(Path(row["path"]).exists(), "ingest must never delete anything")

    def test_duplicate_survivor_does_not_depend_on_walk_order(self):
        # The pair is Camera/IMG_...jpg and Backup/copy-of-first.jpg. Whichever
        # os.walk reaches first used to win; now the rule decides, so the
        # outcome must match the rule rather than the filesystem.
        flagged, kept = self.duplicate_pair()
        self.assertLess(
            ingest.survivor_rank(kept["rel_path"]),
            ingest.survivor_rank(flagged["rel_path"]),
        )

    def test_walk_order_is_sorted(self):
        walked = [str(p.relative_to(self.tmp)) for p in ingest.walk_media(self.library)]
        by_dir = {}
        for rel in walked:
            by_dir.setdefault(str(Path(rel).parent), []).append(Path(rel).name)
        for folder, names in by_dir.items():
            self.assertEqual(names, sorted(names), folder)

    def test_burst_frames_share_a_near_duplicate_group(self):
        groups = {
            r["dup_group"]
            for r in self.conn.execute("SELECT dup_group FROM media WHERE rel_path LIKE 'Trip%'")
        }
        self.assertEqual(len(groups), 1)
        self.assertIsNotNone(groups.pop())

    def test_buckets_and_years_are_recorded(self):
        shot = self.conn.execute(
            "SELECT * FROM media WHERE filename LIKE 'Screenshot%'"
        ).fetchone()
        self.assertEqual(shot["bucket"], "Screenshots")
        dated = self.conn.execute(
            "SELECT * FROM media WHERE filename LIKE 'IMG_2019%' LIMIT 1"
        ).fetchone()
        self.assertEqual(dated["year"], "2019")

    def test_rescan_is_incremental_and_keeps_decisions(self):
        row = self.conn.execute("SELECT id FROM media LIMIT 1").fetchone()
        db.record_decision(self.conn, row["id"], "keep")
        self.conn.commit()
        second = ingest.scan(self.library)
        self.assertEqual(second.added, 0)
        self.assertGreater(second.skipped, 0)
        self.assertEqual(
            self.conn.execute(
                "SELECT action FROM verdicts WHERE media_id=?", (row["id"],)
            ).fetchone()["action"],
            "keep",
        )

    def test_deleted_file_is_marked_missing_then_restored(self):
        victim = self.conn.execute("SELECT * FROM media LIMIT 1").fetchone()
        backup = Path(victim["path"]).read_bytes()
        Path(victim["path"]).unlink()
        ingest.scan(self.library)
        self.assertEqual(
            self.conn.execute("SELECT missing FROM media WHERE id=?", (victim["id"],)).fetchone()[0], 1
        )
        Path(victim["path"]).write_bytes(backup)
        ingest.scan(self.library)
        self.assertEqual(
            self.conn.execute("SELECT missing FROM media WHERE id=?", (victim["id"],)).fetchone()[0], 0
        )


class MetadataOnlyTests(LibraryTestCase):
    """Swipes on files that could not be previewed must still teach the model."""

    def add_unpreviewable_video(self, name="beach.mp4", padding=0):
        path = make_mp4(self.tmp / "Trip" / name, padding=padding)
        ingest.scan(self.library)
        return self.conn.execute("SELECT * FROM media WHERE path=?", (str(path),)).fetchone()

    def test_swipes_on_unpreviewable_files_still_train(self):
        video = self.add_unpreviewable_video()
        self.assertIsNone(video["features"], "nothing could extract a frame")
        self.swipe_all()
        db.record_decision(self.conn, video["id"], "love")

        self.assertIn(video["id"], {r["id"] for r in db.labelled_rows(self.conn)})
        report = model.train(self.conn)
        self.assertEqual(report.n_without_image, 1)
        model.score_all(self.conn)
        score = self.conn.execute(
            "SELECT score FROM media WHERE id=?", (video["id"],)
        ).fetchone()["score"]
        self.assertIsNotNone(score, "it gets a score like everything else")
        self.assertTrue(0.0 <= score <= 1.0)

    def test_a_library_with_no_previews_at_all_still_learns(self):
        tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        # Big clips are keepers, tiny ones are accidental pocket recordings.
        for i in range(16):
            make_mp4(tmp / f"keep_{i:02d}.mp4", padding=400_000 + i * 20_000)
            make_mp4(tmp / f"bin_{i:02d}.mp4", padding=2_000 + i * 100)
        library = Library.resolve(tmp)
        ingest.scan(library)
        conn = db.connect(library.db_path)
        self.addCleanup(conn.close)
        for row in conn.execute("SELECT id, filename FROM media").fetchall():
            db.record_decision(conn, row["id"], "keep" if row["filename"].startswith("keep") else "drop")

        report = model.train(conn)
        self.assertIsNotNone(report)
        self.assertEqual(report.feat_kind, "metadata-only")
        self.assertEqual(report.n_without_image, 32)
        model.score_all(conn)
        scores = {r["filename"]: r["score"] for r in conn.execute("SELECT filename, score FROM media")}
        keep_mean = np.mean([s for f, s in scores.items() if f.startswith("keep")])
        bin_mean = np.mean([s for f, s in scores.items() if f.startswith("bin")])
        self.assertGreater(keep_mean, bin_mean + 0.3, "file size alone separates them")

    def test_gaps_explain_unpreviewable_videos(self):
        self.add_unpreviewable_video()
        gaps = ingest.decode_gaps(self.conn)
        self.assertEqual([(g.kind, g.count) for g in gaps], [("video", 1)])
        self.assertIn("ffmpeg", gaps[0].fix)

    def test_gap_fix_names_ffmpeg_install_only_when_it_is_missing(self):
        self.add_unpreviewable_video()
        with mock.patch("swipesort.ingest.ffmpeg_path", return_value=None):
            self.assertIn("winget install", ingest.decode_gaps(self.conn)[0].fix)
        with mock.patch("swipesort.ingest.ffmpeg_path", return_value="/usr/bin/ffmpeg"):
            self.assertNotIn("winget", ingest.decode_gaps(self.conn)[0].fix)

    def test_gap_fix_for_raw_names_rawpy_only_when_it_is_missing(self):
        (self.tmp / "DSCF9999.RAF").write_bytes(b"FUJIFILMCCD-RAW " + bytes(512))
        ingest.scan(self.library)
        with mock.patch.object(features, "rawpy_available", return_value=False):
            raw = [g for g in ingest.decode_gaps(self.conn) if g.kind == "raw"]
            self.assertIn("pip install rawpy", raw[0].fix)
        with mock.patch.object(features, "rawpy_available", return_value=True):
            raw = [g for g in ingest.decode_gaps(self.conn) if g.kind == "raw"]
            self.assertNotIn("pip install", raw[0].fix)

    def test_vectors_from_two_backends_do_not_stop_training(self):
        # A partial re-ingest with another backend leaves two vector widths.
        # Training must carry on, treating the minority as metadata-only.
        self.swipe_all()
        odd = self.conn.execute(
            "SELECT id FROM media WHERE features IS NOT NULL LIMIT 3"
        ).fetchall()
        for row in odd:
            self.conn.execute(
                "UPDATE media SET features=?, feat_kind='other' WHERE id=?",
                (features.pack(np.ones(7, dtype=np.float32)), row["id"]),
            )
        self.conn.commit()
        report = model.train(self.conn)
        self.assertIsNotNone(report)
        self.assertEqual(model.score_all(self.conn),
                         self.conn.execute("SELECT COUNT(*) FROM media WHERE missing=0").fetchone()[0])

        width = model.image_width(self.conn)
        vector = model.row_vector(
            self.conn.execute("SELECT * FROM media WHERE id=?", (odd[0]["id"],)).fetchone(),
            image_dim=width,
        )
        self.assertTrue(np.isnan(vector[:width]).all(), "its foreign vector is not used")
        self.assertEqual(vector[-1], 0.0, "and has_image says so")

    def test_a_decode_that_starts_failing_clears_stale_pixels(self):
        row = self.conn.execute("SELECT * FROM media WHERE kind='photo' LIMIT 1").fetchone()
        self.assertIsNotNone(row["features"])
        Path(row["path"]).write_bytes(b"truncated")
        ingest.scan(self.library)
        after = self.conn.execute("SELECT * FROM media WHERE id=?", (row["id"],)).fetchone()
        self.assertIsNone(after["features"])
        self.assertIsNone(after["thumb"])


class ModelOnLibraryTests(LibraryTestCase):
    def test_model_learns_the_fixture_preference(self):
        self.swipe_all()
        report = model.train(self.conn)
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.n_labels, self.keepers + self.junk - 1)
        self.assertIsNotNone(report.holdout_accuracy)
        self.assertGreater(report.holdout_accuracy, 0.8)
        self.assertGreaterEqual(report.holdout_accuracy, report.baseline_accuracy)

        scored = model.score_all(self.conn)
        self.assertGreater(scored, 0)
        scores = {
            r["path"]: r["score"]
            for r in self.conn.execute("SELECT path, score FROM media WHERE score IS NOT NULL")
        }
        keep_mean = np.mean([scores[p] for p in self.manifest["keep"] if p in scores])
        drop_mean = np.mean([scores[p] for p in self.manifest["drop"] if p in scores])
        self.assertGreater(keep_mean, drop_mean + 0.3)

    def test_training_refuses_until_there_are_enough_labels(self):
        row = self.conn.execute("SELECT id FROM media LIMIT 1").fetchone()
        db.record_decision(self.conn, row["id"], "keep")
        self.assertIsNone(model.train(self.conn))

    def test_training_refuses_when_only_one_class_is_present(self):
        for row in self.conn.execute("SELECT id FROM media").fetchall():
            db.record_decision(self.conn, row["id"], "keep")
        self.assertIsNone(model.train(self.conn))


class QueueTests(LibraryTestCase):
    def test_cold_start_returns_items_in_every_mode(self):
        for mode in queue_mod.MODES:
            items = queue_mod.build(self.conn, mode=mode, limit=10)
            self.assertTrue(items, mode)
            self.assertNotIn(None, [i["id"] for i in items])

    def test_decided_items_leave_the_queue(self):
        first = queue_mod.build(self.conn, mode="backlog", limit=1)[0]
        db.record_decision(self.conn, first["id"], "drop")
        again = queue_mod.build(self.conn, mode="backlog", limit=10)
        self.assertNotIn(first["id"], [i["id"] for i in again])

    def test_later_is_hidden_unless_asked_for(self):
        first = queue_mod.build(self.conn, mode="backlog", limit=1)[0]
        db.record_decision(self.conn, first["id"], "later")
        self.assertNotIn(first["id"], [i["id"] for i in queue_mod.build(self.conn, mode="backlog", limit=99)])
        self.assertIn(
            first["id"],
            [i["id"] for i in queue_mod.build(self.conn, mode="backlog", limit=99, include_later=True)],
        )

    def test_exact_duplicates_never_reach_the_queue(self):
        flagged, kept = self.duplicate_pair()
        ids = [i["id"] for i in queue_mod.build(self.conn, mode="backlog", limit=999)]
        self.assertNotIn(flagged["id"], ids, "the duplicate is not worth a swipe")
        self.assertIn(kept["id"], ids, "but the copy that survives still is")

    def test_near_duplicates_arrive_together(self):
        items = queue_mod.build(self.conn, mode="learn", limit=999)
        positions = [i for i, item in enumerate(items) if item["dup_group"] is not None
                     and item["rel_path"].startswith("Trip")]
        self.assertGreater(len(positions), 1)
        self.assertEqual(positions, list(range(positions[0], positions[0] + len(positions))))

    def test_clean_mode_puts_likely_rubbish_first(self):
        self.swipe_all()
        model.train(self.conn)
        model.score_all(self.conn)
        # Undo every verdict so the whole library is queueable again.
        self.conn.execute("DELETE FROM decisions")
        self.conn.commit()
        items = queue_mod.build(self.conn, mode="clean", limit=10)
        keepers = queue_mod.build(self.conn, mode="keepers", limit=10)
        self.assertLess(items[0]["score"], 0.5)
        self.assertLess(np.mean([i["score"] for i in items]),
                        np.mean([i["score"] for i in keepers]))
        self.assertGreater(np.mean([i["score"] for i in keepers]), 0.5)

    def test_clean_mode_orders_likely_rubbish_by_size(self):
        self.swipe_all()
        model.train(self.conn)
        model.score_all(self.conn)
        self.conn.execute("DELETE FROM decisions")
        self.conn.commit()
        items = queue_mod.build(self.conn, mode="clean", limit=999)
        rubbish = [i for i in items if i["score"] < 0.5]
        liked = [i for i in items if i["score"] >= 0.5]
        self.assertTrue(rubbish and liked)
        # Every likely-rubbish item comes before every liked one, except where a
        # near-duplicate group deliberately keeps its members together.
        loose = [i for i in items if i["dup_group"] is None]
        last_rubbish = max(i for i, item in enumerate(loose) if item["score"] < 0.5)
        first_liked = min(i for i, item in enumerate(loose) if item["score"] >= 0.5)
        self.assertLess(last_rubbish, first_liked)

    def test_learn_mode_prefers_uncertainty(self):
        self.swipe_all()
        model.train(self.conn)
        model.score_all(self.conn)
        self.conn.execute("DELETE FROM decisions")
        self.conn.commit()
        learn = queue_mod.build(self.conn, mode="learn", limit=8)
        confident = queue_mod.build(self.conn, mode="keepers", limit=8)
        def spread(items):
            return np.mean([abs(i["score"] - 0.5) for i in items])

        self.assertLess(spread(learn), spread(confident))

    def test_filters(self):
        items = queue_mod.build(self.conn, mode="backlog", limit=999, year="2019")
        self.assertTrue(items)
        self.assertTrue(all(i["year"] == "2019" for i in items))
        shots = queue_mod.build(self.conn, mode="backlog", limit=999, bucket="Screenshots")
        self.assertTrue(all(i["bucket"] == "Screenshots" for i in shots))


class ApplyTests(LibraryTestCase):
    def test_dry_run_moves_nothing(self):
        self.swipe_all()
        before = sorted(p.name for p in self.tmp.rglob("*.jpg"))
        report = apply_mod.apply_decisions(self.library, dry_run=True)
        self.assertTrue(report.dry_run)
        self.assertGreater(report.quarantined, 0)
        self.assertEqual(sorted(p.name for p in self.tmp.rglob("*.jpg")), before)

    def test_apply_quarantines_drops_and_files_keepers(self):
        self.swipe_all()
        report = apply_mod.apply_decisions(self.library)
        self.assertFalse(report.dry_run)
        self.assertGreater(report.quarantined, 0)
        self.assertGreater(report.sorted_, 0)

        self.assertTrue((self.library.quarantine_dir / report.batch).exists())
        self.assertTrue((self.library.quarantine_dir / report.batch / "manifest.json").exists())
        self.assertTrue((self.tmp / "2019 Photos").is_dir())

        # Nothing was deleted: every dropped file is still on disk, in quarantine.
        quarantined = list((self.library.quarantine_dir / report.batch).rglob("*.jpg"))
        self.assertEqual(len(quarantined), report.quarantined)

    def test_undo_restores_every_file(self):
        self.swipe_all()
        before = {str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*") if p.is_file()
                  and not self.library.is_internal(p)}
        report = apply_mod.apply_decisions(self.library)
        restored, errors = apply_mod.undo_batch(self.library, report.batch)
        self.assertEqual(errors, [])
        self.assertGreater(restored, 0)
        after = {str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*") if p.is_file()
                 and not self.library.is_internal(p)}
        self.assertEqual(before, after)

    def test_same_name_different_content_keeps_both(self):
        target = self.tmp / "2019 Photos"
        target.mkdir(parents=True, exist_ok=True)
        row = self.a_plain_keeper()
        clash = target / row["filename"]
        clash.write_bytes(b"a different file entirely")
        db.record_decision(self.conn, row["id"], "keep")
        self.conn.commit()

        apply_mod.apply_decisions(self.library, quarantine_drops=False, include_exact_duplicates=False)
        self.assertTrue(clash.exists(), "the pre-existing file must survive")
        self.assertEqual(clash.read_bytes(), b"a different file entirely")
        survivors = list(target.glob(f"{Path(row['filename']).stem}*"))
        self.assertEqual(len(survivors), 2)

    def test_identical_name_and_content_collapses(self):
        target = self.tmp / "2019 Photos"
        target.mkdir(parents=True, exist_ok=True)
        row = self.a_plain_keeper()
        shutil.copy2(row["path"], target / row["filename"])
        db.record_decision(self.conn, row["id"], "keep")
        self.conn.commit()

        report = apply_mod.apply_decisions(
            self.library, quarantine_drops=False, include_exact_duplicates=False
        )
        self.assertEqual(report.collapsed, 1)
        self.assertEqual(len(list(target.glob(f"{Path(row['filename']).stem}*"))), 1)

    def test_exact_duplicates_are_quarantined_and_one_copy_stays(self):
        flagged, kept = self.duplicate_pair()
        apply_mod.apply_decisions(self.library, sort_keeps=False, quarantine_drops=False)
        self.assertFalse(Path(flagged["path"]).exists(), "the duplicate left its place")
        self.assertTrue(Path(kept["path"]).exists(), "exactly one copy must survive")
        quarantined = list(self.library.quarantine_dir.rglob(Path(flagged["path"]).name))
        self.assertEqual(len(quarantined), 1, "and it is in quarantine, not deleted")

    def test_prune_empty_clears_emptied_folders_but_not_the_root(self):
        emptied = self.conn.execute(
            "SELECT id FROM media WHERE rel_path LIKE 'Trip%' AND missing=0"
        ).fetchall()
        self.assertTrue(emptied)
        for row in emptied:
            db.record_decision(self.conn, row["id"], "drop")
        self.conn.commit()

        apply_mod.apply_decisions(self.library, sort_keeps=False, prune_empty=True)
        self.assertFalse((self.tmp / "Trip").exists(), "a folder emptied by the move goes")
        self.assertTrue(self.tmp.is_dir(), "the library root must survive")

    def test_prune_empty_leaves_folders_that_still_hold_files(self):
        self.swipe_all()
        (self.tmp / "Camera" / "keepme.txt").write_text("not media", encoding="utf-8")
        apply_mod.apply_decisions(self.library, prune_empty=True)
        self.assertTrue((self.tmp / "Camera" / "keepme.txt").exists())

    def test_empty_quarantine_is_the_only_destructive_step(self):
        self.swipe_all()
        report = apply_mod.apply_decisions(self.library)
        count, freed = apply_mod.empty_quarantine(self.library, report.batch)
        self.assertGreater(count, 0)
        self.assertGreater(freed, 0)
        self.assertFalse((self.library.quarantine_dir / report.batch).exists())


class ApiTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient

        from swipesort.server import create_app

        self.client = TestClient(create_app(self.library))

    def test_state_and_queue(self):
        state = self.client.get("/api/state").json()
        self.assertEqual(state["library"], str(self.tmp))
        self.assertGreater(state["counts"]["total"], 0)

        queue = self.client.get("/api/queue", params={"mode": "learn", "limit": 5}).json()
        self.assertEqual(len(queue["items"]), 5)
        self.assertIn("thumb_url", queue["items"][0])

    def test_queue_items_say_whether_there_is_a_preview(self):
        item = self.client.get("/api/queue", params={"limit": 1}).json()["items"][0]
        self.assertTrue(item["has_thumb"])
        self.assertTrue(item["has_features"])

    def test_unknown_mode_is_rejected(self):
        self.assertEqual(self.client.get("/api/queue", params={"mode": "nope"}).status_code, 400)

    def test_decide_then_undo(self):
        item = self.client.get("/api/queue", params={"limit": 1}).json()["items"][0]
        decided = self.client.post("/api/decide", json={"id": item["id"], "action": "drop"})
        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json()["counts"]["drop"], 1)

        undone = self.client.post("/api/undo").json()
        self.assertTrue(undone["ok"])
        self.assertEqual(undone["undone"]["media_id"], item["id"])
        self.assertEqual(undone["counts"]["drop"], 0)

    def test_bad_action_is_rejected(self):
        item = self.client.get("/api/queue", params={"limit": 1}).json()["items"][0]
        response = self.client.post("/api/decide", json={"id": item["id"], "action": "burn"})
        self.assertEqual(response.status_code, 422)

    def test_drop_rest_of_group(self):
        items = self.client.get("/api/queue", params={"mode": "backlog", "limit": 999}).json()["items"]
        burst = next(i for i in items if i["dup_group_size"] > 1)
        result = self.client.post(
            "/api/decide",
            json={"id": burst["id"], "action": "love", "drop_rest_of_group": True},
        ).json()
        self.assertEqual(result["also_dropped"], burst["dup_group_size"] - 1)

    def test_thumbnails_and_media_are_served(self):
        item = self.client.get("/api/queue", params={"limit": 1}).json()["items"][0]
        thumb = self.client.get(item["thumb_url"])
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.headers["content-type"], "image/jpeg")
        self.assertEqual(self.client.get(item["media_url"]).status_code, 200)
        self.assertEqual(self.client.get("/media/999999").status_code, 404)

    def test_apply_defaults_to_a_dry_run(self):
        item = self.client.get("/api/queue", params={"limit": 1}).json()["items"][0]
        self.client.post("/api/decide", json={"id": item["id"], "action": "drop"})
        result = self.client.post("/api/apply", json={}).json()
        self.assertTrue(result["dry_run"])
        self.assertTrue(Path(item["rel_path"]).name)
        self.assertTrue((self.tmp / item["rel_path"]).exists(), "dry run must not move anything")

    def test_train_endpoint_reports_when_undertrained(self):
        result = self.client.post("/api/train", json={}).json()
        self.assertFalse(result["ok"])
        self.assertIn("more swipes", result["reason"])

    def test_index_page_is_served(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("swipesort", page.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
