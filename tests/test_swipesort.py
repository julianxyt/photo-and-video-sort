"""End-to-end and unit tests. Run with: python -m unittest discover -s tests"""
from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from swipesort import apply as apply_mod, capture_date, classify, db, features, ingest, model, phash
from swipesort import queue as queue_mod
from swipesort.config import Library

from fixtures import build_library


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
        created = int((datetime(2019, 6, 1) - datetime(1904, 1, 1)).total_seconds())
        mvhd = b"mvhd" + bytes([0, 0, 0, 0]) + struct.pack(">I", created) + b"\x00" * 80
        mvhd = struct.pack(">I", len(mvhd) + 4) + mvhd
        moov = struct.pack(">I", len(mvhd) + 8) + b"moov" + mvhd
        ftyp = struct.pack(">I", 16) + b"ftyp" + b"isom" + b"\x00" * 4

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(ftyp + moov)
            found, source = capture_date.capture_datetime(path)
        self.assertIsNotNone(found)
        self.assertEqual(found.year, 2019)
        self.assertEqual(source, "quicktime")

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
