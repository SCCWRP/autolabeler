import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

import autolabeler as app


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.refs = self.root / "references"
        self.refs.mkdir()
        (self.refs / "senorita.md").write_text("Guide: senorita", encoding="utf-8")
        Image.new("RGB", (40, 20), "yellow").save(self.refs / "positive.png")

    def add_group(self, name, split, colors):
        source = self.root / name
        source.mkdir()
        for i, color in enumerate(colors):
            Image.new("RGB", (100, 50), color).save(source / f"{i}.png")
        args = SimpleNamespace(source=str(source), group=name, split=split,
                               every=5, limit=20, work=self.work)
        with contextlib.redirect_stdout(io.StringIO()):
            app.prepare(args)

    def record(self, path, group, status, boxes):
        app.save_json(self.work / "annotations" / group["group"] / (path.stem + ".json"),
                      {"image_sha256": app.digest(path), "signature": "same-experiment",
                       "annotation": {"status": status, "reason": "<test>", "boxes": boxes}})

    def test_reject_invalid_boxes_and_statuses(self):
        valid = dict(xmin=.1, ymin=.2, xmax=.8, ymax=.9)
        for change in ({"xmin": -.1}, {"xmax": .1}, {"ymax": 2}, {"xmin": float("nan")}):
            with self.assertRaises(ValidationError):
                app.Box(**(valid | change))
        for value in (
            {"status": "negative", "reason": "", "boxes": [valid]},
            {"status": "labeled", "reason": "", "boxes": []},
            {"status": "banana", "reason": "", "boxes": []},
        ):
            with self.assertRaises(ValidationError):
                app.Annotation.model_validate(value)

    def test_yolo_coordinates(self):
        box = app.Box(xmin=.1, ymin=.2, xmax=.5, ymax=.8)
        self.assertEqual(app.yolo_line(box), "0 0.300000 0.500000 0.400000 0.600000")

    def test_export_negatives_uncertainty_and_test_exclusion(self):
        self.add_group("dive_a", "train", ["red", "green", "blue"])
        self.add_group("dive_b", "val", ["purple"])
        self.add_group("dive_c", "test", ["white"])
        for i, (path, group) in enumerate(app.inventory(self.work)):
            status = ["labeled", "negative", "uncertain", "labeled", "labeled"][i]
            boxes = [] if status == "negative" else [dict(xmin=.1, ymin=.2, xmax=.5, ymax=.8)]
            self.record(path, group, status, boxes)
        output = self.root / "yolo"
        app.export(SimpleNamespace(work=self.work, output=output))
        self.assertEqual(len(list((output / "images/train").glob("*"))), 2)
        self.assertEqual((output / "labels/train/dive_a--000002.txt").read_text(), "")
        self.assertFalse((output / "images/test").exists())
        data = json.loads((output / "data.yaml").read_text())
        self.assertEqual(data["names"], {"0": "senorita"})
        app.preview(SimpleNamespace(work=self.work))
        page = (self.work / "preview/index.html").read_text()
        self.assertIn("&lt;test&gt;", page)
        self.assertEqual(len(list((self.work / "preview").glob("*.jpg"))), 5)

    def test_duplicate_images_cannot_cross_splits(self):
        self.add_group("a", "train", ["red"])
        self.add_group("b", "val", ["red"])
        for path, group in app.inventory(self.work):
            self.record(path, group, "labeled", [dict(xmin=.1, ymin=.1, xmax=.9, ymax=.9)])
        with self.assertRaisesRegex(ValueError, "leakage"):
            app.export(SimpleNamespace(work=self.work, output=self.root / "yolo"))

    def test_changed_image_rejected(self):
        self.add_group("a", "train", ["red"])
        path, group = app.inventory(self.work)[0]
        self.record(path, group, "negative", [])
        Image.new("RGB", (100, 50), "black").save(path)
        with self.assertRaisesRegex(ValueError, "changed"):
            list(app.annotated_entries(self.work))

    def test_label_requests_resume_and_config_change(self):
        self.add_group("a", "train", ["red", "green"])
        annotation = app.Annotation(status="negative", reason="No target", boxes=[])
        response = SimpleNamespace(
            status="completed", output_parsed=annotation, id="fake-response", usage=None,
            model_dump=lambda **_: {"id": "fake-response", "status": "completed"})
        args = SimpleNamespace(model="test-model", references=str(self.refs),
                               work=self.work, limit=1, dry_run=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-only-for-test"}), \
                patch("openai.OpenAI") as client:
            client.return_value.responses.parse.return_value = response
            app.label(args)
            self.assertEqual(client.return_value.responses.parse.call_count, 1)
            sent = client.return_value.responses.parse.call_args.kwargs
            self.assertFalse(sent["store"])
            content = sent["input"][1]["content"]
            self.assertEqual(sum(c["type"] == "input_image" for c in content), 2)
            self.assertEqual(content[-2]["text"], "TARGET IMAGE: annotate only this image")
            app.label(args)
            self.assertEqual(client.return_value.responses.parse.call_count, 2)
            app.label(args)
            self.assertEqual(client.return_value.responses.parse.call_count, 2)
            (self.refs / "senorita.md").write_text("Changed guide")
            with self.assertRaisesRegex(ValueError, "Stale"):
                app.label(args)

    def test_refusal_does_not_become_negative_label(self):
        self.add_group("a", "train", ["red"])
        response = SimpleNamespace(
            status="completed", output_parsed=None,
            model_dump=lambda **_: {"status": "completed", "output": [{"type": "refusal"}]})
        args = SimpleNamespace(model="test-model", references=str(self.refs),
                               work=self.work, limit=1, dry_run=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-only-for-test"}), \
                patch("openai.OpenAI") as client:
            client.return_value.responses.parse.return_value = response
            with self.assertRaisesRegex(ValueError, "failed"):
                app.label(args)
        self.assertEqual(len(list((self.work / "responses").rglob("*.json"))), 1)
        self.assertEqual(list((self.work / "annotations").rglob("*.json")), [])

    def test_dry_run_does_not_construct_client(self):
        self.add_group("a", "train", ["red"])
        args = SimpleNamespace(model="test-model", references=str(self.refs),
                               work=self.work, limit=1, dry_run=True)
        with patch("openai.OpenAI") as client:
            app.label(args)
            client.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_video_extraction(self):
        video = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
                        "-i", "color=c=blue:s=64x48:r=10:d=2",
                        "-c:v", "libx264", str(video)], check=True)
        app.prepare(SimpleNamespace(source=str(video), group="video", split="train",
                                    every=1, limit=10, work=self.work))
        entries = app.inventory(self.work)
        self.assertEqual(len(entries), 2)
        with Image.open(entries[0][0]) as im:
            self.assertEqual(im.size, (64, 48))

    def test_prepare_does_not_overwrite_group(self):
        self.add_group("a", "train", ["red"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            app.prepare(SimpleNamespace(source=str(self.root / "a"), group="a", split="val",
                                        every=5, limit=20, work=self.work))


if __name__ == "__main__":
    unittest.main()
