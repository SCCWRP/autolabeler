"""Optional integration smoke: synthetic rectangles, no API calls or real fish.
Run: uv run --extra train python tests/smoke_yolo.py
This is a plumbing test, not an accuracy benchmark.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import autolabeler as app
from types import SimpleNamespace


def run(arguments, cwd):
    result = subprocess.run([sys.executable, str(ROOT / "autolabeler.py"), *arguments],
                            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:] + "\n" + result.stderr[-6000:])


def main():
    with tempfile.TemporaryDirectory(prefix="autolabeler-smoke-") as tmp:
        root = Path(tmp)
        work = root / "work"
        for split in ("train", "val"):
            source = root / split
            source.mkdir()
            for i in range(4):
                im = Image.new("RGB", (64, 64), (i * 25, 20 if split == "train" else 60, 80))
                ImageDraw.Draw(im).rectangle((16, 16, 48, 48), fill="yellow")
                im.save(source / f"{i}.png")
            app.prepare(SimpleNamespace(source=str(source), group=split, split=split,
                                        every=5, limit=4, work=work))
        for path, group in app.inventory(work):
            app.save_json(work / "annotations" / group["group"] / (path.stem + ".json"), {
                "signature": "synthetic-only", "image_sha256": app.digest(path),
                "annotation": {"status": "labeled", "reason": "Synthetic test rectangle",
                               "boxes": [{"xmin": .25, "ymin": .25, "xmax": .75, "ymax": .75}]}})
        dataset = root / "yolo"
        app.export(SimpleNamespace(work=work, output=dataset))
        # YAML initializes an untrained architecture, so no pretrained weight download.
        run(["train", "--weights", "yolo26n.yaml", "--data", str(dataset / "data.yaml"),
             "--epochs", "1", "--imgsz", "64", "--device", "cpu"], root)
        weights = root / "runs/senorita/weights/best.pt"
        assert weights.is_file(), "Training did not produce best.pt"
        run(["predict", str(dataset / "images/val"), "--weights", str(weights)], root)
        assert list((root / "runs/predictions").glob("*.jpg")), "No prediction previews"
        run(["evaluate", "--data", str(dataset / "data.yaml"),
             "--weights", str(weights), "--split", "val"], root)
        print("PASS: actual YOLO train, predict, and evaluate commands on synthetic data.")
        print("No fish-model accuracy is established by this test.")


if __name__ == "__main__":
    main()
