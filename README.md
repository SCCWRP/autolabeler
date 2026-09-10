# Autolabeler

A small experiment: give a multimodal model a señorita field guide and reference
images, ask it to label new images, then train a one-class YOLO detector on those
annotations. The trained YOLO model runs on its own.

This is separate from AutoTrimMarineEnvironment. There is no fish counting or
above/below-water logic here.

## Setup

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). From this folder:

```powershell
uv sync
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`. Use an image-capable OpenAI
model that supports Responses structured outputs and is available to your
account. The model is deliberately configurable; no paid requests run during
setup or tests. API usage is billed separately from a chat subscription.

For video extraction, install ffmpeg and put it on PATH, or set `FFMPEG_DIR`
in `.env` to its bin directory. Image-folder import does not require ffmpeg.

## 1. Prepare images

Assign each recording/deployment to exactly one split. Use different videos
for training, validation and testing. A group is a recording, not a species.
Put representative negatives (other fish, kelp, empty water) in the source
material too.

```powershell
uv run autolabeler.py prepare "C:/footage/dive-a.mp4" --group dive-a --split train --every 5 --limit 200
uv run autolabeler.py prepare "C:/footage/dive-b.mp4" --group dive-b --split val --every 5 --limit 50
uv run autolabeler.py prepare "C:/footage/dive-c.mp4" --group dive-c --split test --every 5 --limit 50
```

Or pass a folder of JPG/PNG/WebP images instead of a video. Import is top-level
only, normalizes orientation and saves JPEG copies without changing the source.
Video samples start at the beginning; `--every` is seconds per sample.
Names such as 000001.jpg are sample IDs, **not original frame numbers**.
A failed extraction has no group.json and is not eligible for labeling; remove
that incomplete group yourself or choose a new name.

## 2. Generate labels

`references/senorita.md` and the reference images are sent with **every** target
image. These images come from the original conversation and are context only.
Add better views/lookalikes to the guide as the experiment develops. Check
source-image rights before distributing the references or datasets.

Start small:

```powershell
uv run autolabeler.py label --dry-run --limit 10
uv run autolabeler.py label --limit 10
uv run autolabeler.py preview
```

Open `data/preview/index.html` to see the boxes. Preview generation is optional;
it neither calls the model nor blocks automatic export.

To label more images, repeat with a higher limit:
```powershell
uv run autolabeler.py label --limit 300
```

One request per image, processed sequentially. `--limit` bounds new target images,
not tokens or dollars; SDK retries may issue extra requests. The reference guide
adds input tokens to each request. Original-sized prepared images remain local;
API copies preserve aspect ratio and are capped at 1600 pixels on the long edge.

Successful annotations are saved immediately and skipped on restart. Full API
responses and token usage are retained locally. No keys are logged. Errors and
refusals do not create negative labels. Changing the model, guide, references,
prompt or image invalidates the cache; use a fresh `--work data/experiment-2`
folder and prepare its groups again. This prevents silently mixing experiments.

The model reports:
- **labeled:** target fish with normalized bounding boxes.
- **negative:** confidently no señorita; exports an empty label file.
- **uncertain:** questionable identity, boxes or completeness; entire image excluded.

These statuses are model judgments, not calibrated accuracy guarantees. A
confidently missed fish can still yield a bad training label.

## 3. Export and train

```powershell
uv run autolabeler.py export --output data/yolo
uv sync --extra train
uv run --extra train autolabeler.py train --data data/yolo/data.yaml --epochs 30 --device cpu
```

For a configured CUDA environment use `--device 0`. Default weights are
`yolo26n.pt`; Ultralytics downloads pretrained weights when needed. Training is
optional and installs PyTorch through Ultralytics. CPU training may be slow.
The current Ultralytics distribution has its own licensing terms; review those
before deployment/distribution.

Export requires usable train and val groups and at least one positive training
box. It refuses existing output folders, mixed annotation configurations,
reused sources across splits, and byte-identical images across splits.
It cannot detect re-encoded near-duplicates or different files from the same dive:
correct grouping is still your responsibility. Unlabeled and uncertain images
are omitted. Test groups are **never** exported into training data.

Labels: `0 x_center y_center width height`, normalized to 0..1.
`data.yaml` is written as JSON, which is also valid YAML.
`provenance.json` identifies these as model-generated labels.

## 4. Show YOLO working without the frontier model

```powershell
uv run --extra train autolabeler.py predict "C:/footage/unseen.mp4" --weights runs/senorita/weights/best.pt
```

Outputs go under `runs/`. Repeated training/prediction runs may get numbered
directories; use the actual weights path printed by Ultralytics.

For a meaningful evaluation, prepare a **separate, independently annotated**
YOLO dataset from held-out videos. Then:

```powershell
uv run --extra train autolabeler.py evaluate --weights runs/senorita/weights/best.pt --data gold/data.yaml --split test
```

Validation against automatically generated labels measures agreement with the
labeler, not biological accuracy. The goal is to show whether YOLO learns useful
detections from those labels. Check missed fish, incorrect species, and box
placement separately. Precise localization and dense schools are open questions,
even when species recognition works.

## Files and development

- `autolabeler.py`: commands and small annotation schemas.
- `references/`: field guide and positive examples.
- `tests/`: offline tests, including mocked API calls; no credentials required.
- `AGENTS.md`: context and instructions for Codex in VS Code.
- `data/images/<group>/`: prepared images and source/split metadata.
- `data/annotations/`, `data/responses/`: resumable annotations and provenance.
- `data/preview/`: local contact sheet.
- `data/yolo/`: exported dataset.

```powershell
uv run python -m unittest discover -s tests -v
```

Optional end-to-end plumbing test (one CPU epoch on synthetic rectangles, no API):
```powershell
uv run --extra train python tests/smoke_yolo.py
```
This tests training/prediction integration, not fish recognition accuracy.

Open **this folder** in VS Code so Codex finds AGENTS.md. Start with:
"Read AGENTS.md and continue the señorita auto-labeling experiment."

Implementation references:
- [OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Ultralytics detection](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect/)

