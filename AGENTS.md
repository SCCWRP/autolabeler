# Autolabeler: handoff for Codex / other coding agents

## Objective and context
Robert and his coworker were asked to build a small demo showing a frontier
multimodal model creating labels to train YOLO to identify a fish species.
Start with one class: senorita (señorita wrasse / Oxyjulis californica).
The model gets a field guide and labeled reference images in its context.
The trained YOLO model must predict independently, without the frontier model.

The user explicitly wants to test automated labeling. Do not impose human
review on every annotation. A small independently labeled test set measures
whether the approach works. Species identification and box localization are
separate capabilities; don't assert one proves the other.

This is a NEW project at Projects/autolabeler. Do not modify the sibling
AutoTrimMarineEnvironment project. The old conversation involved fish counting
and above/below-water video trimming; those are out of scope here.

## Working style
KISS matters to Robert. Keep one straightforward CLI, small functions, and plain
files. No provider framework, queues, web service, database, or plugin architecture
unless an actual need appears. Brief updates and results; no long speculative
diagnoses or repeated full-video benchmarks.
Inspect changes before editing. Never delete a file to bypass an edit-conflict
warning. Do not rewrite this document as a diary of unverified claims.

## Current implementation
autolabeler.py has prepare, label, preview, export, train, predict, evaluate.
OpenAI Responses API plus Pydantic structured output. OPENAI_MODEL is explicit
in .env, not a claimed latest/best default. Ultralytics is an optional dependency.
references/senorita.md and user-supplied positive images are the initial context.

Class ID 0 = senorita. Boxes use normalized xyxy in saved JSON and normalized
center/width/height in YOLO text. Prepared images normalize EXIF orientation;
API image resizing preserves aspect ratio.
Uncertain images are excluded whole, never converted into negative training
images. Refusals/errors are not labels. Successful labels cache by image and
configuration hashes. Full API responses/token usage persist locally.
Source recording groups stay in one split. Export checks source reuse and exact
duplicates, but cannot detect related/re-encoded footage. Test groups are excluded.
Pseudo-label validation metrics are teacher agreement, not independent accuracy.

## Commands
uv sync
uv run python -m unittest discover -s tests -v
uv run autolabeler.py --help
uv sync --extra train
See README.md for the end-to-end example.

.env is ignored. Never print or copy API keys from sibling repositories.
Network labeling is billable; do not launch bulk runs merely to test a refactor.
Default label limit is 10; --dry-run makes no requests.
Tests are offline and mock the API. Unit-test success is not proof of labeling
quality, API account access or a successful trained model.
Optional training may download weights; no model weights are checked in.

## Next experiment
1. Configure the chosen OpenAI model/key and prepare distinct video groups.
2. Label a small representative set and inspect annotated previews for missed
   fish, wrong species and box placement.
3. Build train/val data, train YOLO, predict on held-out video.
4. Independently label a small held-out test set for evaluation.
Record costs and error examples before scaling. Do not silently train on the
reference images or claim success based only on synthetic tests.

## Sources
README.md links the official image input, structured output and YOLO docs.
Reference screenshots were supplied by the user; keep them as context only and
don't publish them without checking rights.

## Verification at initial handoff (2026-09-09)
- 10 offline unittest cases passed, including real ffmpeg extraction.
- tests/smoke_yolo.py passed: actual YOLO training (1 CPU epoch), prediction,
  and validation on synthetic rectangles via the real CLI.
- OpenAI calls are mocked in tests. No paid labeling calls have been made,
  no real fish dataset has been labeled, and no fish accuracy is established.
- Base and optional training dependencies are installed; uv.lock is included.
- .env exists with blank OPENAI_API_KEY / OPENAI_MODEL / FFMPEG_DIR.
- ffmpeg is available on this machine's current PATH.
- Git is initialized locally on main; no commits or remote were created.
- First experiment needs source footage and the chosen API key/model.
- YOLO project output is an absolute path to cwd/runs, avoiding dependence
  on a user's global Ultralytics runs_dir setting.

