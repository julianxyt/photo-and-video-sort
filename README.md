# photo-and-video-sort

[![CI](https://github.com/julianxyt/photo-and-video-sort/actions/workflows/ci.yml/badge.svg)](https://github.com/julianxyt/photo-and-video-sort/actions/workflows/ci.yml)

Two generations of the same job: getting a messy photo and video library into
`<year> <type>` folders without losing anything.

* **`Split_To_Years_Greedy_EXIF.ps1`** and **`Remove-DuplicateFiles.txt`** -
  the original PowerShell scripts. Still here, still work, still Windows-only.
* **`swipesort/`** - the new one. Ingest the library once, triage it by swiping
  on your phone, and let a local model learn which photos you actually keep so
  it can rank the rest of the backlog for you.

---

## swipesort

The problem with a 40,000-photo travel backlog is not sorting it into folders.
It is deciding what to *delete*, one photo at a time, on a laptop, in a file
browser. That is the part that never gets done.

So: your computer holds the files and does the work, your phone is the remote
control, and the model watches what you keep.

```
┌─────────────┐        LAN         ┌──────────────┐
│  your PC    │◄──────────────────►│  your phone  │
│             │                    │              │
│  library    │  thumbnails  ──►   │  swipe deck  │
│  SQLite     │  ◄── swipes        │              │
│  the model  │                    └──────────────┘
└─────────────┘
```

Swipe right to keep, left to bin, up to favourite, down to decide later.
Every ~15 swipes the model retrains and re-ranks what it shows you next.

### Getting started

```bash
pip install -r requirements.txt          # includes rawpy, for RAW files
pip install pillow-heif                  # if you have iPhone .HEIC photos
winget install --id Gyan.FFmpeg          # if you want the model to see inside videos

# 1. Index the library. Reads EXIF, builds thumbnails, hashes for duplicates,
#    extracts features. Safe to re-run; it only does new or changed files.
python -m swipesort ingest --library "D:\Photos\Unsorted"

# 2. Serve the swipe UI to your phone (same Wi-Fi; it prints the LAN address).
python -m swipesort serve --library "D:\Photos\Unsorted"

# 3. Swipe. Then, when you are ready, see what it would do:
python -m swipesort apply --library "D:\Photos\Unsorted"

# 4. And do it:
python -m swipesort apply --library "D:\Photos\Unsorted" --confirm
```

Set `SWIPESORT_LIBRARY` once and you can drop the `--library` flag.

### What each kind of file needs

| your files | needs | what the model learns from |
|---|---|---|
| JPEG, PNG, WebP, TIFF | nothing extra | the picture and its metadata |
| RAW (`.raf` `.nef` `.arw` `.cr2` `.dng` ...) | `rawpy` (in `requirements.txt`) | the camera's embedded preview and its metadata |
| iPhone HEIC | `pip install pillow-heif` | the picture and its metadata |
| Videos | ffmpeg on `PATH` | one frame from ~2 s in, plus length, size and date |

**A file that cannot be previewed still counts.** You can swipe on it, and that
swipe still trains the model - on metadata alone (size, date, length, type).
`ingest` and `stats` list anything in that state, grouped by why, with the
install that fixes it; re-running `ingest` afterwards picks those files up. In
the app such cards say "No preview" and carry a "metadata only" badge. On an
iPhone, Safari can often display a HEIC the computer could not decode, so the
app tries the original file before giving up.

RAW files are read through the JPEG preview the camera embeds in them - what the
camera's own screen showed, film simulation and all - which is far faster than
developing the sensor data. The same preview carries the camera's EXIF, so RAW
capture dates work without exiftool. That matters for Fuji: `DSCF1234.RAF` has
no date in its name, so before this every RAF landed in `Unclassified RAWs`.

### Nothing is ever deleted

`apply` **moves** binned files to `<library>/_Quarantine/<batch>/`, keeping their
original relative path, and writes a `manifest.json` next to them. Every move is
logged in the index, so:

```bash
python -m swipesort undo                       # reverse the last batch
python -m swipesort undo --batch 20250413-2131 # or a specific one
```

puts the library back exactly as it was. Files only actually disappear when you
run `empty-quarantine --confirm`, which is the single destructive command in the
whole tool. `apply` with no `--confirm` is always a dry run.

### How the learning works

Each image becomes a vector, and a logistic regression learns P(you keep this)
from your swipes. It is deliberately small: a few hundred swipes is all the
training data it will ever have, so it has to work from cold and never be so
confident that it hides something you wanted.

**Features** come from one of two backends:

| backend | what it sees | needs |
|---|---|---|
| `classic` (default) | ~100 numbers: colour histograms, exposure, blown highlights and crushed shadows, sharpness overall and in the centre, edge density, entropy, and a 3×3 composition grid | numpy, Pillow |
| `clip` | CLIP ViT-B/32 embeddings - semantic, so it can learn "mountains yes, whiteboards no" | `pip install open_clip_torch torch` |

Both get metadata appended: file size, media type, time of day, month, whether
it is a screenshot, how many near-identical siblings it has, resolution, and
whether there was a picture to look at at all. For a file with no preview, the
image columns are filled with the training average - which contributes nothing -
and that last flag lets the model learn how such files differ on their own.
`classic` is very good at the statistical half of storage cleaning (blurry,
dark, badly framed). `clip` is better at subject matter, at the cost of a
~2 GB install. Pick with `ingest --backend clip`; switching re-extracts, since
the two feature spaces are not comparable.

**Ranking.** The queue has four orderings, because "what next" has different
answers depending on why you opened the app:

| mode | order | use it when |
|---|---|---|
| `learn` | closest to the model's 50/50 line first | you want the model good, fast |
| `clean` | confident rubbish, biggest files first | you want the disk space back |
| `keepers` | most-likely favourites first | you are looking for the good ones |
| `backlog` | oldest first | you want to grind through chronologically |

`learn` is active learning: the swipes that teach the model most are the ones it
is least sure about, so it asks about those first. In every mode, near-duplicate
groups arrive **together**, so the seven near-identical shots of the same doorway
show up in a row and you can keep the best one while the others are still fresh.
Favourite one of them and the app offers to bin the rest in a single tap.

**Honesty about accuracy.** `swipesort stats` reports held-out accuracy against
the always-guess-the-majority baseline. If the model is only matching the
baseline, it has not learned your taste yet - keep swiping, or try `--backend clip`.

### Duplicates

Two kinds, handled differently:

* **Exact** - byte-identical. Found by SHA-256, but only for files that already
  share a size, so a big library does not get fully hashed. These are linked to
  their original, never shown in the queue, and quarantined by `apply`.
* **Near** - burst frames, re-crops, re-encodes. Found with a 64-bit dHash and
  clustered by Hamming distance (default 6/64, `ingest --dup-threshold`). These
  are always shown; only you can say which frame is the good one.

Note that flat, low-detail images (a pocket shot, a black frame) genuinely do
hash alike, so they may cluster into one large group. That is correct - they
really are interchangeable - but if it bothers you, lower the threshold.

### Commands

| command | what it does |
|---|---|
| `ingest` | scan the library: metadata, thumbnails, hashes, features |
| `serve` | serve the swipe UI on the local network |
| `train` | fit the preference model on your swipes |
| `stats` | library and model numbers, including held-out accuracy |
| `queue` | print what the app would show next (handy for checking a mode) |
| `apply` | quarantine the binned files, file the keepers (dry run by default) |
| `undo` | reverse an apply batch |
| `empty-quarantine` | permanently delete quarantined files |

### A note on security

`serve` has no authentication and hands out your photo library over HTTP to
anything that can reach the port. Use it on a network you trust, and do not
port-forward it.

---

## What carried over from the PowerShell scripts

`swipesort/classify.py` is a direct port of the sorting rules, including the
guard-clause order that decides a `Screenshot*.png` is a screenshot before it is
a photo:

1. unknown extension → `Metadata`
2. `.mp` → `Live Photos`
3. name starts with `Screenshot` → `Screenshots`
4. name starts with `FB` or `received` → `Facebook`
5. otherwise `<year> Photos` / `<year> Videos` / `<year> RAWs`

Year detection follows the same preference: EXIF `DateTimeOriginal` first
(for RAW files, read from the embedded preview), then a `YYYYMMDD` run in the
filename, then `Unclassified`. Last-write-time stays
unused unless you pass `--use-mtime`, per the *"no lastwritetime as its
unreliable"* commit.

Four deliberate changes:

* **`.nef` is a RAW**, not a photo. Grouping Nikon raws with JPEGs made the
  `RAWs` folder useless.
* **EXIF actually runs.** The PowerShell script set `$env:EXIFTOOLPATH` but
  called `& $exifToolPath`, an unassigned variable, so the exiftool call threw
  on every file and silently fell through to filename matching. swipesort reads
  EXIF via Pillow - `DateTimeOriginal`, the moment the shutter fired, not the
  `DateTime` field beside it, which is when the file was last *modified* and
  would file an edited photo under the year it was edited. It parses MP4/MOV
  creation time straight out of the container, reads RAW dates from the
  camera's embedded preview via rawpy, and falls back to exiftool when it is
  installed. It looks at
  `$SWIPESORT_EXIFTOOL`, then `PATH`, then
  `C:\Program Files\exiftool-12.97_64\exiftool.exe` - the path the scripts
  assumed.

  Worth knowing if you use the bundled `exiftool-12.97_64.zip`: it unzips to
  `exiftool-12.97_64\exiftool(-k).exe`, not `exiftool.exe`. The `(-k)` name makes
  the executable pause for a keypress before exiting, which is handy when you
  double-click it and fatal when a script calls it. Rename it to `exiftool.exe`,
  as ExifTool's own README says to - the original PowerShell path never resolved
  without that step either.
* **Duplicates are matched by content, not name.** `Remove-DuplicateFiles`
  grouped by filename and hard-deleted every file but the first, so two
  different photos that happened to share a name lost one permanently.
* **Collisions get a counter, not a random suffix**, so a re-run is repeatable
  and the folder stays readable.

## Layout

```
swipesort/
  config.py         library paths and tunables
  classify.py       extension and filename rules (ported from the .ps1)
  capture_date.py   EXIF / QuickTime / exiftool date extraction
  phash.py          dHash and near-duplicate clustering
  features.py       image -> vector, classic and CLIP backends
  model.py          logistic regression, training, scoring
  db.py             SQLite schema; decisions are append-only
  ingest.py         the scan pipeline
  queue.py          the four orderings
  apply.py          reversible moves, quarantine, undo
  server.py         HTTP API
  web/              the phone UI (vanilla JS, no build step, no CDN)
  cli.py            python -m swipesort
tests/              55 tests, stdlib unittest, synthetic library fixtures
```

### Testing

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t tests                   # all 81
python -m unittest discover -s tests -t tests -v                # with names
python -m unittest discover -s tests -t tests -k ApplyTests     # one class
python -m unittest discover -s tests -t tests -k test_undo_restores_every_file
```

(`-s` is where the tests live, `-t` is the import root; both point at `tests/`
because that is also where `fixtures.py` sits. Use `-k` to narrow rather than
naming `test_swipesort.SomeClass` directly - that form only resolves from
inside `tests/`.)

There are no committed image files, and no mocks beyond pretending a tool is or
is not installed. `tests/fixtures.py` generates a
synthetic library on disk in a temp directory - bright detailed "keepers", dark
flat "junk", a five-frame burst, a byte-identical copy, a screenshot, a
`received_*.jpg`, and a non-media sidecar. It can also write a real DNG (RAW)
file, with or without an embedded preview, and a minimal MP4 with no video
stream. The tests then run the real
pipeline over it: ingest, dedup, train, rank, apply, undo, and the HTTP API
through `fastapi.testclient`. Everything is torn down afterwards.

| class | covers |
|---|---|
| `ClassifyTests` (5) | the ported bucket rules and their precedence |
| `HashTests` (3) | dHash distance, clustering, threshold strictness |
| `CaptureDateTests` (6) | date taken vs date modified, QuickTime `mvhd`, timestamp formats, mtime opt-in |
| `ImageLoadingTests` (3) | EXIF orientation, true size under reduced-scale decoding |
| `RawTests` (8) | real LibRaw decoding of generated DNGs: embedded preview, full demosaic, preview dates, LibRaw's 1970 placeholder, running without rawpy |
| `FeatureTests` (3) | fixed width, finite values, sharp vs blurry |
| `ModelTests` (6) | fitting, regularisation, imputation of missing image columns, JSON round trip, AUC edges |
| `MetadataOnlyTests` (7) | swipes on unpreviewable files train and score; a library with no previews at all; two backends' vectors at once; the ingest report and its fixes |
| `IngestTests` (9) | indexing, incremental rescan, duplicates, missing files, walk order |
| `ModelOnLibraryTests` (3) | it beats the baseline, and refuses to train when it should |
| `QueueTests` (9) | all four orderings, filters, deferral, group cohesion |
| `ApplyTests` (9) | dry runs, quarantine, undo, name collisions, pruning |
| `ApiTests` (10) | every endpoint, plus rejection of bad input |

The ones worth knowing about, because they encode promises rather than
behaviour: `test_dry_run_moves_nothing`, `test_undo_restores_every_file` (which
snapshots every path before and after and demands they match), and
`test_same_name_different_content_keeps_both`.

None of them may depend on the order the filesystem returns directory entries
in. `os.walk` order is not sorted and not stable between machines, and an
earlier version of these tests assumed a particular one - it passed on three
runners and failed on the fourth. Ingest now walks in sorted order, and tests
that involve the duplicate pair assert the invariant (exactly one of the two is
flagged, both files still exist) rather than naming which.

A full run takes about 50 seconds; most of that is generating and decoding the
fixture images.

### CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

* **ruff** over `swipesort/` and `tests/`, pinned to an exact version so a new
  linter release cannot turn the build red on its own.
* **the test suite** on Python 3.10 and 3.12, on both Ubuntu and Windows.
  Windows is in the matrix deliberately - this tool is aimed at a library
  sitting on a Windows PC, and paths, moves and file locking are exactly what
  behaves differently there. The matrix earned its keep on the first run, by
  catching a filesystem-ordering bug that all local testing had missed.
* **a CLI smoke test** that builds a fixture library and drives `ingest`,
  `stats`, `queue` and `apply` through the actual command line, then asserts
  that `apply` without `--confirm` moved nothing. The unit tests go through the
  Python API; this catches a broken entry point or argument that they would not.

## Requirements and limits

* Python 3.10+.
* See [What each kind of file needs](#what-each-kind-of-file-needs). In short:
  videos want ffmpeg (and `ffprobe`, which ships with it, for length and
  resolution); HEIC wants `pillow-heif`. Without them those files are still
  indexed, sortable and trainable, but the model learns only their metadata.
* **Even with ffmpeg, a video is thinly represented** - one frame plus its
  length and size. It is enough to tell a pocket recording from a sunset, not to
  judge a whole clip.
* **RAW support has been tested on DNG, not on a real RAF.** Both go through the
  same rawpy calls, and the tests decode a generated DNG through real LibRaw on
  Windows and Linux, but no camera RAF is in the repository. LibRaw's Fuji
  support is mature; if a new body is too recent for it, those files show up in
  the `ingest` report rather than failing silently.
* **One model per library.** Training does not carry across folders - to share
  one model across every trip, point `--library` at the folder containing them.
* The model learns *your* taste from *your* swipes. It starts as a coin flip and
  needs roughly 25 swipes before it says anything at all.
