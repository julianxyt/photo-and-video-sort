# photo-and-video-sort

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
pip install -r requirements.txt

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
it is a screenshot, how many near-identical siblings it has, and resolution.
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

Year detection follows the same preference: EXIF `DateTimeOriginal` first, then
a `YYYYMMDD` run in the filename, then `Unclassified`. Last-write-time stays
unused unless you pass `--use-mtime`, per the *"no lastwritetime as its
unreliable"* commit.

Four deliberate changes:

* **`.nef` is a RAW**, not a photo. Grouping Nikon raws with JPEGs made the
  `RAWs` folder useless.
* **EXIF actually runs.** The PowerShell script set `$env:EXIFTOOLPATH` but
  called `& $exifToolPath`, an unassigned variable, so the exiftool call threw
  on every file and silently fell through to filename matching. swipesort reads
  EXIF via Pillow, parses MP4/MOV creation time straight out of the container,
  and falls back to exiftool for RAW files when it is installed. It looks at
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

Run the tests with:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t tests
```

They build a synthetic library on disk and exercise the real pipeline
end to end - ingest, dedup, train, rank, apply, undo - so they need no photos
of yours and leave nothing behind.

## Requirements and limits

* Python 3.10+.
* **Videos** need `ffmpeg` on `PATH` for thumbnails and features, and `ffprobe`
  for dimensions and duration. Without them videos are still indexed and
  sortable, just judged on metadata alone.
* **HEIC** needs `pip install pillow-heif`.
* **RAW** files are indexed and sorted, but their thumbnails depend on Pillow
  being able to read an embedded preview; some formats will not render.
* The model learns *your* taste from *your* swipes. It starts as a coin flip and
  needs roughly 25 swipes before it says anything at all.
