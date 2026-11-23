# photo-and-video-sort

Two Windows PowerShell scripts for getting a messy photo and video folder into
`<year> <type>` folders.

The swipe-to-triage tool that used to live here (`swipesort/`) moved, with its
history, to [julianxyt/swipesort](https://github.com/julianxyt/swipesort) as
`desktop/`, next to the SwipeSort phone app. It reads real EXIF capture dates and
finds duplicates by content, so it is the better tool for camera files and for
duplicates. The notes below say where.

## The PowerShell scripts

The two original scripts. Neither takes parameters: **both work on the folder you
are standing in**, so `cd` to the folder you want handled first, then run the
script by its full path. Windows PowerShell 5.1 is enough; there is nothing to
install. If PowerShell refuses to run a `.ps1` because it came from a downloaded
zip, run `Unblock-File` on it once.

### Split_To_Years_Greedy_EXIF.ps1

Sorts every file in the current folder **and all of its subfolders** into folders
it creates in the current folder. The first rule that matches wins:

| order | rule | goes to |
|---|---|---|
| 1 | extension is not one of the photo, video or RAW types below | `Metadata\` |
| 2 | `.mp` | `Live Photos\` |
| 3 | name starts with `Screenshot` | `Screenshots\` |
| 4 | name starts with `FB`, or contains `received` | `Facebook\` |
| 5 | `.jpg` `.jpeg` `.heic` `.png` `.nef` | `<year> Photos\` |
| 5 | `.mp4` `.avi` `.mov` `.mkv` `.wmv` `.flv` | `<year> Videos\` |
| 5 | `.raf` | `<year> RAWs\` |

The year comes from a `YYYYMMDD` run in the file name, so `IMG_20230415_101010.jpg`
becomes `2023 Photos\`. A name with no date goes to `Unclassified Photos\`,
`Unclassified Videos\` or `Unclassified RAWs\`.

**Camera files are the catch.** The script is written to read the EXIF capture
date first, but that step never runs (see
[swipesort desktop: What carried over](https://github.com/julianxyt/swipesort/blob/main/desktop/README.md#what-carried-over-from-the-powershell-scripts)), so the year
always comes from the name. Files straight off a Fuji - `DSCF1234.JPG`, `.RAF`,
`.MOV` - have no date in their names and all land in `Unclassified ...`. For
those, use [swipesort desktop](https://github.com/julianxyt/swipesort/tree/main/desktop), which reads the real capture date.

```powershell
cd "D:\Photos\Unsorted"
& "C:\path\to\photo-and-video-sort\Split_To_Years_Greedy_EXIF.ps1"
```

Before you run it:

* **It moves files, and there is no dry run.** Try it on a copy of a few first.
* **Run it once per unsorted folder.** It recurses into everything beneath you,
  including the year folders from a previous run, and each already-sorted file
  then collides with *itself* and is moved into `Recycle Bin\`. Nothing is lost,
  but putting it all back is manual.
* **`Recycle Bin\`** is a plain folder the script creates in the current folder -
  not the Windows one. A file whose name already exists at its destination is
  parked there instead of overwriting anything. If that name is already in
  `Recycle Bin\` too, a byte-identical file (SHA-256) is deleted and a different
  one is kept as `name(1234).ext`. Look through it, then delete it yourself.

### Remove-DuplicateFiles.txt

This is a PowerShell script saved with a `.txt` extension, so PowerShell will not
run it by name (and double-clicking just opens it in a text editor). To run it
without renaming anything:

```powershell
cd "D:\Photos\Some Folder"
Get-Content -Raw "C:\path\to\photo-and-video-sort\Remove-DuplicateFiles.txt" | Invoke-Expression
```

**As shipped it does nothing.** `-Recurse` is commented out on line 7, so it looks
at one folder, and one folder cannot hold two files with the same name. It prints
`Found file with the name:` (always blank - a leftover debug line) and
`Duplicate removal process completed.`, and removes nothing.

**Do not "fix" that by deleting the `#`.** With `-Recurse` on, it groups files by
**name only**, keeps the first one it finds, and permanently deletes - `Remove-Item
-Force`, no Recycle Bin - every other file with that name in any subfolder,
without ever comparing contents. On test files, two *different* photos both named
`a.jpg` lost one. Cameras restart their numbering and phones reuse names, so on a
real library that is real photos gone. For duplicates matched by content, and
reversibly, use [swipesort desktop](https://github.com/julianxyt/swipesort/tree/main/desktop) - see its Duplicates section.

(Both behaviours above were confirmed on placeholder files, not a real library.)

