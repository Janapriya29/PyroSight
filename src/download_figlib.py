"""
src/download_figlib.py
-----------------------
Downloads the five verified FIgLib external evaluation sequences from the
official HPWREN CDN, saves JPEGs to data/figlib_external/images/ with
original filenames unchanged, then runs full post-download checks.

Sequences downloaded (Phase 8.2 verified):
  1. 20221116-Willowfire-om-n-mobo-c
  2. 20200806_SpringsFire_om-n-mobo-c
  3. 20191006_FIRE_lp-n-mobo-c
  4. 20190924_FIRE_ml-w-mobo-c
  5. 20180806_Holy_sp-s-mobo-c

Usage:
    python -m src.download_figlib
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CDN_BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"

SEQUENCES = [
    "20221116-Willowfire-om-n-mobo-c",
    "20200806_SpringsFire_om-n-mobo-c",
    "20191006_FIRE_lp-n-mobo-c",
    "20190924_FIRE_ml-w-mobo-c",
    "20180806_Holy_sp-s-mobo-c",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR     = PROJECT_ROOT / "data" / "figlib_external" / "images"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Label rule: negative offset → NO_SMOKE (0), zero/positive → SMOKE (1)
# Handles literal '+' and URL-encoded '%2B' (server percent-encodes the plus sign)
OFFSET_RE = re.compile(r"_(-|\+|%2B)(\d+)\.jpg$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "PyroSight/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _download_file(url: str, dest: Path) -> bool:
    """Download url → dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PyroSight/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"    [ERROR] {url}: {exc}")
        return False


def get_jpeg_links(index_url: str) -> list[str]:
    """Parse the per-sequence index.html and return all .jpg filenames."""
    # Each sequence has its own index.html listing individual JPEG hrefs
    page_url = index_url.rstrip("/") + "/index.html"
    html     = _fetch_text(page_url)
    names    = re.findall(r'href=([^\s>]+\.jpg)', html, re.IGNORECASE)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for n in names:
        base = Path(n).name
        if base not in seen:
            seen.add(base)
            unique.append(base)
    return unique


def assign_label(filename: str) -> int | None:
    """Return 0 (NO_SMOKE) or 1 (SMOKE) from filename offset, or None if unparseable.

    Handles both literal '+' and URL-encoded '%2B' (plus sign) as SMOKE,
    and '-' as NO_SMOKE.
    """
    m = OFFSET_RE.search(filename)
    if not m:
        return None
    sign = m.group(1).upper()
    return 0 if sign == "-" else 1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_sequences() -> dict[str, list[str]]:
    """Download all five sequences. Returns {seq_name: [downloaded_filenames]}."""
    downloaded: dict[str, list[str]] = {}

    for seq in SEQUENCES:
        index_url = f"{CDN_BASE}/{seq}/"
        print(f"\n{'='*60}")
        print(f"Sequence: {seq}")
        print(f"URL     : {index_url}")

        try:
            jpeg_names = get_jpeg_links(index_url)
        except Exception as exc:
            print(f"  [ERROR] Could not fetch directory listing: {exc}")
            downloaded[seq] = []
            continue

        print(f"  Found {len(jpeg_names)} JPEG links in listing")

        seq_downloaded: list[str] = []
        for i, name in enumerate(jpeg_names, 1):
            dest = DEST_DIR / name
            if dest.exists():
                seq_downloaded.append(name)
                if i == 1 or i % 20 == 0:
                    print(f"  [{i:3d}/{len(jpeg_names)}] Already exists: {name}")
                continue
            file_url = f"{index_url}{name}"
            ok = _download_file(file_url, dest)
            if ok:
                seq_downloaded.append(name)
                if i == 1 or i % 20 == 0 or i == len(jpeg_names):
                    print(f"  [{i:3d}/{len(jpeg_names)}] Downloaded: {name}")
            time.sleep(0.05)  # polite delay

        print(f"  Sequence complete: {len(seq_downloaded)}/{len(jpeg_names)} files saved")
        downloaded[seq] = seq_downloaded

    return downloaded


# ---------------------------------------------------------------------------
# Post-download checks
# ---------------------------------------------------------------------------

def run_checks(downloaded: dict[str, list[str]]) -> dict:
    print(f"\n{'='*60}")
    print("POST-DOWNLOAD CHECKS")
    print(f"{'='*60}")

    all_files  = [name for names in downloaded.values() for name in names]
    total      = len(all_files)

    # 1. Total count
    print(f"\nTotal images downloaded : {total}")

    # 2. Label counts per sequence and overall
    overall_labels: Counter = Counter()
    seq_stats: list[dict]   = []

    for seq, names in downloaded.items():
        smoke = no_smoke = unlabelled = 0
        for n in names:
            lbl = assign_label(n)
            if lbl == 1:
                smoke += 1
            elif lbl == 0:
                no_smoke += 1
            else:
                unlabelled += 1
        overall_labels[0] += no_smoke
        overall_labels[1] += smoke
        seq_stats.append({
            "sequence": seq,
            "total": len(names),
            "smoke": smoke,
            "no_smoke": no_smoke,
            "unlabelled": unlabelled,
        })
        print(f"\n  {seq}")
        print(f"    Total    : {len(names)}")
        print(f"    SMOKE(+) : {smoke}")
        print(f"    NO_SMOKE : {no_smoke}")
        if unlabelled:
            print(f"    Unlabelled (parse error): {unlabelled}")

    print(f"\n--- Overall label counts ---")
    print(f"  NO_SMOKE (0) : {overall_labels[0]}")
    print(f"  SMOKE    (1) : {overall_labels[1]}")
    if total > 0:
        print(f"  NO_SMOKE %   : {overall_labels[0]/total*100:.1f}%")
        print(f"  SMOKE %      : {overall_labels[1]/total*100:.1f}%")

    # 3. Duplicate filename check
    print(f"\n--- Duplicate filename check ---")
    name_counts = Counter(all_files)
    dupes = {k: v for k, v in name_counts.items() if v > 1}
    if dupes:
        print(f"  DUPLICATES FOUND: {len(dupes)}")
        for k, v in list(dupes.items())[:10]:
            print(f"    {k}: {v} copies")
    else:
        print(f"  No duplicate filenames.")

    # 4. JPEG integrity check
    print(f"\n--- JPEG integrity check ---")
    corrupt    = []
    non_jpeg   = []
    dimensions = []

    for name in all_files:
        path = DEST_DIR / name
        if not path.suffix.lower() in {".jpg", ".jpeg"}:
            non_jpeg.append(name)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open to read dimensions (verify closes the file)
            with Image.open(path) as img:
                dimensions.append(img.size)  # (W, H)
        except Exception as exc:
            corrupt.append((name, str(exc)))

    print(f"  Files checked   : {total}")
    print(f"  Non-JPEG files  : {len(non_jpeg)}")
    print(f"  Corrupt / unreadable: {len(corrupt)}")
    if corrupt:
        for fn, err in corrupt[:5]:
            print(f"    {fn}: {err}")

    if dimensions:
        widths  = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        print(f"  Width  range    : {min(widths)} – {max(widths)} px")
        print(f"  Height range    : {min(heights)} – {max(heights)} px")

    integrity_ok = (len(corrupt) == 0 and len(non_jpeg) == 0)
    print(f"  Integrity check : {'PASS' if integrity_ok else 'FAIL'}")

    print(f"\n{'='*60}")
    print(f"ALL CHECKS COMPLETE")
    print(f"{'='*60}")

    return {
        "total":          total,
        "smoke":          overall_labels[1],
        "no_smoke":       overall_labels[0],
        "duplicates":     len(dupes),
        "corrupt":        len(corrupt),
        "non_jpeg":       len(non_jpeg),
        "integrity_ok":   integrity_ok,
        "seq_stats":      seq_stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    downloaded = download_sequences()
    results    = run_checks(downloaded)
    sys.exit(0 if results["integrity_ok"] else 1)
