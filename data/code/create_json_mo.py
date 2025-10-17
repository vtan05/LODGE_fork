#!/usr/bin/env python3
# python create_json_mo.py /path/to/npy -o /path/to/json_out

import argparse
import json
import re
from pathlib import Path
import numpy as np

CH_MARKER_RE = re.compile(r"^ch\d+$", re.IGNORECASE)

def parse_filename(p: Path):
    """
    Extract style1 (strip leading 'kth' if present), style2, and name (title+index)
    from filenames of the form:
        kthjazz_gCH_sFM_cAll_d02_mCH_ch01_<name_with_underscores_and_index>.npy
    """
    stem = p.stem  # filename without extension
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {p.name}")

    # style1 from first token, strip 'kth' prefix if present
    style1_raw = parts[0]
    style1 = style1_raw[3:] if style1_raw.lower().startswith("kth") else style1_raw

    # style2 is the next token
    style2 = parts[1]

    # find 'ch##' marker to locate where the title begins
    ch_idx = None
    for i, token in enumerate(parts):
        if CH_MARKER_RE.match(token):
            ch_idx = i
            break

    if ch_idx is None or ch_idx == len(parts) - 1:
        # If no 'ch##' found, fallback: name is everything after the fixed header tokens
        # (This is a safe fallback but your dataset should have 'ch##')
        name_parts = parts[2:]
    else:
        name_parts = parts[ch_idx + 1 :]

    if not name_parts:
        raise ValueError(f"Could not parse name from filename: {p.name}")

    name = "_".join(name_parts)
    return style1, style2, name

def count_frames(npy_path: Path) -> int:
    # Use mmap to avoid loading the whole array into RAM
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim == 0:
        raise ValueError(f"Array has no dimensions (scalar): {npy_path.name}")
    # Convention: T is the first dimension
    return int(arr.shape[0])

def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(input_dir.glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in: {input_dir}")
        return

    for npy_path in npy_files:
        try:
            style1, style2, name = parse_filename(npy_path)
            frames = count_frames(npy_path)

            data = {
                "name": name,
                "style1": style1,
                "style2": style2,
                "frames": frames,  # 30fps assumed; frames = T
            }

            json_path = output_dir / (npy_path.stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"Wrote: {json_path.name} -> {data}")

        except Exception as e:
            print(f"⚠️ Skipped {npy_path.name}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Create JSON metadata for motion .npy files.")
    ap.add_argument("input_dir", type=Path, help="Folder containing .npy files")
    ap.add_argument(
        "-o", "--output-dir", type=Path,
        default=None,
        help="Folder to write .json files (default: same as input)"
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
