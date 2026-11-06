#!/usr/bin/env python3
"""Write black-background foreground images from the public CFR SAM3 masks.

Inputs are the 518 px images and combined_all.csv produced by the official
wildlife-tools preparation notebook. Masks come from the Hugging Face dataset
imageomics/cfr-wildlife-masks. Each foreground image is written as
<output-root>/<path>.png, where <path> is the row's path in combined_all.csv.
Rows without an accepted mask get no file, so training uses their full image.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from pycocotools import mask as mask_utils

from common import MASKS_REPO, MASKS_REVISION, TRAIN_DATASETS, foreground_path

# Mask and image may differ in size only by rounding when a mask was made on a
# source crop instead of the prepared image (ATRW detection-test crops).
MAX_ASPECT_RATIO_DIFF = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder that contains images/size-518 and metadata/combined/combined_all.csv.",
    )
    parser.add_argument("--output-root", type=Path, help="Default: <data-root>/images/size-518-fg.")
    parser.add_argument("--masks-dir", type=Path, help="Local copy of the masks dataset. Default: download it.")
    parser.add_argument("--revision", default=MASKS_REVISION, help="Revision of the masks dataset to download.")
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Overwrite existing foreground images.")
    return parser.parse_args()


def load_mask_index(masks_dir: Path, dataset: str) -> tuple[dict, dict, dict] | None:
    """Return (rle by annotation_id, annotation_id by path, annotation_id by image_id)."""
    folder = masks_dir / "data" / dataset
    if not (folder / "metadata.parquet").is_file():
        return None
    metadata = pd.read_parquet(folder / "metadata.parquet")
    with gzip.open(folder / "masks.json.gz", "rt") as stream:
        records = json.load(stream)["annotations"]
    rles = {str(r["annotation_id"]): r["segmentation"] for r in records if r["present"]}

    accepted = metadata[metadata["accepted"]]
    # Most masks were made on the prepared 518 px image itself.
    prepared = accepted[accepted["input_root_label"] == "prepared_518"]
    by_path = dict(zip(dataset + "/" + prepared["input_image"].astype(str), prepared["annotation_id"].astype(str)))
    # The rest were made on a crop of the source photo; match them by image_id.
    other = accepted[accepted["input_root_label"] != "prepared_518"]
    other = other.drop_duplicates("image_id", keep=False)
    by_image_id = dict(zip(other["image_id"].astype(str), other["annotation_id"].astype(str)))
    return rles, by_path, by_image_id


def decode(rle: dict) -> np.ndarray:
    rle = dict(rle)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def process_dataset(task: dict) -> dict:
    dataset = task["dataset"]
    image_root = Path(task["image_root"])
    output_root = Path(task["output_root"])
    counts = {"dataset": dataset, "rows": len(task["rows"]), "written": 0, "existing": 0, "no_mask": 0, "size_mismatch": 0}

    index = load_mask_index(Path(task["masks_dir"]), dataset)
    if index is None:
        counts["no_mask"] = counts["rows"]
        return counts
    rles, by_path, by_image_id = index

    for row in task["rows"]:
        output_path = foreground_path(output_root, row["path"])
        if output_path.is_file() and not task["force"]:
            counts["existing"] += 1
            continue
        on_prepared = row["path"] in by_path
        annotation_id = by_path.get(row["path"]) or by_image_id.get(str(row["image_id"]))
        if annotation_id is None or annotation_id not in rles:
            counts["no_mask"] += 1
            continue

        image = np.asarray(Image.open(image_root / row["path"]).convert("RGB")).copy()
        mask = decode(rles[annotation_id])
        if mask.shape != image.shape[:2]:
            mask_ratio = mask.shape[1] / mask.shape[0]
            image_ratio = image.shape[1] / image.shape[0]
            if on_prepared or abs(mask_ratio / image_ratio - 1) > MAX_ASPECT_RATIO_DIFF:
                counts["size_mismatch"] += 1
                continue
            size = (image.shape[1], image.shape[0])
            mask = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))

        image[~mask] = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)
        counts["written"] += 1
    return counts


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    image_root = data_root / "images" / "size-518"
    output_root = (args.output_root or data_root / "images" / "size-518-fg").expanduser().resolve()
    metadata = pd.read_csv(data_root / "metadata" / "combined" / "combined_all.csv")
    metadata = metadata[metadata["dataset"].isin(args.datasets)]

    if args.masks_dir:
        masks_dir = args.masks_dir.expanduser().resolve()
    else:
        masks_dir = Path(
            snapshot_download(
                MASKS_REPO,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=[f"data/{name}/*" for name in args.datasets],
            )
        )
    print(f"Masks: {masks_dir}\nImages: {image_root}\nOutput: {output_root}", flush=True)

    tasks = [
        {
            "dataset": name,
            "rows": frame[["path", "image_id"]].to_dict("records"),
            "image_root": str(image_root),
            "output_root": str(output_root),
            "masks_dir": str(masks_dir),
            "force": args.force,
        }
        for name, frame in metadata.groupby("dataset")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_dataset, tasks))

    summary = pd.DataFrame(results).sort_values("dataset")
    summary["with_foreground"] = summary["written"] + summary["existing"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "coverage.csv", index=False)
    print(summary.to_string(index=False))
    total = summary[["rows", "with_foreground"]].sum()
    print(f"Foreground available for {total['with_foreground']} / {total['rows']} training images.")


if __name__ == "__main__":
    main()
