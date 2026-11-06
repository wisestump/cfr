"""Constants shared by the foreground-mix training scripts."""

from __future__ import annotations

from pathlib import Path

# The 22 training datasets. These are the datasets prepared by the official
# wildlife-tools notebook (baselines/data/prepare_datasets.ipynb), minus SealID,
# LeopardID2022 and five datasets kept unseen for evaluation: NyalaData,
# SeaTurtleIDHeads, HyenaID2022, StripeSpotter and WhaleSharkID.
TRAIN_DATASETS = (
    "AAUZebraFish",
    "ATRW",
    "AerialCattle2017",
    "BelugaID",
    "BirdIndividualID",
    "CTai",
    "CZoo",
    "Cows2021",
    "FriesianCattle2015",
    "FriesianCattle2017",
    "GiraffeZebraID",
    "Giraffes",
    "HappyWhale",
    "HumpbackWhaleID",
    "IPanda50",
    "LionData",
    "MacaqueFaces",
    "NDD20",
    "NOAARightWhale",
    "OpenCows2020",
    "SMALST",
    "ZindiTurtleRecall",
)

MASKS_REPO = "imageomics/cfr-wildlife-masks"
# Pinned dataset revision; pass --revision main to use the latest masks.
MASKS_REVISION = "07cd2a5f2f8a1418651b775667bffd404598eb46"


def foreground_path(foreground_root: Path, path: str) -> Path:
    """Foreground PNG for a prepared image path from combined_all.csv."""
    return Path(foreground_root) / Path(path).with_suffix(".png")
