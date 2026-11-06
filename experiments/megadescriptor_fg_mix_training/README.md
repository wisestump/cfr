# MegaDescriptor Training on Mixed Foreground and Full Images

Trains MegaDescriptor-L-384 with the public
[wildlife-tools recipe](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-L-384.ipynb).
With probability `p`, each training image is replaced by its foreground-only
version (background set to black), made from the
[CFR SAM3 masks](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks).
Images without a mask use the full image. `p = 0` is the unchanged recipe.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the datasets

Use the official wildlife-tools preparation at the pinned commit:

```bash
git clone https://github.com/WildlifeDatasets/wildlife-tools.git
cd wildlife-tools && git checkout 54449df7a4e4ffd2406e3c4f5c2bdd1fe46ec878
```

Run `baselines/data/prepare_datasets.ipynb` (518 px only). `DATA_ROOT` below is
the folder that holds `images/size-518` and
`metadata/combined/combined_all.csv`. Training uses the 22 datasets listed in
`common.py`.

## 2. Make the foreground images

```bash
python make_foregrounds.py --data-root "$DATA_ROOT" --workers 8
```

This writes the foreground images to `$DATA_ROOT/images/size-518-fg`.

## 3. Train

```bash
python train_mix.py --data-root "$DATA_ROOT" --foreground-prob 0.25 --output-dir runs/fg25
```

Run it with `p` = 0, 0.25, 0.5 and 0.75. On SLURM, submit from this folder:

```bash
sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> \
  --export=ALL,DATA_ROOT=$DATA_ROOT,FOREGROUND_PROB=0.25,OUTPUT_DIR=$PWD/runs/fg25 \
  slurm/train_mix.sh
```

To resume after a job stops, rerun the same command. The output is
`megadescriptor-l-384-backbone.pt`, a state dict for the timm model
`swin_large_patch4_window12_384` with `num_classes=0`.
