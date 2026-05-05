# DSLNet-Upgraded (WLASL)

This repository is scaffolded from your upgraded DSLNet plan:

- Dual-hand handshape encoding (`TSSN-R` + `TSSN-L`)
- Dual-wrist trajectory encoder (`FTDE-Dual`)
- Face non-manual stream (`NMN`)
- 3-way fusion with geometric consistency losses

## Project Structure

```text
data/
  preprocess.py
models/
  tssn.py
  ftde.py
  nmn.py
  fusion.py
  dslnet_upgraded.py
  losses.py
train.py
eval.py
config.yaml
```

## Obtaining WLASL (official download)

This training code does not fetch videos automatically. Use the **official WLASL repository** and its scripts:

- Repository: [dxli94/WLASL](https://github.com/dxli94/WLASL)
- Read the **Computational Use of Data Agreement (C-UDA)** in that repo before using the data.
- Typical flow (see their `README`):
  1. `git clone https://github.com/dxli94/WLASL.git`
  2. In `start_kit/`, run `python video_downloader.py` (uses `yt-dlp` to pull YouTube sources listed in the JSON).
  3. Run `python preprocess.py` to cut clips into `videos/` as in their instructions.
- If many URLs are dead, their README describes `find_missing.py` and a form to request missing or pre-processed videos.

After you have video clips, you still need a **skeleton extraction** step (e.g. MediaPipe Holistic) to produce the arrays described below, then save one `.npz` per instance for this project.

## Data Format

Put sample `.npz` files in:

- `data/skeletons/train`
- `data/skeletons/val`
- `data/skeletons/test`

Each `.npz` must contain `label`, and either:

1) Preprocessed keys:
- `right_hand_shape` `(T, 21, 3)`
- `left_hand_shape` `(T, 21, 3)`
- `dual_wrist_traj` `(T, 6)`
- `face_landmarks_norm` `(T, K, 3)` where `K` is fixed semantic facial keypoints (recommended 60-80)

or 2) Raw keys (will be preprocessed on load):
- `right_hand` `(T, 21, 3)`
- `left_hand` `(T, 21, 3)`
- `face` `(T, 468/478, 3)` or selected facial keypoints `(T, K, 3)`
- `pose` `(T, 33, 3)`

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --config config.yaml
```

## Evaluate

```bash
python eval.py --config config.yaml --checkpoint checkpoints/dslnet_upgraded_best.pt
```

## Notes

- `data/preprocess.py` includes:
  - wrist-centric and face-centric normalization
  - missing landmark interpolation
  - temporal resampling to `T_max`
  - augmentation hooks (rotation, scaling, noise, temporal stretch, mirroring)
- Hyperparameters match the plan defaults in `config.yaml`.
