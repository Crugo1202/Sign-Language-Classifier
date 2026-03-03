## LSTM Sign Language Classifier

This project implements an LSTM-based sign language classifier on the WLASL dataset using MediaPipe hand keypoints and PyTorch.

### Project structure

- **`requirements.txt`**: Python dependencies.
- **`config.py`**: Paths and hyperparameters.
- **`data/`**: Place WLASL JSON and videos here (see below).
- **`src/extract_keypoints.py`**: Extract MediaPipe hand landmarks from videos into `.npy` sequences and build metadata.
- **`src/dataset.py`**: PyTorch dataset and collate utilities.
- **`src/model.py`**: BiLSTM classifier with attention pooling.
- **`src/train.py`**: Training loop (train/val).
- **`src/evaluate.py`**: Offline evaluation on the test split.

### Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download WLASL (e.g. from the official repo) and arrange files as:

```text
data/
  wlasl/
    WLASL_v0.3.json
    videos/
      <video_id>.mp4
    keypoints/        # created by extract_keypoints.py
    metadata.csv      # created by extract_keypoints.py
```

### Keypoint extraction

From the repo root:

```bash
python -m src.extract_keypoints \
  --json data/wlasl/WLASL_v0.3.json \
  --videos data/wlasl/videos \
  --out data/wlasl/keypoints \
  --metadata data/wlasl/metadata.csv
```

This will run MediaPipe Hands on each WLASL instance and write sequences as `.npy` files plus a `metadata.csv` mapping filenames to gloss and split.

### Training

```bash
python -m src.train \
  --metadata data/wlasl/metadata.csv \
  --keypoints data/wlasl/keypoints \
  --epochs 20
```

The best model checkpoint is saved under `checkpoints/`.

### Evaluation

```bash
python -m src.evaluate \
  --checkpoint checkpoints/best_model.pt \
  --metadata data/wlasl/metadata.csv \
  --keypoints data/wlasl/keypoints
```

This yields overall test accuracy, per-class accuracy, and a confusion matrix (`confusion_matrix.npy`).

