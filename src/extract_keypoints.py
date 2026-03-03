import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import DATA_ROOT, WLASL_JSON, VIDEO_DIR, KEYPOINT_DIR, METADATA_PATH


mp_hands = mp.solutions.hands


def load_wlasl_json(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_hand_keypoints_from_frame(
    image_bgr: np.ndarray,
    hands: mp_hands.Hands,
    expect_both_hands: bool = True,
) -> np.ndarray:
    """
    Returns a 1D numpy array of length 126 (21 * 2 * 3).
    If hands are missing, corresponding entries are zeroed.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 21 landmarks * 3 coords per hand
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label.lower()
            target = left if label == "left" else right
            for i, lm in enumerate(hand_landmarks.landmark):
                target[i] = [lm.x, lm.y, lm.z]

    if not expect_both_hands:
        # Use only right hand if available, otherwise left, flattened to 63 dims
        if np.any(right):
            return right.reshape(-1)
        return left.reshape(-1)

    return np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)


def process_instance(
    instance: Dict,
    gloss: str,
    hands: mp_hands.Hands,
    output_dir: Path,
    expect_both_hands: bool = True,
) -> Optional[Tuple[str, str, str]]:
    """
    Process a single WLASL instance and save keypoints as .npy.
    Returns (filename, gloss, split) metadata or None on failure.
    """
    video_id = instance["video_id"]
    split = instance.get("split", "train")
    frame_start = instance.get("frame_start", 0)
    frame_end = instance.get("frame_end", None)

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    # WLASL is specified at 25 FPS; we assume provided frame indices match.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_end is None or frame_end <= 0 or frame_end > total_frames:
        frame_end = total_frames - 1

    frame_idx = 0
    seq: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx > frame_end:
            break

        if frame_idx >= frame_start:
            keypoints = extract_hand_keypoints_from_frame(
                frame, hands, expect_both_hands=expect_both_hands
            )
            seq.append(keypoints)

        frame_idx += 1

    cap.release()

    if not seq:
        return None

    sequence = np.stack(seq, axis=0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use video_id plus an index to avoid collisions if needed
    instance_idx = instance.get("instance_id", 0)
    filename = f"{video_id}_{instance_idx}.npy"
    np.save(output_dir / filename, sequence)

    return filename, gloss, split


def build_metadata(
    records: List[Tuple[str, str, str]], metadata_path: Path
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gloss", "split"])
        for filename, gloss, split in records:
            writer.writerow([filename, gloss, split])


def extract_all_keypoints(
    json_path: Path,
    output_dir: Path,
    metadata_path: Path,
    subset: Optional[str] = None,
    expect_both_hands: bool = True,
) -> None:
    data = load_wlasl_json(json_path)

    # Optionally filter to WLASL100 / 300 / 1000 by gloss frequency
    if subset is not None:
        # Assume JSON is already subsetted; in practice you may want to
        # filter here by a provided gloss list.
        pass

    records: List[Tuple[str, str, str]] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for entry in tqdm(data, desc="Extracting keypoints"):
            gloss = entry["gloss"]
            for i, instance in enumerate(entry.get("instances", [])):
                instance = dict(instance)
                instance.setdefault("instance_id", i)
                meta = process_instance(
                    instance,
                    gloss=gloss,
                    hands=hands,
                    output_dir=output_dir,
                    expect_both_hands=expect_both_hands,
                )
                if meta is not None:
                    records.append(meta)

    build_metadata(records, metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand keypoints from WLASL videos."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=str(WLASL_JSON),
        help="Path to WLASL_v0.3.json.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=str(VIDEO_DIR),
        help="Directory containing WLASL videos.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(KEYPOINT_DIR),
        help="Output directory for keypoint .npy files.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(METADATA_PATH),
        help="Output CSV file for metadata.",
    )
    parser.add_argument(
        "--single-hand",
        action="store_true",
        help="Use only a single hand (63-dim) instead of both hands (126-dim).",
    )

    args = parser.parse_args()

    json_path = Path(args.json)
    videos_dir = Path(args.videos)
    out_dir = Path(args.out)
    metadata_path = Path(args.metadata)

    if not json_path.exists():
        raise FileNotFoundError(f"WLASL JSON not found at {json_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found at {videos_dir}")

    # Update global dirs in case user overrides via CLI
    global VIDEO_DIR
    VIDEO_DIR = videos_dir

    extract_all_keypoints(
        json_path=json_path,
        output_dir=out_dir,
        metadata_path=metadata_path,
        subset=None,
        expect_both_hands=not args.single_hand,
    )


if __name__ == "__main__":
    main()

