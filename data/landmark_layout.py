from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class LandmarkSliceSpec:
    pose_slice: slice
    face_slice: slice
    left_hand_slice: slice
    right_hand_slice: slice


# MediaPipe Holistic common flattened layouts.
# - 543 = 33 pose + 468 face + 21 left hand + 21 right hand
# - 553 = 33 pose + 478 face(refined landmarks with iris) + 21 + 21
HOLISTIC_543_POSE_FACE_LH_RH = LandmarkSliceSpec(
    pose_slice=slice(0, 33),
    face_slice=slice(33, 501),
    left_hand_slice=slice(501, 522),
    right_hand_slice=slice(522, 543),
)

HOLISTIC_553_POSE_FACE_LH_RH = LandmarkSliceSpec(
    pose_slice=slice(0, 33),
    face_slice=slice(33, 511),
    left_hand_slice=slice(511, 532),
    right_hand_slice=slice(532, 553),
)


def infer_holistic_spec(num_points: int) -> LandmarkSliceSpec:
    if num_points == 553:
        return HOLISTIC_553_POSE_FACE_LH_RH
    if num_points == 543:
        return HOLISTIC_543_POSE_FACE_LH_RH
    raise ValueError(
        f"Unsupported landmark count {num_points}. "
        "Expected 553 (pose+face478+hands) or 543 (pose+face468+hands)."
    )


def split_holistic_landmarks(landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split holistic landmarks into streams expected by preprocess_sample.

    Args:
        landmarks: (T, V, 3), where V is 553 or 543

    Returns:
        dict with raw stream keys:
          - pose: (T, 33, 3)
          - face: (T, 478, 3) or (T, 468, 3)
          - left_hand: (T, 21, 3)
          - right_hand: (T, 21, 3)
    """
    if landmarks.ndim != 3 or landmarks.shape[-1] != 3:
        raise ValueError(f"Expected (T, V, 3), got {landmarks.shape}.")

    spec = infer_holistic_spec(landmarks.shape[1])
    return {
        "pose": landmarks[:, spec.pose_slice, :].astype(np.float32),
        "face": landmarks[:, spec.face_slice, :].astype(np.float32),
        "left_hand": landmarks[:, spec.left_hand_slice, :].astype(np.float32),
        "right_hand": landmarks[:, spec.right_hand_slice, :].astype(np.float32),
    }
