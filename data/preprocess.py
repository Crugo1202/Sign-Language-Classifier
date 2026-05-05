from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Fixed semantic facial landmarks (non-random):
# eyebrows + eyes + nose + mouth contour.
# Works for MediaPipe face meshes with 468 or 478 landmarks.
FACE_KEY_INDICES = np.array(
    [
        # Eyebrows (20)
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
        # Eyes (24)
        33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155,
        362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382,
        # Nose (8)
        1, 2, 98, 327, 168, 6, 197, 195,
        # Mouth contour (20)
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    ],
    dtype=np.int64,
)


@dataclass
class PreprocessConfig:
    t_max: int = 64
    rotation_deg: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.01
    temporal_stretch: Tuple[float, float] = (0.8, 1.2)
    mirror_prob: float = 0.3
    align_hands_to_palm: bool = False


def _interpolate_missing_track(track: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Linear interpolation for (T, 3) joint tracks where missing_mask[t] is True."""
    track = track.copy()
    valid_idx = np.where(~missing_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(track)
    if len(valid_idx) == 1:
        track[:] = track[valid_idx[0]]
        return track

    all_idx = np.arange(track.shape[0])
    for c in range(track.shape[1]):
        valid_values = track[valid_idx, c]
        track[:, c] = np.interp(all_idx, valid_idx, valid_values)
    return track


def interpolate_missing_hand(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    hand_landmarks: (T, 21, 3)
    Missing frame criterion follows the plan's near-zero rule.
    """
    hand_landmarks = hand_landmarks.copy()
    missing_mask = np.sum(np.abs(hand_landmarks), axis=-1) < 1e-6  # (T, 21)
    for j in range(hand_landmarks.shape[1]):
        hand_landmarks[:, j, :] = _interpolate_missing_track(
            hand_landmarks[:, j, :], missing_mask[:, j]
        )
    return hand_landmarks


def _uniform_resample(sequence: np.ndarray, target_t: int) -> np.ndarray:
    if sequence.shape[0] == target_t:
        return sequence
    if sequence.shape[0] < target_t:
        pad_shape = (target_t - sequence.shape[0],) + sequence.shape[1:]
        return np.concatenate([sequence, np.zeros(pad_shape, dtype=sequence.dtype)], axis=0)

    idx = np.linspace(0, sequence.shape[0] - 1, target_t).round().astype(np.int64)
    return sequence[idx]


def _uniform_resample_with_valid_length(sequence: np.ndarray, target_t: int) -> Tuple[np.ndarray, int]:
    valid_len = min(sequence.shape[0], target_t)
    return _uniform_resample(sequence, target_t), int(valid_len)


def _rotation_y(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return points @ rot.T


def _temporal_stretch(sequence: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0:
        return sequence
    new_t = max(2, int(round(sequence.shape[0] * ratio)))
    old_idx = np.linspace(0, sequence.shape[0] - 1, sequence.shape[0])
    new_idx = np.linspace(0, sequence.shape[0] - 1, new_t)
    out = np.zeros((new_t,) + sequence.shape[1:], dtype=sequence.dtype)
    for i in range(np.prod(sequence.shape[1:-1], dtype=np.int64)):
        idx = np.unravel_index(i, sequence.shape[1:-1])
        sl = (slice(None),) + idx + (slice(None),)
        for c in range(sequence.shape[-1]):
            out[sl][..., c] = np.interp(new_idx, old_idx, sequence[sl][..., c])
    return out


def _safe_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def _palm_basis(hand_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wrist = hand_landmarks[:, 0, :]
    index_mcp = hand_landmarks[:, 5, :]
    middle_mcp = hand_landmarks[:, 9, :]
    pinky_mcp = hand_landmarks[:, 17, :]

    x_axis = _safe_normalize(index_mcp - pinky_mcp)
    palm_dir = middle_mcp - wrist
    z_axis = _safe_normalize(np.cross(x_axis, palm_dir))
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    # (T, 3, 3), rows are basis vectors in world coordinates.
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    return wrist, basis


def _align_hand_to_palm_frame(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Align hand landmarks to a per-frame palm local coordinate system.
    Uses landmarks: wrist(0), index_mcp(5), middle_mcp(9), pinky_mcp(17).
    """
    wrist, basis = _palm_basis(hand_landmarks)
    centered = hand_landmarks - wrist[:, None, :]
    aligned = np.einsum("tvc,tuc->tvu", centered, basis)
    return aligned.astype(np.float32)


def _rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to quaternions in (w, x, y, z) order.
    rot: (T, 3, 3)
    """
    q = np.empty((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    pos = trace > 0.0
    s = np.sqrt(np.maximum(trace[pos] + 1.0, 1e-6)) * 2.0
    q[pos, 0] = 0.25 * s
    q[pos, 1] = (rot[pos, 2, 1] - rot[pos, 1, 2]) / s
    q[pos, 2] = (rot[pos, 0, 2] - rot[pos, 2, 0]) / s
    q[pos, 3] = (rot[pos, 1, 0] - rot[pos, 0, 1]) / s

    rem = ~pos
    idx0 = rem & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + rot[idx0, 0, 0] - rot[idx0, 1, 1] - rot[idx0, 2, 2], 1e-6)) * 2.0
    q[idx0, 0] = (rot[idx0, 2, 1] - rot[idx0, 1, 2]) / s
    q[idx0, 1] = 0.25 * s
    q[idx0, 2] = (rot[idx0, 0, 1] + rot[idx0, 1, 0]) / s
    q[idx0, 3] = (rot[idx0, 0, 2] + rot[idx0, 2, 0]) / s

    idx1 = rem & ~idx0 & (rot[:, 1, 1] > rot[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + rot[idx1, 1, 1] - rot[idx1, 0, 0] - rot[idx1, 2, 2], 1e-6)) * 2.0
    q[idx1, 0] = (rot[idx1, 0, 2] - rot[idx1, 2, 0]) / s
    q[idx1, 1] = (rot[idx1, 0, 1] + rot[idx1, 1, 0]) / s
    q[idx1, 2] = 0.25 * s
    q[idx1, 3] = (rot[idx1, 1, 2] + rot[idx1, 2, 1]) / s

    idx2 = rem & ~idx0 & ~idx1
    s = np.sqrt(np.maximum(1.0 + rot[idx2, 2, 2] - rot[idx2, 0, 0] - rot[idx2, 1, 1], 1e-6)) * 2.0
    q[idx2, 0] = (rot[idx2, 1, 0] - rot[idx2, 0, 1]) / s
    q[idx2, 1] = (rot[idx2, 0, 2] + rot[idx2, 2, 0]) / s
    q[idx2, 2] = (rot[idx2, 1, 2] + rot[idx2, 2, 1]) / s
    q[idx2, 3] = 0.25 * s

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    q[q[:, 0] < 0.0] *= -1.0
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q.astype(np.float32)


def _hand_orientation_quaternion(hand_landmarks: np.ndarray) -> np.ndarray:
    _, basis = _palm_basis(hand_landmarks)
    return _rotation_matrix_to_quaternion(basis)


def apply_augmentation(sample: Dict[str, np.ndarray], cfg: PreprocessConfig) -> Dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in sample.items()}
    angle = np.deg2rad(np.random.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    scale = np.random.uniform(cfg.scale_range[0], cfg.scale_range[1])
    noise_std = cfg.noise_std

    for key in ("right_hand", "left_hand", "face", "pose"):
        if key in out and out[key].size > 0:
            out[key] = _rotation_y(out[key], angle) * scale
            out[key] += np.random.normal(0.0, noise_std, size=out[key].shape).astype(np.float32)

    stretch = np.random.uniform(cfg.temporal_stretch[0], cfg.temporal_stretch[1])
    for key in ("right_hand", "left_hand", "face", "pose"):
        if key in out and out[key].size > 0:
            out[key] = _temporal_stretch(out[key], stretch)

    if np.random.rand() < cfg.mirror_prob:
        if "right_hand" in out and "left_hand" in out:
            out["right_hand"], out["left_hand"] = out["left_hand"], out["right_hand"]
        for key in ("right_hand", "left_hand", "face", "pose"):
            if key in out and out[key].size > 0:
                out[key][..., 0] *= -1.0

    return out


def normalize_streams(
    right_hand: np.ndarray,
    left_hand: np.ndarray,
    face_468: np.ndarray,
    pose_33: np.ndarray,
    align_hands_to_palm: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Inputs:
      right_hand: (T, 21, 3)
      left_hand: (T, 21, 3)
      face_468: (T, 468/478, 3) or already selected keypoints (T, K, 3)
      pose_33: (T, 33, 3), uses nose/eyes for face-centric normalization
    """
    right_hand = interpolate_missing_hand(right_hand)
    left_hand = interpolate_missing_hand(left_hand)

    if face_468.shape[1] >= 468:
        face = face_468[:, FACE_KEY_INDICES, :]
    else:
        face = face_468

    face_center = pose_33[:, 0, :]  # nose
    left_eye = pose_33[:, 2, :]
    right_eye = pose_33[:, 5, :]
    face_scale = np.linalg.norm(left_eye - right_eye, axis=-1, keepdims=True) + 1e-6

    right_wrist = right_hand[:, 0, :]
    left_wrist = left_hand[:, 0, :]

    right_orientation = _hand_orientation_quaternion(right_hand)
    left_orientation = _hand_orientation_quaternion(left_hand)

    right_hand_shape = right_hand - right_wrist[:, None, :]
    left_hand_shape = left_hand - left_wrist[:, None, :]
    if align_hands_to_palm:
        right_hand_shape = _align_hand_to_palm_frame(right_hand)
        left_hand_shape = _align_hand_to_palm_frame(left_hand)

    right_wrist_traj = (right_wrist - face_center) / face_scale
    left_wrist_traj = (left_wrist - face_center) / face_scale
    dual_wrist_traj = np.concatenate([right_wrist_traj, left_wrist_traj], axis=-1)

    face_landmarks_norm = (face - face_center[:, None, :]) / face_scale[:, None, :]

    return {
        "right_hand_shape": right_hand_shape.astype(np.float32),
        "left_hand_shape": left_hand_shape.astype(np.float32),
        "dual_wrist_traj": dual_wrist_traj.astype(np.float32),
        "face_landmarks_norm": face_landmarks_norm.astype(np.float32),
        "hand_orientation": np.concatenate([right_orientation, left_orientation], axis=-1).astype(
            np.float32
        ),
    }


def preprocess_sample(
    sample: Dict[str, np.ndarray],
    cfg: PreprocessConfig,
    training: bool = False,
) -> Dict[str, np.ndarray]:
    """
    sample keys: right_hand, left_hand, face, pose
    shapes:
      right_hand/left_hand: (T, 21, 3)
      face: (T, 468, 3) or (T, 60, 3)
      pose: (T, 33, 3)
    """
    if training:
        sample = apply_augmentation(sample, cfg)

    streams = normalize_streams(
        right_hand=sample["right_hand"],
        left_hand=sample["left_hand"],
        face_468=sample["face"],
        pose_33=sample["pose"],
        align_hands_to_palm=cfg.align_hands_to_palm,
    )
    streams["right_hand_shape"], valid_len = _uniform_resample_with_valid_length(
        streams["right_hand_shape"], cfg.t_max
    )
    streams["left_hand_shape"] = _uniform_resample(streams["left_hand_shape"], cfg.t_max)
    streams["dual_wrist_traj"] = _uniform_resample(streams["dual_wrist_traj"], cfg.t_max)
    streams["face_landmarks_norm"] = _uniform_resample(streams["face_landmarks_norm"], cfg.t_max)
    streams["hand_orientation"] = _uniform_resample(streams["hand_orientation"], cfg.t_max)
    valid_mask = np.zeros((cfg.t_max,), dtype=np.bool_)
    valid_mask[:valid_len] = True
    streams["valid_mask"] = valid_mask
    streams["valid_length"] = np.int64(valid_len)
    return streams
