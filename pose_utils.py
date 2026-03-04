from typing import List, Optional, Sequence, Tuple
import time  # Add this line

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """
    Calculate the angle ABC (in degrees) with B as the vertex.
    Points a, b, c are (x, y) in the same coordinate system.
    Returns angle in [0, 180].
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    # Normalize
    nba = ba / (np.linalg.norm(ba) + 1e-8)
    nbc = bc / (np.linalg.norm(bc) + 1e-8)

    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return float(angle)


def angle_accuracy(angle: float, target: float, tol: float) -> float:
    """
    Returns an accuracy score (0..100) for how close 'angle' is to 'target' within 'tol'.
    100 when |angle-target| <= tol, linearly decreasing to 0 at 2*tol, and clamped at 0.
    """
    diff = abs(angle - target)
    if diff <= tol:
        return 100.0
    if diff >= 2 * tol:
        return 0.0
    # Linear falloff between tol and 2*tol
    rem = (2 * tol - diff) / tol  # 1..0
    return max(0.0, min(100.0, rem * 100.0))


def get_pose_landmarks(results, image_shape) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[float]]]:
    """
    Converts normalized MediaPipe Pose landmarks to pixel coordinates.
    Returns (landmarks_px, visibility) where landmarks_px[i] = (x, y) in pixels.
    If landmarks are not present, returns (None, None).
    """
    if not results.pose_landmarks:
        return None, None

    h, w = image_shape[:2]
    lm = results.pose_landmarks.landmark
    points = []
    vis = []
    for p in lm:
        x = int(p.x * w)
        y = int(p.y * h)
        points.append((x, y))
        vis.append(p.visibility)
    return points, vis


def draw_pose(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def draw_highlights(frame, landmarks_px: List[Tuple[int, int]], indices: List[int], color=(0, 0, 255)):
    for idx in indices:
        if 0 <= idx < len(landmarks_px):
            x, y = landmarks_px[idx]
            cv2.circle(frame, (x, y), 10, color, thickness=3)


# Convenience alias for PoseLandmark enum
L = mp_pose.PoseLandmark