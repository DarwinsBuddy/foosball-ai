import collections
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

HSV = tuple[int, int, int]
RGB = tuple[int, int, int]

Frame = np.array
Mask = np.array


class ScaleDirection(Enum):
    UP = 1
    DOWN = 2


Point = [int, int]
Rect = (Point, Point, Point, Point)

@dataclass
class FrameDimensions:
    original: [int, int]
    scaled: [int, int]
    scale: float


@dataclass
class Blob:
    center: Point
    bbox: [int, int, int, int]  # x y width height


@dataclass
class DetectionResult:
    blob: Blob
    frame: np.array


@dataclass
class Bounds:
    ball: [HSV, HSV]


Track = collections.deque
Info = list[tuple[str, str]]


@dataclass
class TrackResult:
    frame: Frame
    ball_track: Track
    ball: Blob
    info: Info


@dataclass
class PreprocessResult:
    original: Frame
    preprocessed: Optional[Frame]
    homography_matrix: Optional[np.ndarray]  # 3x3 matrix used to warp the image and project points
    info: Info
