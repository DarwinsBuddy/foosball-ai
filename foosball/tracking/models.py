import collections
from dataclasses import dataclass
from enum import Enum

import numpy as np

HSV = tuple[int, int, int]
RGB = tuple[int, int, int]

Frame = np.array
Mask = np.array

class ScaleDirection(Enum):
    UP = 1
    DOWN = 2


@dataclass
class FrameDimensions:
    original: [int, int]
    scaled: [int, int]
    scale: float


@dataclass
class Blob:
    center: [int, int]
    bbox: [int, int, int, int]


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
    rendered: Frame
    ball_track: Track
    ball: Blob
    info: list[tuple]
