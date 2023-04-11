import collections
from dataclasses import dataclass
import numpy as np

from ..utils import HSV

Frame = np.array
Mask = np.array


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
    frame: Frame
    rendered: Frame
    ball_track: Track
    ball: Blob
    info: list[tuple]
