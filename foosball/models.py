import collections
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import cv2
import numpy as np

HSV = np.ndarray  # list[int, int, int]
RGB = np.ndarray  # list[int, int, int]


def rgb2hsv(rgb: RGB) -> HSV:
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]


def hsv2rgb(hsv: HSV) -> RGB:
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]


CPUFrame = np.array
GPUFrame = cv2.UMat
Frame = Union[CPUFrame, GPUFrame]

Mask = np.array


class ScaleDirection(Enum):
    UP = 1
    DOWN = 2


class Team(Enum):
    RED = 'RED'
    BLUE = 'BLUE'


Point = [int, int]
Rect = (Point, Point, Point, Point)
BBox = [int, int, int, int]  # x y width height


@dataclass
class Score:
    blue: int = 0
    red: int = 0

    def reset(self):
        self.blue = 0
        self.red = 0

    def inc(self, team: Team):
        if team == Team.BLUE:
            self.blue += 1
        elif team == Team.RED:
            self.red += 1

    def to_string(self):
        return f"{self.blue} : {self.red}"


@dataclass
class FrameDimensions:
    original: [int, int]
    scaled: [int, int]
    scale: float


@dataclass
class Blob:
    center: Point
    bbox: BBox

    def area(self):
        [_, _, w, h] = self.bbox
        return w * h


@dataclass
class DetectedBall:
    ball: Optional[Blob]
    frame: np.array


Goal = Blob


@dataclass
class Goals:
    left: Goal
    right: Goal


@dataclass
class DetectedGoals:
    goals: Optional[Goals]
    frame: np.array


Track = collections.deque


class Verbosity(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2


@dataclass
class Info:
    verbosity: Verbosity
    title: str
    value: str

    def to_string(self) -> str:
        return f'{self.title} {self.value}'


class InfoLog:
    def __init__(self, infos=None):
        self.infos: [Info] = [] if infos is None else infos

    def __iter__(self):
        return (i for i in self.infos)

    def append(self, info: Info):
        self.infos.append(info)

    def extend(self, info_log):
        self.infos.extend(info_log.infos)

    def filter(self, infoVerbosity: Verbosity = Verbosity.TRACE) -> [Info]:
        return [i for i in self.infos if infoVerbosity is not None and infoVerbosity.value <= i.verbosity.value]

    def to_string(self):
        return " - ".join([i.to_string() for i in self.infos])


@dataclass
class TrackerResult:
    frame: Frame
    goals: Goals | None
    ball_track: Track | None
    ball: Blob | None


@dataclass
class PreprocessorResult:
    original: Frame
    preprocessed: Optional[Frame]
    homography_matrix: Optional[np.ndarray]  # 3x3 matrix used to warp the image and project points
    goals: Optional[Goals]


@dataclass
class RendererResult:
    frame: Optional[Frame]
