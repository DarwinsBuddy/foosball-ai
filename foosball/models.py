import collections
from dataclasses import dataclass
from enum import Enum
from typing import Union

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


Goal = Blob


@dataclass
class Goals:
    left: Goal
    right: Goal


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


def filter_info(infos: [Info], infoVerbosity: Verbosity = Verbosity.TRACE) -> [Info]:
    return [i for i in infos if infoVerbosity is not None and infoVerbosity.value <= i.verbosity.value]

def infos_to_string(infos: [Info]):
    return " - ".join([i.to_string() for i in infos])


