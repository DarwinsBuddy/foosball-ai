import collections
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import cv2
import numpy as np
import yaml

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
class BallDetectionResult:
    ball: Blob
    frame: np.array


Goal = Blob


@dataclass
class Goals:
    left: Goal
    right: Goal


@dataclass
class GoalsDetectionResult:
    goals: Optional[Goals]
    frame: np.array


@dataclass
class BallConfig:
    bounds_hsv: [HSV, HSV]
    invert_frame: bool = False
    invert_mask: bool = False

    def bounds(self, mode="hsv"):
        if mode == "hsv":
            return self.bounds_hsv
        else:
            return [hsv2rgb(x) for x in self.bounds_hsv]

    def store(self):
        filename = f"ball.yaml"
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def load(filename='ball.yaml'):
        if os.path.isfile(filename):
            logging.info("Loading ball config ball.yaml")
            with open(filename, 'r') as f:
                c = yaml.safe_load(f)
                return BallConfig(invert_frame=c['invert_frame'], invert_mask=c['invert_mask'], bounds_hsv=np.array(c['bounds_hsv']))
        else:
            logging.info("No ball config found")
        return None

    def to_dict(self):
        return {
            "bounds_hsv": [x.tolist() for x in self.bounds_hsv],
            "invert_frame": self.invert_frame,
            "invert_mask": self.invert_mask
        }


@dataclass
class GoalConfig:
    bounds: [int, int]
    invert_frame: bool = True
    invert_mask: bool = True

    def store(self):
        filename = f"goal.yaml"
        [lower, upper] = self.bounds
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def load(filename='goal.yaml'):
        with open(filename, 'r') as f:
            c = yaml.safe_load(f)
            return GoalConfig(**c)

    def to_dict(self):
        return {
            "bounds": self.bounds,
            "invert_frame": self.invert_frame,
            "invert_mask": self.invert_mask
        }

Track = collections.deque
Info = list[tuple[str, str]]


@dataclass
class TrackResult:
    frame: CPUFrame
    goals: Optional[Goals]
    ball_track: Track
    ball: Blob
    info: Info


@dataclass
class AnalyzeResult:
    frame: CPUFrame
    score: Score
    goals: Optional[Goals]
    ball_track: Track
    ball: Blob
    info: Info


@dataclass
class PreprocessResult:
    original: CPUFrame
    preprocessed: Optional[CPUFrame]
    homography_matrix: Optional[np.ndarray]  # 3x3 matrix used to warp the image and project points
    goals: Optional[Goals]
    info: Info
