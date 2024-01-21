from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Callable, Generic, TypeVar

import cv2
import numpy as np
import yaml

from const import CalibrationMode
from . import Sink
from ..detectors.color import BallColorConfig, GoalColorConfig
from ..utils import int2bool, avg


class Key(Enum):
    UP = 2490368
    DOWN = 2621440
    LEFT = 2424832
    RIGHT = 2555904
    SPACE = 32
    DELETE = 3014656
    ESC = 27


class DisplaySink(Sink):

    def __init__(self, name='frame', pos='tl', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        cv2.namedWindow(self.name)
        [x, y] = self._position(pos)
        cv2.moveWindow(self.name, x, y)

    @staticmethod
    def _position(pos):
        return {
            'tl': [10, 0],
            'tr': [1310, 0],
            'bl': [10, 900],
            'br': [1310, 900]
        }[pos]

    @staticmethod
    def title(s):
        print(f'{"=" * 8} {s} {"=" * 8}')

    def stop(self):
        cv2.destroyWindow(self.name)

    def show(self, frame):
        if frame is not None:
            cv2.imshow(self.name, frame)

    def render(self, callbacks: dict[int, Callable] = None):
        return wait(loop=False, interval=1, callbacks=callbacks)


def wait(loop=False, interval=1, callbacks=None):
    if callbacks is None:
        callbacks = {ord('q'): lambda: True}
    while True:
        key = cv2.waitKey(interval) & 0xFF
        # if the expected key is pressed, return
        if key in callbacks:
            return callbacks[key]()
        if not loop:
            break
    return False


def slider_label(name, bound):
    return f"{name} ({bound})"


CalibrationConfig = TypeVar('CalibrationConfig')


class Calibration(ABC, Generic[CalibrationConfig]):
    def __init__(self, config: CalibrationConfig, *args, **kwargs):
        self.config: CalibrationConfig = config
        self.init_config: CalibrationConfig = deepcopy(config)

    def reset(self):
        print(f"Reset calibration config", end="\n\n\n")
        self.set_slider_config(self.init_config)

    @abstractmethod
    def set_slider_config(self, config: CalibrationConfig):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def get_slider_config(self) -> CalibrationConfig:
        pass


class GoalColorCalibration(Calibration[GoalColorConfig]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        [lower, upper] = self.init_config.bounds
        cv2.createTrackbar('invert_frame', CalibrationMode.GOAL, 1 if self.init_config.invert_frame else 0, 1, lambda v: None)
        cv2.createTrackbar('invert_mask', CalibrationMode.GOAL, 1 if self.init_config.invert_mask else 0, 1, lambda v: None)
        # create trackbars for color change
        cv2.createTrackbar("lower", CalibrationMode.GOAL, lower, 255, lambda v: None)
        cv2.createTrackbar("upper", CalibrationMode.GOAL, upper, 255, lambda v: None)
        # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))

    def set_slider_config(self, config: GoalColorConfig):
        [lower, upper] = config.bounds
        print(f"Reset config {CalibrationMode.GOAL}", end="\n\n\n")

        cv2.setTrackbarPos('invert_frame', CalibrationMode.GOAL, 1 if config.invert_frame else 0)
        cv2.setTrackbarPos('invert_mask', CalibrationMode.GOAL, 1 if config.invert_mask else 0)

        cv2.setTrackbarPos('lower', CalibrationMode.GOAL, lower)
        cv2.setTrackbarPos('upper', CalibrationMode.GOAL, upper)

    def get_slider_config(self) -> GoalColorConfig:
        # get current positions of four trackbars
        invert_frame = cv2.getTrackbarPos('invert_frame', CalibrationMode.GOAL)
        invert_mask = cv2.getTrackbarPos('invert_mask', CalibrationMode.GOAL)

        lower = cv2.getTrackbarPos('lower', CalibrationMode.GOAL)
        upper = cv2.getTrackbarPos('upper', CalibrationMode.GOAL)

        return GoalColorConfig(bounds=[lower, upper], invert_mask=int2bool(invert_mask), invert_frame=int2bool(invert_frame))

    def store(self):
        filename = "goal.yaml"
        c = self.get_slider_config()
        [lower, upper] = c
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump({
                "lower": lower,
                "upper": upper,
                "invert_frame": c.invert_frame,
                "invert_mask": c.invert_mask
            }, f)


class BallColorCalibration(Calibration[BallColorConfig]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        [lower_hsv, upper_hsv] = self.init_config.bounds
        cv2.createTrackbar('invert_frame', CalibrationMode.BALL, 1 if self.init_config.invert_frame else 0, 1, lambda v: None)
        cv2.createTrackbar('invert_mask', CalibrationMode.BALL, 1 if self.init_config.invert_mask else 0, 1, lambda v: None)
        # create trackbars for color change
        cv2.createTrackbar('Hue', CalibrationMode.BALL, avg(lower_hsv[0], upper_hsv[0]), 179, lambda v: None)
        cv2.createTrackbar(slider_label('S', 'low'), CalibrationMode.BALL, lower_hsv[1], 255, lambda v: None)
        cv2.createTrackbar(slider_label('V', 'low'), CalibrationMode.BALL, lower_hsv[2], 255, lambda v: None)
        cv2.createTrackbar(slider_label('S', 'high'), CalibrationMode.BALL, upper_hsv[1], 255, lambda v: None)
        cv2.createTrackbar(slider_label('V', 'high'), CalibrationMode.BALL, upper_hsv[2], 255, lambda v: None)
        # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))

    def set_slider_config(self, config: BallColorConfig):
        [lower_hsv, upper_hsv] = config.bounds
        print(f"Reset config {CalibrationMode.BALL}", end="\n\n\n")

        cv2.setTrackbarPos('invert_frame', CalibrationMode.BALL, 1 if config.invert_frame else 0)
        cv2.setTrackbarPos('invert_mask', CalibrationMode.BALL, 1 if config.invert_mask else 0)

        cv2.setTrackbarPos('Hue', CalibrationMode.BALL, avg(lower_hsv[0], upper_hsv[0]))
        cv2.setTrackbarPos(slider_label('S', 'low'), CalibrationMode.BALL, lower_hsv[1])
        cv2.setTrackbarPos(slider_label('V', 'low'), CalibrationMode.BALL, lower_hsv[2])
        cv2.setTrackbarPos(slider_label('S', 'high'), CalibrationMode.BALL, upper_hsv[1])
        cv2.setTrackbarPos(slider_label('V', 'high'), CalibrationMode.BALL, upper_hsv[2])

    def get_slider_config(self) -> BallColorConfig:
        # get current positions of four trackbars
        invert_frame = cv2.getTrackbarPos('invert_frame', CalibrationMode.BALL)
        invert_mask = cv2.getTrackbarPos('invert_mask', CalibrationMode.BALL)

        hue = cv2.getTrackbarPos('Hue', CalibrationMode.BALL)
        hl = max(0, hue - 10)
        hh = min(179, hue + 10)

        sl = cv2.getTrackbarPos(slider_label('S', 'low'), CalibrationMode.BALL)
        sh = cv2.getTrackbarPos(slider_label('S', 'high'), CalibrationMode.BALL)

        vl = cv2.getTrackbarPos(slider_label('V', 'low'), CalibrationMode.BALL)
        vh = cv2.getTrackbarPos(slider_label('V', 'high'), CalibrationMode.BALL)
        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])
        return BallColorConfig(bounds=[lower, upper], invert_mask=int2bool(invert_mask), invert_frame=int2bool(invert_frame))

    def store(self):
        filename = "ball.yaml"
        c = self.get_slider_config()
        [lower, upper] = c.bounds
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump({
                "bounds": [lower.tolist(), upper.tolist()],
                "invert_frame": c.invert_frame,
                "invert_mask": c.invert_mask
            }, f)
