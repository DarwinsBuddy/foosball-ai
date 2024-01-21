from enum import Enum
from typing import Callable

import cv2
import numpy as np
import yaml

from . import Sink
from ..detectors.color import BallConfig, GoalConfig

GOAL = "goal"
BALL = "ball"


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


def slider_label(rgb, bound):
    return f"{rgb} ({bound})"


def add_config_input(calibrationMode, config):
    if calibrationMode == GOAL:
        add_goals_config_input(config)
    elif calibrationMode == BALL:
        add_ball_config_input(config)


def add_ball_config_input(bounds: BallConfig):
    [lower_hsv, upper_hsv] = bounds.bounds
    cv2.createTrackbar(f'invert_frame', BALL, 1 if bounds.invert_frame else 0, 1, lambda v: None)
    cv2.createTrackbar(f'invert_mask', BALL, 1 if bounds.invert_mask else 0, 1, lambda v: None)
    # create trackbars for color change
    cv2.createTrackbar('Hue', BALL, avg(lower_hsv[0], upper_hsv[0]), 179, lambda v: None)
    cv2.createTrackbar(slider_label('S', 'low'), BALL, lower_hsv[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('V', 'low'), BALL, lower_hsv[2], 255, lambda v: None)
    cv2.createTrackbar(slider_label('S', 'high'), BALL, upper_hsv[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('V', 'high'), BALL, upper_hsv[2], 255, lambda v: None)
    # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))


def add_goals_config_input(config: GoalConfig):
    [lower, upper] = config.bounds
    cv2.createTrackbar(f'invert_frame', GOAL, 1 if config.invert_frame else 0, 1, lambda v: None)
    cv2.createTrackbar(f'invert_mask', GOAL, 1 if config.invert_mask else 0, 1, lambda v: None)
    # create trackbars for color change
    cv2.createTrackbar("lower", GOAL, lower, 255, lambda v: None)
    cv2.createTrackbar("upper", GOAL, upper, 255, lambda v: None)
    # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))


def reset_config(calibrationMode, config):
    if calibrationMode == GOAL:
        reset_goal_config(config)
    elif calibrationMode == BALL:
        reset_ball_config(config)


def avg(x, y):
    return int((x + y) / 2)

def reset_ball_config(bounds: BallConfig):
    [lower_hsv, upper_hsv] = bounds.bounds
    print(f"Reset config {BALL}", end="\n\n\n")

    cv2.setTrackbarPos('invert_frame', BALL, 1 if bounds.invert_frame else 0)
    cv2.setTrackbarPos('invert_mask', BALL, 1 if bounds.invert_mask else 0)

    cv2.setTrackbarPos('Hue', BALL, avg(lower_hsv[0], upper_hsv[0]))
    cv2.setTrackbarPos(slider_label('S', 'low'), BALL, lower_hsv[1])
    cv2.setTrackbarPos(slider_label('V', 'low'), BALL, lower_hsv[2])
    cv2.setTrackbarPos(slider_label('S', 'high'), BALL, upper_hsv[1])
    cv2.setTrackbarPos(slider_label('V', 'high'), BALL, upper_hsv[2])


def reset_goal_config(config: GoalConfig):
    [lower, upper] = config.bounds
    print(f"Reset config {GOAL}", end="\n\n\n")

    cv2.setTrackbarPos('invert_frame', GOAL, 1 if config.invert_frame else 0)
    cv2.setTrackbarPos('invert_mask', GOAL, 1 if config.invert_mask else 0)

    cv2.setTrackbarPos('lower', GOAL, lower)
    cv2.setTrackbarPos('upper', GOAL, upper)


def store_ball_config(config: BallConfig):
    filename = f"ball.yaml"
    [lower, upper] = config.bounds
    print(f"Store config {filename}" + (" " * 50), end="\n\n")
    with open(filename, "w") as f:
        yaml.dump({
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "invert_frame": config.invert_frame,
            "invert_mask": config.invert_mask
        }, f)


def store_goals_config(config: GoalConfig):
    filename = f"goal.yaml"
    [lower, upper] = config.bounds
    print(f"Store config {filename}" + (" " * 50), end="\n\n")
    with open(filename, "w") as f:
        yaml.dump({
            "lower": lower,
            "upper": upper,
            "invert_frame": config.invert_frame,
            "invert_mask": config.invert_mask
        }, f)


def get_slider_config(calibrationMode):
    if calibrationMode == GOAL:
        return get_slider_goals_config()
    elif calibrationMode == BALL:
        return get_slider_ball_config()


def int2bool(x: int) -> bool:
    return True if x == 1 else False


def get_slider_ball_config():
    # get current positions of four trackbars
    invert_frame = cv2.getTrackbarPos('invert_frame', BALL)
    invert_mask = cv2.getTrackbarPos('invert_mask', BALL)

    hue = cv2.getTrackbarPos('Hue', BALL)
    hl = max(0, hue - 10)
    hh = min(179, hue + 10)

    sl = cv2.getTrackbarPos(slider_label('S', 'low'), BALL)
    sh = cv2.getTrackbarPos(slider_label('S', 'high'), BALL)

    vl = cv2.getTrackbarPos(slider_label('V', 'low'), BALL)
    vh = cv2.getTrackbarPos(slider_label('V', 'high'), BALL)
    lower = np.array([hl, sl, vl])
    upper = np.array([hh, sh, vh])
    return BallConfig(bounds=[lower, upper], invert_mask=int2bool(invert_mask), invert_frame=int2bool(invert_frame))


def get_slider_goals_config():
    # get current positions of four trackbars
    invert_frame = cv2.getTrackbarPos('invert_frame', GOAL)
    invert_mask = cv2.getTrackbarPos('invert_mask', GOAL)

    lower = cv2.getTrackbarPos('lower', GOAL)
    upper = cv2.getTrackbarPos('upper', GOAL)

    return GoalConfig(bounds=[lower, upper], invert_mask=int2bool(invert_mask), invert_frame=int2bool(invert_frame))
