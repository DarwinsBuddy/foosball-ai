import logging
import os.path
from multiprocessing import Queue

import cv2
import numpy as np

from .analyze import Analyzer
from .preprocess import PreProcessor
from .render import Renderer
from .tracker import Tracker
from ..models import Mask, FrameDimensions, BallConfig, rgb2hsv, GoalConfig, Frame
from ..pipe.BaseProcess import Msg
from ..pipe.Pipe import Pipe


def dim(frame) -> [int, int]:
    return [frame.shape[1], frame.shape[0]]


def yellow_ball() -> BallConfig:
    lower = rgb2hsv(np.array([140, 86, 73]))
    upper = rgb2hsv(np.array([0, 255, 94]))

    return BallConfig(bounds_hsv=[lower, upper], invert_frame=False, invert_mask=False)


def orange_ball() -> BallConfig:
    lower = rgb2hsv(np.array([166, 94, 72]))
    upper = rgb2hsv(np.array([0, 249, 199]))

    return BallConfig(bounds_hsv=[lower, upper], invert_frame=False, invert_mask=False)


def get_goal_config() -> GoalConfig:
    default_config = GoalConfig(bounds=[0, 235], invert_frame=True, invert_mask=True)
    if os.path.isfile('goal.yaml'):
        return GoalConfig.load() or default_config
    return default_config


def get_ball_config(ball: str) -> BallConfig:
    if ball == 'yaml':
        return BallConfig.load() or yellow_ball()
    elif ball == 'orange' or ball == 'o':
        return orange_ball()
    elif ball == 'yellow' or ball == 'y':
        return yellow_ball()
    else:
        logging.error("Unknown ball color. Falling back to 'yellow'")
        return yellow_ball()


def generate_frame_mask(width, height) -> Mask:
    bar_color = 255
    bg = 0
    mask = np.full((height, width), bg, np.uint8)
    start = (int(width / 12), int(height / 20))
    end = (int(width / 1.2), int(height / 1.2))
    frame_mask = cv2.rectangle(mask, start, end, bar_color, -1)
    return frame_mask


class Tracking:

    def __init__(self, stream, dims: FrameDimensions, ball_config: BallConfig, goal_config: GoalConfig, headless=False, **kwargs):
        super().__init__()
        self.calibration = kwargs.get('calibration')
        self.dims = dims
        width, height = dims.scaled
        mask = generate_frame_mask(width, height)
        self.preprocessor = PreProcessor(dims, goal_config, mask=mask, headless=headless, useGPU=kwargs.get('preprocess-gpu'),
                                         **kwargs)
        self.tracker = Tracker(ball_config, useGPU=kwargs.get('tracker-gpu'), **kwargs)
        self.analyzer = Analyzer(**kwargs)
        self.renderer = Renderer(dims, headless=headless, useGPU=kwargs.get('render-gpu'), **kwargs)

        self.stream = stream
        self.pipe = Pipe(stream, [self.preprocessor, self.tracker, self.analyzer, self.renderer])
        self.frame_queue = self.pipe.input

    def start(self):
        self.pipe.start()

    def stop(self):
        self.pipe.stop()

    def pause(self):
        self.pipe.pause()

    def resume(self):
        self.pipe.resume()

    def step(self):
        self.pipe.step()

    @property
    def output(self) -> Queue:
        return self.pipe.output

    @property
    def calibration_output(self) -> Queue:
        if self.calibration == "ball":
            return self.tracker.calibration_out
        elif self.calibration == "goal":
            return self.preprocessor.calibration_out

    def config_input(self, config) -> None:
        if self.calibration == "ball":
            return self.tracker.config_input(config)
        elif self.calibration == "goal":
            return self.preprocessor.config_input(config)
