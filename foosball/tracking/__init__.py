import logging
import os.path
from multiprocessing import Queue

import cv2
import numpy as np

from const import GPU, CalibrationMode, BallPresets
from .analyze import Analyzer
from .preprocess import PreProcessor
from .render import Renderer
from .tracker import Tracker
from ..models import Mask, FrameDimensions, BallConfig, rgb2hsv, GoalConfig
from ..pipe.Pipe import Pipe


def dim(frame) -> [int, int]:
    return [frame.shape[1], frame.shape[0]]


def yellow_ball() -> BallConfig:
    lower = rgb2hsv(np.array([140, 86, 73]))
    upper = rgb2hsv(np.array([0, 255, 94]))

    return BallConfig(bounds=[lower, upper], invert_frame=False, invert_mask=False)


def orange_ball() -> BallConfig:
    lower = rgb2hsv(np.array([166, 94, 72]))
    upper = rgb2hsv(np.array([0, 249, 199]))

    return BallConfig(bounds=[lower, upper], invert_frame=False, invert_mask=False)


def get_goal_config() -> GoalConfig:
    default_config = GoalConfig(bounds=[0, 235], invert_frame=True, invert_mask=True)
    if os.path.isfile('goal.yaml'):
        return GoalConfig.load() or default_config
    return default_config


def get_ball_config(ball: str) -> BallConfig:
    match ball:
        case BallPresets.YAML:
            return BallConfig.load() or yellow_ball()
        case BallPresets.ORANGE:
            return orange_ball()
        case BallPresets.YELLOW:
            return yellow_ball()
        case _:
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

    def __init__(self, stream, dims: FrameDimensions, ball_config: BallConfig, goal_config: GoalConfig, headless=False, maxPipeSize=128, calibrationMode=None, **kwargs):
        super().__init__()
        self.calibrationMode = calibrationMode

        width, height = dims.scaled
        mask = generate_frame_mask(width, height)
        gpu_flags = kwargs.get(GPU)
        self.preprocessor = PreProcessor(dims, goal_config, mask=mask, headless=headless, useGPU='preprocess' in gpu_flags,
                                         calibrationMode=calibrationMode, **kwargs)
        self.tracker = Tracker(ball_config, useGPU='tracker' in gpu_flags, calibrationMode=calibrationMode, **kwargs)
        self.analyzer = Analyzer(**kwargs)
        self.renderer = Renderer(dims, headless=headless, useGPU='render' in gpu_flags, **kwargs)

        self.stream = stream
        self.pipe = Pipe(stream, [self.preprocessor, self.tracker, self.analyzer, self.renderer], maxsize=maxPipeSize)
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

    def status(self) -> str:
        return self.pipe.status()

    @property
    def calibration_output(self) -> Queue:
        match self.calibrationMode:
            case CalibrationMode.BALL:
                return self.tracker.calibration_out
            case CalibrationMode.GOAL:
                return self.preprocessor.calibration_out

    def config_input(self, config) -> None:
        match self.calibrationMode:
            case CalibrationMode.BALL:
                return self.tracker.config_input(config)
            case CalibrationMode.GOAL:
                return self.preprocessor.config_input(config)

    def reset_score(self):
        self.analyzer.reset_score()
