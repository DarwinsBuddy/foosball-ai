import logging

import cv2
import numpy as np
import pypeln as pl
from pypeln import BaseStage

from .analyze import Analyzer
from ..models import Mask, FrameDimensions, BallConfig, rgb2hsv, GoalConfig, Frame
from .preprocess import PreProcessor
from .render import Renderer
from .tracker import Tracker
from ..pipe import Pipeline


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
    return GoalConfig(bounds=[0, 235], invert_frame=True, invert_mask=True)


def get_ball_config(ball: str) -> BallConfig:
    if ball == 'orange' or ball == 'o':
        return orange_ball()
    elif ball == 'yellow' or ball == 'y':
        return yellow_ball()
    else:
        logging.error("Unknown ball color. Falling back to 'orange'")
        return orange_ball()


def generate_frame_mask(width, height) -> Mask:
    bar_color = 255
    bg = 0
    mask = np.full((height, width), bg, np.uint8)
    start = (int(width / 12), int(height / 20))
    end = (int(width / 1.2), int(height / 1.2))
    frame_mask = cv2.rectangle(mask, start, end, bar_color, -1)
    return frame_mask


class Tracking(Pipeline):

    def _stop(self):
        self.preprocessor.stop()
        self.tracker.stop()
        self.analyzer.stop()
        self.renderer.stop()
        self.frame_queue.stop()

    def __init__(self, dims: FrameDimensions, ball_config: BallConfig, goal_config: GoalConfig, headless=False, **kwargs):
        super().__init__()
        self.frame_queue = pl.process.IterableQueue()
        self.calibration = kwargs.get('calibration')
        self.dims = dims
        width, height = dims.scaled
        mask = generate_frame_mask(width, height)
        use_gpu = kwargs.get('gpu')
        self.preprocessor = PreProcessor(goal_config, mask=mask, headless=headless, useGPU=('preprocess' in use_gpu), **kwargs)
        self.tracker = Tracker(ball_config, useGPU=('tracker' in use_gpu), **kwargs)
        self.analyzer = Analyzer(**kwargs)
        self.renderer = Renderer(dims, headless=headless, useGPU=('render' in use_gpu), **kwargs)
        self.build()

    @property
    def output(self) -> pl.process.IterableQueue:
        return self.renderer.out

    @property
    def calibration_output(self) -> pl.process.IterableQueue | None:
        if self.calibration == "ball":
            return self.tracker.calibration_out
        elif self.calibration == "goal":
            return self.preprocessor.calibration_out
        return None

    def config_input(self, config) -> None:
        if self.calibration == "ball":
            self.tracker.config_input(config)
        elif self.calibration == "goal":
            self.preprocessor.config_input(config)

    def _build(self) -> BaseStage:
        return (
                self.frame_queue
                | pl.process.map(self.preprocessor.process)
                | pl.process.map(self.tracker.track)
                | pl.process.map(self.analyzer.analyze)
                | pl.process.map(self.renderer.render)
                # | pl.process.each(self.log)
                | pl.thread.filter(lambda x: False)
                | list
        )

    def track(self, frame: Frame) -> None:
        self.frame_queue.put(frame)
