import logging

import pypeln as pl
import cv2
import numpy as np
from pypeln import BaseStage

from .models import Mask, FrameDimensions, BallConfig, ScaleDirection, RGB, HSV, rgb2hsv, GoalConfig
from .preprocess import PreProcessor
from .render import Renderer
from ..pipe import Pipeline
from .tracker import Tracker


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
    # TODO: instead of doing this approx. calculations
    #       scale the whole stream down to a standardized size
    #       and fix the frame according to dewarped image's recognized boundaries
    #       don't forget to scale renderings accordingly (if original image is shown)
    start = (int(width / 12), int(height / 20))
    end = (int(width / 1.2), int(height / 1.2))
    frame_mask = cv2.rectangle(mask, start, end, bar_color, -1)
    return frame_mask


class Tracking(Pipeline):

    def _stop(self):
        self.frame_queue.stop()
        self.tracker.stop()
        self.renderer.stop()

    def __init__(self, dims: FrameDimensions, ball_config: BallConfig, goal_config: GoalConfig, verbose=False,
                 headless=False,
                 off=False, **kwargs):
        super().__init__()
        self.frame_queue = pl.process.IterableQueue()
        self.calibration = kwargs.get('calibration')
        self.dims = dims
        width, height = dims.scaled
        mask = generate_frame_mask(width, height)

        self.preprocessor = PreProcessor(goal_config, mask=mask, headless=headless, **kwargs)
        self.tracker = Tracker(ball_config, off=off, verbose=verbose, **kwargs)
        self.renderer = Renderer(dims, headless=headless, **kwargs)
        self.build()

    @property
    def output(self) -> pl.process.IterableQueue:
        return self.renderer.out

    @property
    def calibration_output(self) -> pl.process.IterableQueue:
        if self.calibration == "ball":
            return self.tracker.calibration_out
        elif self.calibration == "goal":
            return self.preprocessor.calibration_out

    def config_input(self, config) -> None:
        if self.calibration == "ball":
            return self.tracker.config_input(config)
        elif self.calibration == "goal":
            return self.preprocessor.config_input(config)

    def _build(self) -> BaseStage:
        return (
                self.frame_queue
                | pl.process.map(self.preprocessor.process)
                | pl.process.map(self.tracker.track)
                | pl.process.map(self.renderer.render)
                # | pl.process.each(self.log)
                | pl.thread.filter(lambda x: False)
                | list
        )

    def track(self, frame) -> None:
        self.frame_queue.put(frame)
