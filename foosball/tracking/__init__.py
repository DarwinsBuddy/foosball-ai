import logging

import pypeln as pl
import cv2
import numpy as np
from pypeln import BaseStage

from .models import Mask, FrameDimensions, Bounds, ScaleDirection, RGB, HSV
from .preprocess import PreProcessor
from .render import Renderer
from .utils import rgb2hsv
from ..pipe import Pipeline
from .tracker import Tracker


def dim(frame) -> [int, int]:
    return [frame.shape[1], frame.shape[0]]


def yellow_ball() -> [RGB, RGB]:
    lower = rgb2hsv((117, 106, 86))
    upper = rgb2hsv((117, 255, 92))

    return [lower, upper]


def orange_ball() -> [RGB, RGB]:
    lower = rgb2hsv((166, 94, 72))
    upper = rgb2hsv((0, 249, 199))

    return [lower, upper]


def get_ball_bounds(ball: str) -> [RGB, RGB]:
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

    def __init__(self, dims: FrameDimensions, ball_bounds_hsv: [HSV, HSV], calibration=False, verbose=False, headless=False,
                 off=False, **kwargs):
        super().__init__()
        self.frame_queue = pl.process.IterableQueue()

        self.dims = dims
        width, height = dims.scaled
        mask = generate_frame_mask(width, height)

        self.preprocessor = PreProcessor(mask=mask, headless=headless)
        self.tracker = Tracker(ball_bounds_hsv, off=off, verbose=verbose, calibration=calibration, **kwargs)
        self.renderer = Renderer(dims, headless=headless, **kwargs)
        self.build()

    @property
    def output(self) -> pl.process.IterableQueue:
        return self.renderer.out

    @property
    def calibration_output(self) -> pl.process.IterableQueue:
        return self.tracker.calibration_out

    def bounds_input(self, bounds: Bounds) -> None:
        return self.tracker.bounds_input(bounds)

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
