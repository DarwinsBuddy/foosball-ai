
import pypeln as pl
import cv2
import numpy as np
from pypeln import BaseStage

from .models import Mask, FrameDimensions, Bounds
from .render import Renderer
from ..pipe import Pipeline
from .tracker import Tracker, get_ball_bounds_hsv

def dim(frame) -> [int, int]:
    return [frame.shape[1], frame.shape[0]]

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
        self.tracker.stop()
        self.frame_queue.stop()
        self.renderer.stop()

    def __init__(self, dims: FrameDimensions, calibration=False, verbose=False, track_buffer=64, headless=False, off=False, **kwargs):
        super().__init__()
        self.frame_queue        = pl.process.IterableQueue()

        self.dims = dims
        width, height = dims.scaled
        mask = generate_frame_mask(width, height)

        self.tracker = Tracker(mask, ball_bounds_hsv=get_ball_bounds_hsv(), off=off, track_buffer=track_buffer, verbose=verbose, calibration=calibration, **kwargs)
        self.renderer = Renderer(dims, headless=headless, **kwargs)

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
            | pl.process.map(self.tracker.track)
            | pl.process.map(self.renderer.render)
            # | pl.process.each(self.log)
            | pl.thread.filter(lambda x: False)
            | list
        )

    def track(self, frame) -> None:
        self.frame_queue.put(frame)
