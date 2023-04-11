import logging
from queue import Empty

import cv2
import pypeln as pl

from .colordetection import get_bounds, detect
from .models import TrackResult, Mask, Track, Bounds, Frame, Info, Blob
from ..utils import rgb2hsv, HSV, RGB

def log(result: TrackResult) -> None:
    logging.debug(result.info)

def get_ball_bounds_hsv() -> [RGB, RGB]:
    # TODO: #2 calibration for the demo footage (other ball => other values)
    lower = rgb2hsv((166, 94, 72))
    upper = rgb2hsv((0, 249, 199))

    return [lower, upper]

def get_goal_bounds_hsv() -> [HSV, HSV]:
    lower = rgb2hsv((0, 0, 0))
    upper = rgb2hsv((0, 0, 8))

    return [lower, upper]

class Tracker:

    def __init__(self, mask: Mask, ball_bounds_hsv: [HSV, HSV], off=False, track_buffer=64, verbose=False, calibration=False, **kwargs):
        self.kwargs = kwargs
        self.mask = mask
        self.ball_track = Track(maxlen=track_buffer)
        self.off = off
        self.verbose = verbose
        self.calibration = calibration
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.bounds = Bounds(ball=ball_bounds_hsv)

        self.bounds_in = pl.process.IterableQueue() if calibration else None
        self.calibration_out = pl.process.IterableQueue() if calibration else None

    def stop(self) -> None:
        if self.calibration:
            self.bounds_in.stop()
            self.calibration_out.stop()

    def update_ball_track(self, detected_ball: Blob) -> Track:
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def mask_frame(self, frame: Frame) -> Frame:
        return cv2.bitwise_and(frame, frame, mask=self.mask)

    def get_info(self, ball_track: Track) -> Info:
        info = [
            ("Track length", f"{sum([1 for p in ball_track if p is not None])}"),
            ("Calibration", f"{'on' if self.calibration else 'off'}"),
            ("Tracker", f"{'off' if self.off else 'on'}")
        ]
        if self.calibration:
            [lower_rgb, upper_rgb] = get_bounds(self.bounds.ball, "rgb")
            info.append(("Ball Lower RGB", f'{lower_rgb}'))
            info.append(("Ball Upper RGB", f'{upper_rgb}'))
        return info
    @property
    def calibration_output(self) -> pl.process.IterableQueue:
        return self.calibration_out
    def bounds_input(self, bounds: Bounds) -> None:
        if self.calibration:
            self.bounds_in.put_nowait(bounds)

    def track(self, frame) -> TrackResult:
        masked = None
        ball = None
        ball_track = None
        if not self.off:
            masked = self.mask_frame(frame)
            if self.calibration:
                try:
                    self.bounds = self.bounds_in.get_nowait()
                except Empty:
                    pass
            detection_result = detect(masked, self.bounds.ball)
            ball = detection_result.blob
            if self.calibration:
                    self.calibration_out.put_nowait(detection_result.frame)
            ball_track = self.update_ball_track(ball)
        info = self.get_info(ball_track)
        return TrackResult(frame, masked, ball_track, ball, info)