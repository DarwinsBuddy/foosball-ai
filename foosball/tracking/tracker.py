from collections import deque

import cv2

from . import DetectionResult
from ..tracker import ColorDetection
from ..utils import rgb2hsv


def get_ball_bounds_hsv():
    # TODO: #2 calibration for the demo footage (other ball => other values)
    lower = rgb2hsv((166, 94, 72))
    upper = rgb2hsv((0, 249, 199))

    return [lower, upper]

def get_goal_bounds_hsv():
    lower = rgb2hsv((0, 0, 0))
    upper = rgb2hsv((0, 0, 8))

    return [lower, upper]

class Tracker:

    def __init__(self, mask, off=False, track_buffer=64, verbose=False, ball_calibration=False, **kwargs):
        self.kwargs = kwargs
        self.mask = mask
        self.ball_track = deque(maxlen=track_buffer)
        self.off = off
        self.verbose = verbose
        self.ball_calibration = ball_calibration

        self.ball_bounds_hsv = get_ball_bounds_hsv()
        [self.init_lower_bounds_hsv, self.init_upper_bounds_hsv] = self.ball_bounds_hsv
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.ball_detection = ColorDetection('ball', self.ball_bounds_hsv, self.ball_calibration, self.verbose)

    def reset(self):
        self.ball_detection.reset_bounds(self.init_lower_bounds_hsv, self.init_upper_bounds_hsv)

    def update_ball_track(self, detected_ball):
        if detected_ball is not None:
            [center, bbox] = detected_ball
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def mask_frame(self, image):
        return cv2.bitwise_and(image, image, mask=self.mask)

    def get_info(self, ball_track):
        info = [
            ("Track length", f"{sum([1 for p in ball_track if p is not None])}"),
            ("Calibration", f"{'on' if self.ball_calibration else 'off'}"),
            ("Tracker", f"{'off' if self.off else 'on'}")
        ]
        if self.ball_calibration:
            [lower_rgb, upper_rgb] = self.ball_detection.get_bounds("rgb")
            info.append(("Ball Lower RGB", f'{lower_rgb}'))
            info.append(("Ball Upper RGB", f'{upper_rgb}'))
        return info

    def track(self, frame):
        masked = None
        ball = None
        ball_track = None
        if not self.off:
            masked = self.mask_frame(frame)
            ball = self.ball_detection.detect(masked)
            ball_track = self.update_ball_track(ball)
        info = self.get_info(ball_track)
        return DetectionResult(frame, masked, ball_track, ball, info)