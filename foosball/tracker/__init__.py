from collections import deque

import cv2
import numpy as np

from foosball.tracker.colordetection import ColorDetection
from foosball.utils import rgb2hsv


class Tracker:

    def __init__(self, frame_dimensions, display, ball_calibration=False, verbose=False, track_buffer=64):
        self.ball_track = deque(maxlen=track_buffer)
        self.display = display
        self.verbose = verbose
        self.ball_calibration = ball_calibration

        self.ball_bounds_hsv = self.get_ball_bounds_hsv()
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.ball_detection = ColorDetection('ball', display, self.ball_bounds_hsv, self.ball_calibration, self.verbose)

        width, height = frame_dimensions
        self.frame_mask = self.generate_frame_mask(width, height)

    def reset(self):
        self.ball_detection.reset_bounds()

    @staticmethod
    def get_ball_bounds_hsv():
        # TODO: #2 calibration for the demo footage (other ball => other values)
        lower = rgb2hsv((166, 94, 72))
        upper = rgb2hsv((0, 249, 199))

        return [lower, upper]

    @staticmethod
    def get_goal_bounds_hsv():
        lower = rgb2hsv((0, 0, 0))
        upper = rgb2hsv((0, 0, 8))

        return [lower, upper]

    @staticmethod
    def generate_frame_mask(width, height):
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

    @staticmethod
    def mask_img(image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    def track(self, frame):
        frame = self.mask_img(frame, self.frame_mask)

        detected_ball = self.ball_detection.detect(frame)

        if detected_ball is not None:
            [center, bbox] = detected_ball
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)

        info = [("Track length", f"{sum([1 for p in self.ball_track if p is not None])}")]
        if self.ball_calibration:
            [lower_rgb, upper_rgb] = self.ball_detection.get_bounds("rgb")
            info.append(("Ball Lower RGB", f'{lower_rgb}'))
            info.append(("Ball Upper RGB", f'{upper_rgb}'))
        return detected_ball, info

    def render_track(self, frame, render_mask=False):

        if render_mask:
            frame = cv2.bitwise_and(frame, frame, mask=self.frame_mask)

        # loop over the set of tracked points
        for i in range(1, len(self.ball_track)):
            # if either of the tracked points are None, ignore
            # them
            if self.ball_track[i - 1] is None or self.ball_track[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            r = 255 - (255 * (i / self.ball_track.maxlen))
            g = 255 * (i / self.ball_track.maxlen)
            b = 255 - (255 * (i / self.ball_track.maxlen))
            thickness = int(np.sqrt(self.ball_track.maxlen / float(i + 1)) * 2.5)
            cv2.line(frame, self.ball_track[i - 1], self.ball_track[i], (r, g, b), thickness)

        return frame
