# TODO: #1 solve fps issue (we are not capturing high speed) maybe buffer it?
from collections import deque

import cv2
import imutils
import numpy as np


class Tracker:

    def __init__(self, frame_dimensions, display, ball_calibration=False, goals_calibration=False, verbose=False, track_buffer=64):
        self.ball_track = deque(maxlen=track_buffer)
        self.display = display
        self.verbose = verbose
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        # list of tracked points
        [self.lower_ball, self.upper_ball] = self.get_ball_bounds_hsv()
        [self.init_lower_ball, self.init_upper_ball] = [self.hsv2rgb(self.lower_ball), self.hsv2rgb(self.upper_ball)]

        [self.lower_goal, self.upper_goal] = self.get_goal_bounds_hsv()
        [self.init_lower_goal, self.init_upper_goal] = [self.hsv2rgb(self.lower_goal), self.hsv2rgb(self.upper_goal)]

        self.init_bounds = []
        self.ball_calibration = ball_calibration
        self.goals_calibration = goals_calibration
        # init slider window
        if self.ball_calibration:
            #cv2.namedWindow('ball')
            self.init_bounds += [['ball', self.init_lower_ball, self.init_upper_ball]]
            self.add_calibration_input('ball', self.hsv2rgb(self.lower_ball), self.hsv2rgb(self.upper_ball))
        if self.goals_calibration:
            #cv2.namedWindow('goals')
            self.init_bounds += [['goals', self.init_lower_goal, self.init_upper_goal]]
            self.add_calibration_input('goals', self.hsv2rgb(self.lower_goal), self.hsv2rgb(self.upper_goal))

        width, height = frame_dimensions
        self.frame_mask = self.generate_frame_mask(width, height)

    def get_ball_bounds_hsv(self):
        # greenLower = (0, 140, 170)
        # greenUpper = (80, 255, 255)

        # green_lower = (0, 143, 175)
        # green_upper = (89, 255, 255)

        # green = [green_lower, green_upper]

        # orange_lower_rgb = (115, 70, 80)
        # orange_upper_rgb = (190, 120, 25)
        # orange_hsv = [rgb2hsv(orange_lower_rgb), rgb2hsv(orange_upper_rgb)]

        # TODO: #2 calibration for the demo footage (other ball => other values)
        lower = self.rgb2hsv((166, 94, 72))
        upper = self.rgb2hsv((0, 249, 199))

        return [lower, upper]

    def get_goal_bounds_hsv(self):
        lower = self.rgb2hsv((0, 0, 0))
        upper = self.rgb2hsv((0, 0, 8))

        return [lower, upper]

    @staticmethod
    def rgb2hsv(rgb):
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    @staticmethod
    def hsv2rgb(hsv):
        return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]

    @staticmethod
    def slider_label(rgb, bound):
        return f"{rgb} ({bound})"

    @staticmethod
    def generate_bar_mask(width, height):
        bar_positions = [250, 350, 450, 550]

        bar_width = 20
        bar_color = 0
        bg = 255
        bar_mask = np.full((height, width), bg, np.uint8)
        for x_pos in bar_positions:
            [x, y] = [x_pos, 0]
            bar_dim = [x + bar_width, y + height]
            bar_mask = cv2.rectangle(bar_mask, (x, y), bar_dim, bar_color, -1)
        return bar_mask

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

    def add_calibration_input(self, window_name, lower_rgb, upper_rgb):
        # create trackbars for color change
        cv2.createTrackbar(self.slider_label('R', 'low'), window_name, lower_rgb[0], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('G', 'low'), window_name, lower_rgb[1], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('B', 'low'), window_name, lower_rgb[2], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('R', 'high'), window_name, upper_rgb[0], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('G', 'high'), window_name, upper_rgb[1], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('B', 'high'), window_name, upper_rgb[2], 255, lambda v: None)
        # cv2.createButton("Reset", reset_bounds, (window_name, lower_rgb, upper_rgb))

    def reset_bounds(self):
        for window_name, lower, upper in self.init_bounds:
            print(f"Reset color bounds {window_name}")
            cv2.setTrackbarPos(self.slider_label('R', 'low'), window_name, lower[0])
            cv2.setTrackbarPos(self.slider_label('G', 'low'), window_name, lower[1])
            cv2.setTrackbarPos(self.slider_label('B', 'low'), window_name, lower[2])

            cv2.setTrackbarPos(self.slider_label('R', 'high'), window_name, upper[0])
            cv2.setTrackbarPos(self.slider_label('G', 'high'), window_name, upper[1])
            cv2.setTrackbarPos(self.slider_label('B', 'high'), window_name, upper[2])

    def get_slider_bounds(self, window_name):
        # get current positions of four trackbars
        rl = cv2.getTrackbarPos(self.slider_label('R', 'low'), window_name)
        rh = cv2.getTrackbarPos(self.slider_label('R', 'high'), window_name)

        gl = cv2.getTrackbarPos(self.slider_label('G', 'low'), window_name)
        gh = cv2.getTrackbarPos(self.slider_label('G', 'high'), window_name)

        bl = cv2.getTrackbarPos(self.slider_label('B', 'low'), window_name)
        bh = cv2.getTrackbarPos(self.slider_label('B', 'high'), window_name)
        lower = self.rgb2hsv((rl, gl, bl))
        upper = self.rgb2hsv((rh, gh, bh))
        return [lower, upper]

    def filter_color_range(self, frame, lower, upper, verbose=False):
        blurred = cv2.GaussianBlur(frame, (1, 1), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the simple frame
        simple = cv2.inRange(hsv, lower, upper)
        simple = cv2.erode(simple, None, iterations=2)
        simple = cv2.dilate(simple, None, iterations=2)
        if verbose:
            self.display.show("dilate", simple, 'bl')

        # ## for masking
        # cleaned = mask_img(simple, mask=bar_mask)
        # contrast = mask_img(frame, cleaned)
        # show("contrast", contrast, 'br')
        # show("frame", mask_img(frame, bar_mask), 'tl')

        return cv2.bitwise_and(frame, frame, mask=simple)

    @staticmethod
    def detect_largest_blob(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            ball_contour = max(cnts, key=cv2.contourArea)
            # ((x, y), radius) = cv2.minEnclosingCircle(c)
            [x, y, w, h] = cv2.boundingRect(ball_contour)
            ms = cv2.moments(ball_contour)
            center = (int(ms["m10"] / ms["m00"]), int(ms["m01"] / ms["m00"]))

            return [center, [x, y, w, h]]
        return None

    def track(self, frame):
        frame = self.mask_img(frame, self.frame_mask)

        # see if some sliders changed
        if self.ball_calibration:
            [self.lower_ball, self.upper_ball] = self.get_slider_bounds('ball')
        if self.goals_calibration:
            [self.lower_goal, self.upper_goal] = self.get_slider_bounds('goals')

        ball_frame = self.filter_color_range(frame, self.lower_ball, self.upper_ball, self.verbose)
        goals_frame = self.filter_color_range(frame, self.lower_goal, self.upper_goal, self.verbose)
        if self.verbose or self.ball_calibration:
            self.display.show("ball", ball_frame, 'br')
        if self.verbose or self.goals_calibration:
            self.display.show("goals", cv2.cvtColor(goals_frame, cv2.COLOR_RGB2HSV), 'bl')

        detected_ball = self.detect_largest_blob(ball_frame)

        if detected_ball is not None:
            [center, bbox] = detected_ball
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)

        info = [("Track length", f"{sum([1 for p in self.ball_track if p is not None])}")]
        if self.goals_calibration:
            info.append(("Goal Lower RGB", f'{self.hsv2rgb(self.lower_goal)}'))
            info.append(("Goal Upper RGB", f'{self.hsv2rgb(self.upper_goal)}'))
        if self.ball_calibration:
            info.append(("Ball Lower RGB", f'{self.hsv2rgb(self.lower_ball)}'))
            info.append(("Ball Upper RGB", f'{self.hsv2rgb(self.upper_ball)}'))
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
