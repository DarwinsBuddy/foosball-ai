import cv2
import imutils

from ..display.cv import OpenCVDisplay, add_calibration_input, reset_bounds, get_slider_bounds
from ..utils import hsv2rgb


class ColorDetection:

    def __init__(self, name, bounds_hsv, calibration=False, verbose=False):
        self.verbose = verbose
        self.calibration = calibration
        self.bounds_hsv = bounds_hsv
        self.name = name
        self.init_lower_hsv, self.init_upper_hsv = self.bounds_hsv

        if self.calibration:
            self.calibration_display = OpenCVDisplay(self.name, pos='br')
            # init slider window
            add_calibration_input(self.name, self.init_lower_hsv, self.init_upper_hsv)

    def reset_bounds(self, lower_hsv, upper_hsv):
        if self.calibration_display is not None:
            reset_bounds(self.name, lower_hsv, upper_hsv)

    def detect(self, frame):
        # see if some sliders changed
        if self.calibration:
            self.bounds_hsv = get_slider_bounds(self.name)

        detection_frame = self.filter_color_range(frame, self.bounds_hsv)

        if self.verbose or self.calibration:
            self.calibration_display.show(detection_frame)

        detected_blob = self.detect_largest_blob(detection_frame)

        return detected_blob

    def get_bounds(self, mode="hsv"):
        if mode == "hsv":
            return self.bounds_hsv
        else:
            return [hsv2rgb(x) for x in self.bounds_hsv]

    @staticmethod
    def filter_color_range(frame, bounds):
        [lower, upper] = bounds
        blurred = cv2.GaussianBlur(frame, (1, 1), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the simple frame
        simple = cv2.inRange(hsv, lower, upper)
        simple = cv2.erode(simple, None, iterations=2)
        simple = cv2.dilate(simple, None, iterations=2)
        # if verbose:
        #     self.display.show("dilate", simple, 'bl')

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
