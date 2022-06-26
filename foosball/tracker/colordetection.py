import cv2
import imutils

from ..utils import hsv2rgb, rgb2hsv


class ColorDetection:

    # TODO: find a way to support opengl sliders

    def __init__(self, name, display, bounds_hsv, calibration=False, verbose=False):
        self.verbose = verbose
        self.calibration = calibration
        self.bounds_hsv = bounds_hsv
        self.name = name
        self.init_lower_hsv, self.init_upper_hsv = self.bounds_hsv

        self.display = display
        if self.calibration:
            # init slider window
            self.display.show(self.name, None)
            self.add_calibration_input(self.init_lower_hsv, self.init_upper_hsv)

    def detect(self, frame):
        # see if some sliders changed
        if self.calibration:
            self.bounds_hsv = self.get_slider_bounds()

        detection_frame = self.filter_color_range(frame, self.bounds_hsv, self.verbose)

        if self.verbose or self.calibration:
            self.display.show(self.name, detection_frame, 'br')

        detected_blob = self.detect_largest_blob(detection_frame)

        return detected_blob

    def get_bounds(self, mode="hsv"):
        if mode == "hsv":
            return self.bounds_hsv
        else:
            return [hsv2rgb(x) for x in self.bounds_hsv]

    def filter_color_range(self, frame, bounds, verbose=False):
        [lower, upper] = bounds
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

    # CALIBRATION #

    @staticmethod
    def slider_label(rgb, bound):
        return f"{rgb} ({bound})"

    def add_calibration_input(self, lower_hsv, upper_hsv):
        lower_rgb = hsv2rgb(lower_hsv)
        upper_rgb = hsv2rgb(upper_hsv)
        # create trackbars for color change
        cv2.createTrackbar(self.slider_label('R', 'low'), self.name, lower_rgb[0], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('G', 'low'), self.name, lower_rgb[1], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('B', 'low'), self.name, lower_rgb[2], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('R', 'high'), self.name, upper_rgb[0], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('G', 'high'), self.name, upper_rgb[1], 255, lambda v: None)
        cv2.createTrackbar(self.slider_label('B', 'high'), self.name, upper_rgb[2], 255, lambda v: None)
        # cv2.createButton("Reset", reset_bounds, (self.name, lower_rgb, upper_rgb))

    def reset_bounds(self):
        print(f"Reset color bounds {self.name}")
        lower_rgb = hsv2rgb(self.init_lower_hsv)
        upper_rgb = hsv2rgb(self.init_upper_hsv)

        cv2.setTrackbarPos(self.slider_label('R', 'low'), self.name, lower_rgb[0])
        cv2.setTrackbarPos(self.slider_label('G', 'low'), self.name, lower_rgb[1])
        cv2.setTrackbarPos(self.slider_label('B', 'low'), self.name, lower_rgb[2])
        cv2.setTrackbarPos(self.slider_label('R', 'high'), self.name, upper_rgb[0])
        cv2.setTrackbarPos(self.slider_label('G', 'high'), self.name, upper_rgb[1])
        cv2.setTrackbarPos(self.slider_label('B', 'high'), self.name, upper_rgb[2])

    def get_slider_bounds(self, mode="hsv"):
        # get current positions of four trackbars
        rl = cv2.getTrackbarPos(self.slider_label('R', 'low'), self.name)
        rh = cv2.getTrackbarPos(self.slider_label('R', 'high'), self.name)

        gl = cv2.getTrackbarPos(self.slider_label('G', 'low'), self.name)
        gh = cv2.getTrackbarPos(self.slider_label('G', 'high'), self.name)

        bl = cv2.getTrackbarPos(self.slider_label('B', 'low'), self.name)
        bh = cv2.getTrackbarPos(self.slider_label('B', 'high'), self.name)
        if mode == 'hsv':
            lower = rgb2hsv((rl, gl, bl))
            upper = rgb2hsv((rh, gh, bh))
        else:
            lower = (rl, gl, bl)
            upper = (rh, gh, bh)
        return [lower, upper]