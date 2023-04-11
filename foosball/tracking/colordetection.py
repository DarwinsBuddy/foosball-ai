import logging

import cv2
import imutils

from .models import DetectionResult, Frame, Blob
from ..utils import hsv2rgb


def detect(frame, bounds_hsv, **kwargs) -> DetectionResult:
    if bounds_hsv is not None:
        detection_frame = filter_color_range(frame, bounds_hsv)
        detected_blob = detect_largest_blob(detection_frame)
        return DetectionResult(blob=detected_blob, frame=detection_frame)
    else:
        logging.error("ColorDetection not possible. bounds are None")


def get_bounds(bounds_hsv, mode="hsv"):
    if mode == "hsv":
        return bounds_hsv
    else:
        return [hsv2rgb(x) for x in bounds_hsv]


def filter_color_range(frame, bounds) -> Frame:
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


def detect_largest_blob(img) -> Blob | None:
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

        return Blob(center=center, bbox=[x, y, w, h])
    return None
