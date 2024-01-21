import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Generic

import cv2
import imutils
import numpy as np
import yaml

from . import Detector, DetectorResult
from ..models import Frame, DetectedGoals, Point, Goal, Blob, Goals, DetectedBall, HSV

logger = logging.getLogger(__name__)

DetectorConfig = TypeVar('DetectorConfig')


@dataclass
class BallConfig:
    bounds: [HSV, HSV]
    invert_frame: bool = False
    invert_mask: bool = False

    def store(self):
        filename = f"ball.yaml"
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def load(filename='ball.yaml'):
        if os.path.isfile(filename):
            logging.info("Loading ball config ball.yaml")
            with open(filename, 'r') as f:
                c = yaml.safe_load(f)
                return BallConfig(invert_frame=c['invert_frame'], invert_mask=c['invert_mask'],
                                  bounds=np.array(c['bounds']))
        else:
            logging.info("No ball config found")
        return None

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BallConfig):
            return (all([a == b for a, b in zip(self.bounds[0], other.bounds[0])]) and
                    all([a == b for a, b in zip(self.bounds[1], other.bounds[1])]) and
                    self.invert_mask == other.invert_mask and
                    self.invert_frame == other.invert_frame)
        return False

    def to_dict(self):
        return {
            "bounds": [x.tolist() for x in self.bounds],
            "invert_frame": self.invert_frame,
            "invert_mask": self.invert_mask
        }



@dataclass
class GoalConfig:
    bounds: [int, int]
    invert_frame: bool = True
    invert_mask: bool = True

    def store(self):
        filename = f"goal.yaml"
        print(f"Store config {filename}" + (" " * 50), end="\n\n")
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def load(filename='goal.yaml'):
        with open(filename, 'r') as f:
            c = yaml.safe_load(f)
            return GoalConfig(**c)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, GoalConfig):
            return (all([a == b for a, b in zip(self.bounds, other.bounds)]) and
                    self.invert_mask == other.invert_mask and
                    self.invert_frame == other.invert_frame)
        return False

    def to_dict(self):
        return {
            "bounds": self.bounds,
            "invert_frame": self.invert_frame,
            "invert_mask": self.invert_mask
        }


class ColorDetector(Generic[DetectorConfig, DetectorResult], Detector[DetectorResult], ABC):
    def __init__(self, config: DetectorConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class BallDetector(ColorDetector[BallConfig, DetectedBall]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def detect(self, frame) -> DetectedBall:
        if self.config:
            detection_frame = filter_color_range(frame, self.config)
            detected_blob = detect_largest_blob(detection_frame)
            return DetectedBall(ball=detected_blob, frame=detection_frame)
        else:
            logger.error("Ball Detection not possible. Config is None")


class GoalDetector(ColorDetector[GoalConfig, DetectedGoals]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def detect_goal_blobs(img) -> list[Goal] | None:
        """
        We take the largest blobs that lay the most to the right and to the left,
        assuming that those are our goals
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            largest_contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
            if len(largest_contours) != 2:
                logger.error("Could not detect 2 goals")
                return None
            centers_and_bboxes = [transform_contour(cnt) for cnt in largest_contours]
            # sort key = x coordinate of the center of mass
            blobs_ordered_by_x = [Goal(center=x[0], bbox=x[1]) for x in
                                  sorted(centers_and_bboxes, key=lambda center_bbox: center_bbox[0][0])]
            return blobs_ordered_by_x
        return None

    def detect(self, frame: Frame) -> DetectedGoals:
        if self.config is not None:
            detection_frame = filter_gray_range(frame, self.config)
            detected_blobs = self.detect_goal_blobs(detection_frame)
            if detected_blobs is not None:
                return DetectedGoals(goals=Goals(left=detected_blobs[0], right=detected_blobs[1]),
                                     frame=detection_frame)
            else:
                return DetectedGoals(goals=None, frame=detection_frame)
        else:
            logger.error("Goal Detection not possible. config is None")


def filter_color_range(frame, config: BallConfig) -> Frame:
    [lower, upper] = config.bounds
    f = frame if not config.invert_frame else cv2.bitwise_not(frame)

    blurred = cv2.GaussianBlur(f, (1, 1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the chosen color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the simple frame
    simple_mask = cv2.inRange(hsv, lower, upper)
    # kernel=None <=> 3x3
    simple_mask = cv2.erode(simple_mask, None, iterations=2)
    simple_mask = cv2.dilate(simple_mask, None, iterations=2)

    simple_mask = simple_mask if not config.invert_mask else cv2.bitwise_not(simple_mask)
    # if verbose:
    #     self.display.show("dilate", simple, 'bl')

    # ## for masking
    # cleaned = mask_img(simple, mask=bar_mask)
    # contrast = mask_img(frame, cleaned)
    # show("contrast", contrast, 'br')
    # show("frame", mask_img(frame, bar_mask), 'tl')
    return cv2.bitwise_and(f, f, mask=simple_mask)


def filter_gray_range(frame, config: GoalConfig) -> Frame:
    try:
        [lower, upper] = config.bounds
        f = frame if not config.invert_frame else cv2.bitwise_not(frame)

        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # Create a binary mask using cv2.inRange
        mask = cv2.inRange(gray, lower, upper)

        # Apply morphological operations for noise reduction and region connection
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=6)
        final_mask = eroded_mask if not config.invert_mask else cv2.bitwise_not(eroded_mask)
        x = cv2.bitwise_and(f, f, mask=final_mask)
        return cv2.dilate(x, kernel, iterations=2)
    except Exception as e:
        logger.error(f"Exception: {e}\n\n")
        return frame


def detect_largest_blob(img) -> Blob | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        largest_contour = max(cnts, key=cv2.contourArea)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        center, [x, y, w, h] = transform_contour(largest_contour)

        return Blob(center=center, bbox=[x, y, w, h])
    return None


def transform_contour(contour) -> (Point, [int, int, int, int]):
    """
    calculate bounding box and center of mass
    """
    [x, y, w, h] = cv2.boundingRect(contour)
    ms = cv2.moments(contour)
    center = (int(ms["m10"] / ms["m00"]), int(ms["m01"] / ms["m00"]))
    return center, [x, y, w, h]
