import cv2
import numpy as np

from foosball.tracking.models import FrameDimensions, ScaleDirection, HSV, RGB


def scale_point(pt, dimensions: FrameDimensions, scaling: ScaleDirection):
    if scaling == ScaleDirection.DOWN:
        return pt * dimensions.scale
    else:
        return pt / dimensions.scale
def scale(frame, dimensions: FrameDimensions, scaling: ScaleDirection):
    return cv2.resize(frame, dimensions.scaled if scaling == ScaleDirection.DOWN else dimensions.original)

def rgb2hsv(rgb: RGB) -> HSV:
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]


def hsv2rgb(hsv: HSV) -> RGB:
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]
