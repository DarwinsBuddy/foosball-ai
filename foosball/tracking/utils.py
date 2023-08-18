import cv2

from foosball.tracking.models import FrameDimensions, ScaleDirection, Point, BBox


def scale_point(pt, dimensions: FrameDimensions, scaling: ScaleDirection):
    if scaling == ScaleDirection.DOWN:
        return pt * dimensions.scale
    else:
        return pt / dimensions.scale


def scale(frame, dimensions: FrameDimensions, scaling: ScaleDirection):
    return cv2.resize(frame, dimensions.scaled if scaling == ScaleDirection.DOWN else dimensions.original)


def contains(bbox: BBox, pt: Point) -> bool:
    [x, y, w, h] = bbox
    return x < pt[0] < x + w and y < pt[1] < y + h
