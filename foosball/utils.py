from typing import Callable

import cv2

from .models import FrameDimensions, ScaleDirection, Point, BBox, Frame, CPUFrame, GPUFrame


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


def to_gpu(frame: CPUFrame) -> GPUFrame:
    return cv2.UMat(frame)


def from_gpu(frame: GPUFrame) -> CPUFrame:
    return frame.get()


def relative_change(old_value, new_value):
    return (new_value / old_value) - 1
def generate_processor_switches(useGPU: bool = False) -> [Callable[[Frame], Frame], Callable[[Frame], Frame]]:
    if not useGPU:
        return [lambda x: x, lambda x: x]
    else:
        return [to_gpu, from_gpu]


def ensure_cpu(frame: Frame) -> CPUFrame:
    if type(frame) == cv2.UMat:
        return from_gpu(frame)
    return frame
