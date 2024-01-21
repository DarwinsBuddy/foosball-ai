from multiprocessing import Queue

import cv2
import numpy as np

from const import GPU, CalibrationMode
from .analyze import Analyzer
from .preprocess import PreProcessor
from .render import Renderer
from .tracker import Tracker
from ..detectors import BallDetector
from ..detectors.color import GoalDetector
from ..models import Mask, FrameDimensions
from ..pipe.Pipe import Pipe


def dim(frame) -> [int, int]:
    return [frame.shape[1], frame.shape[0]]


def generate_frame_mask(width, height) -> Mask:
    bar_color = 255
    bg = 0
    mask = np.full((height, width), bg, np.uint8)
    start = (int(width / 12), int(height / 20))
    end = (int(width / 1.2), int(height / 1.2))
    frame_mask = cv2.rectangle(mask, start, end, bar_color, -1)
    return frame_mask


class Tracking:

    def __init__(self, stream, dims: FrameDimensions, goal_detector: GoalDetector, ball_detector: BallDetector, headless=False, maxPipeSize=128, calibrationMode=None, **kwargs):
        super().__init__()
        self.calibrationMode = calibrationMode

        width, height = dims.scaled
        mask = generate_frame_mask(width, height)
        gpu_flags = kwargs.get(GPU)
        self.preprocessor = PreProcessor(dims, goal_detector, mask=mask, headless=headless, useGPU='preprocess' in gpu_flags,
                                         calibrationMode=calibrationMode, **kwargs)
        self.tracker = Tracker(ball_detector, useGPU='tracker' in gpu_flags, calibrationMode=calibrationMode, **kwargs)
        self.analyzer = Analyzer(**kwargs)
        self.renderer = Renderer(dims, headless=headless, useGPU='render' in gpu_flags, **kwargs)

        self.stream = stream
        self.pipe = Pipe(stream, [self.preprocessor, self.tracker, self.analyzer, self.renderer], maxsize=maxPipeSize)
        self.frame_queue = self.pipe.input

    def start(self):
        self.pipe.start()

    def stop(self):
        self.pipe.stop()

    def pause(self):
        self.pipe.pause()

    def resume(self):
        self.pipe.resume()

    def step(self):
        self.pipe.step()

    @property
    def output(self) -> Queue:
        return self.pipe.output

    def status(self) -> str:
        return self.pipe.status()

    @property
    def calibration_output(self) -> Queue:
        match self.calibrationMode:
            case CalibrationMode.BALL:
                return self.tracker.calibration_out
            case CalibrationMode.GOAL:
                return self.preprocessor.calibration_out

    def config_input(self, config) -> None:
        match self.calibrationMode:
            case CalibrationMode.BALL:
                return self.tracker.config_input(config)
            case CalibrationMode.GOAL:
                return self.preprocessor.config_input(config)

    def reset_score(self):
        self.analyzer.reset_score()
