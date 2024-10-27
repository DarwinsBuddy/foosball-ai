import cv2
import numpy as np

from . import Source


class OpenCVSource(Source):

    def __init__(self, source: str | int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(source)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def next_frame(self) -> (bool, np.array):
        return self.cap.read()

    def is_eos(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.total_frames

    def close_capture(self):
        self.cap.release()

    def dim(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return tuple((width, height))
