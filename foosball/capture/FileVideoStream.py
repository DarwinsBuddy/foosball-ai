import cv2
import numpy as np

from foosball.capture.Stream import Stream


class FileVideoStream(Stream):

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(path)
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

        return [width, height]
