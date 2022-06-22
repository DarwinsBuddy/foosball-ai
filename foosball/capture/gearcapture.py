import cv2
from imutils.video import VideoStream

from vidgear.gears import VideoGear


class GearCapture:

    def __init__(self, video=None):
        # if a video path was not supplied, grab the reference
        # to the webcam
        if video is None:
            self.cap = VideoStream(src=1).start()
        # otherwise, grab a reference to the video file
        else:
            options = {
                # "CAP_PROP_FRAME_WIDTH": 800,  # resolution 320x240
                # "CAP_PROP_FRAME_HEIGHT": 600,
                # "CAP_PROP_FPS": 60,  # framerate 60fps
            }
            self.cap = VideoGear(source=video, logging=False, **options).start()

        self.is_file_capture = video is not None

    def next(self):
        return self.cap.read()

    def stop(self):
        self.cap.stop()

    def dim(self):
        width = int(self.cap.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return [width, height]

