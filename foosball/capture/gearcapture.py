import logging
import cv2
from vidgear.gears import VideoGear

logger = logging.getLogger(__name__)

class GearCapture:

    def __init__(self, video=None, **kwargs):
        self.source = 'file' if isinstance(video, str) else 'cam'
        # if a video path was not supplied, grab the reference
        # to the webcam
        if video is None or type(video) == int:
            resolution = kwargs.get('resolution', (640, 480))
            options = {
                 "CAP_PROP_FRAME_WIDTH": resolution[0],  # resolution 320x240
                 "CAP_PROP_FRAME_HEIGHT": resolution[1],
                 "CAP_PROP_FPS": kwargs.get('framerate'),  # framerate 60fps
            }
            self.cap = VideoGear(source=video or 0, logging=True, **options).start()
            # otherwise, grab a reference to the video file
        else:
            options = {
                # "CAP_PROP_FRAME_WIDTH": 800,  # resolution 320x240
                # "CAP_PROP_FRAME_HEIGHT": 600,
                # "CAP_PROP_FPS": 60,  # framerate 60fps
            }
            self.cap = VideoGear(source=video, logging=False, **options).start()
            logger.info(f"framerate = {self.cap.framerate}")

        self.is_file_capture = video is not None

    def next(self):
        return self.cap.read()

    def stop(self):
        self.cap.stop()

    def stream(self):
        if self.source == 'file':
            return self.cap.stream.stream
        else:
            return self.cap.stream.stream

    def dim(self):
        width = int(self.stream().get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream().get(cv2.CAP_PROP_FRAME_HEIGHT))

        return [width, height]
