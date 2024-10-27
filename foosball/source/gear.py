import logging
import cv2
from vidgear.gears import VideoGear

from . import Source


class GearSource(Source):

    def __init__(self, video=None, resolution=(640, 480), framerate=60, **kwargs):
        # not skipping frames is crucial
        # otherwise gear will not terminate as it's not forwarding sentinel, due to lack of explicit eos support
        super().__init__(skip_frames=False, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.eos = False
        # if a video path was not supplied, grab the reference
        # to the webcam
        if video is None or isinstance(video, int):
            options = {
                 "CAP_PROP_FRAME_WIDTH": resolution[0],
                 "CAP_PROP_FRAME_HEIGHT": resolution[1],
                 "CAP_PROP_FPS": framerate
            }
            self.gear = VideoGear(source=video or 0, logging=True, **options).start()
            # otherwise, grab a reference to the video file
        else:
            options = {
                # "CAP_PROP_FRAME_WIDTH": 800,  # resolution 320x240
                # "CAP_PROP_FRAME_HEIGHT": 600,
                # "CAP_PROP_FPS": 60,  # framerate 60fps
            }
            self.gear = VideoGear(source=video, logging=True, **options).start()
            self.logger.info(f"framerate = {self.gear.framerate}")

    def is_eos(self):
        return self.eos

    def next_frame(self):
        frame = self.gear.read()
        self.eos = (frame is not None)
        return self.eos, frame

    def close_capture(self):
        self.gear.stop()

    def dim(self):
        width = int(self.gear.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.gear.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return tuple((width, height))
