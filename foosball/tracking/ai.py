from queue import Empty

import cv2
from imutils.video import FPS

from . import Tracking, FrameDimensions
from .render import r_text


class AI:

    def __init__(self, cap, dis, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cap = cap
        self.display = dis
        self._stopped = False

    def stop(self):
        self._stopped = True

    def process_video(self):
        ball_calibration = self.kwargs.get('calibration') in ['all', 'ball']
        scale_percentage = 0.4
        original = self.cap.dim()
        scaled_dims = self.scale_dim(original, scale_percentage)
        dims = FrameDimensions(original, scaled_dims, scale_percentage)
        def reset_cb():
            # TODO: Fix this by calibrate in main process and update through a queue
            if ball_calibration:
                tracking.reset()

        tracking = Tracking(dims, **self.kwargs)
        tracking.build()
        out = tracking.output()
        tracking.start()

        fps = FPS()
        fps.start()

        while not self._stopped:
            fps.update()
            f = self.cap.next()
            if f is not None:
                frame = self.scale(f, scaled_dims)
                tracking.track(frame)
                try:
                    frame = out.get(block=False)
                    fps.stop()
                    r_text(frame, f"FPS: {int(fps.fps())}", scaled_dims[0] - 60, scaled_dims[1] - 10, scale_percentage)
                except Empty:
                    # logging.debug("No new frame")
                    pass
                self.display.show(frame)
                if self.display.render(reset_cb=reset_cb):
                    break
            else:
                break

        self.cap.stop()
        self.display.stop()
        tracking.stop()

    @staticmethod
    def scale(src, dim):
        return cv2.resize(src, dim)
    @staticmethod
    def scale_dim(dim, scale_percent):

        # calculate the percent of original dimensions
        width = int(dim[0] * scale_percent)
        height = int(dim[1] * scale_percent)
        return [width, height]

#    @staticmethod
#    def scale(src, scale_percent):
#
#        # calculate the percent of original dimensions
#        width = int(src.shape[1] * scale_percent)
#        height = int(src.shape[0] * scale_percent)
#
#        dim = (width, height)
#        return [cv2.resize(src, dim), dim]


