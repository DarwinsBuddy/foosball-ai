from queue import Empty

from imutils.video import FPS

from . import Tracking, FrameDimensions, ScaleDirection
from .render import r_text
from .tracker import get_ball_bounds_hsv, Bounds
from .utils import scale
from ..display.cv import add_calibration_input, OpenCVDisplay, reset_bounds, get_slider_bounds

BALL = 'ball'


class AI:

    def __init__(self, cap, dis, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cap = cap
        self.display = dis
        self.color_calibration = self.kwargs.get('colorCalibration') in ['all', 'ball']
        self.verbose = self.kwargs.get('verbose')
        self._stopped = False
        self.ball_bounds_hsv = get_ball_bounds_hsv()
        self.detection_frame = None


        original = self.cap.dim()
        scale_percentage = 0.4
        scaled = self.scale_dim(original, scale_percentage)
        self.dims = FrameDimensions(original, scaled, scale_percentage)

        self.tracking = Tracking(self.dims, **self.kwargs)

        if self.color_calibration:
            self.calibration_display = OpenCVDisplay(BALL, pos='br')
            # init slider window
            add_calibration_input(BALL, *self.ball_bounds_hsv)

    def stop(self):
        self._stopped = True

    def process_video(self):
        def reset_cb():
            if self.color_calibration:
                reset_bounds(BALL, *self.ball_bounds_hsv)

        self.tracking.start()

        fps = FPS()
        fps.start()

        while not self._stopped:
            fps.update()
            f = self.cap.next()
            if f is not None:
                f = scale(f, self.dims, ScaleDirection.DOWN)
                self.adjust_calibration()
                self.tracking.track(f)
                try:
                    f = self.tracking.output.get(block=False)
                    fps.stop()
                    frames_per_second = int(fps.fps())
                    if frames_per_second >= 90:
                        color = (0,255,0)
                    elif frames_per_second >= 75:
                        color = (0, 255, 127)
                    else:
                        color = (100, 0,255)
                    r_text(f, f"FPS: {frames_per_second}", self.dims.scaled[0] - 60, self.dims.scaled[1] - 10,
                           self.dims.scale, color)
                except Empty:
                    # logging.debug("No new frame")
                    pass
                self.display.show(f)
                self.render_calibration()
                if self.display.render(reset_cb=reset_cb):
                    break
            else:
                break

        self.cap.stop()
        self.tracking.stop()
        self.display.stop()
        if self.color_calibration:
            self.calibration_display.stop()

    def render_calibration(self):
        if self.color_calibration:
            try:
                self.detection_frame = self.tracking.calibration_output.get_nowait()
                self.calibration_display.show(self.detection_frame)
            except Empty:
                pass

    def adjust_calibration(self):
        # see if some sliders changed
        if self.color_calibration:
            self.tracking.bounds_input(Bounds(ball=get_slider_bounds(BALL)))

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
