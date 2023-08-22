import logging
import traceback
from queue import Empty

from imutils.video import FPS

from . import Tracking, get_ball_config, get_goal_config
from .render import r_text
from ..models import FrameDimensions, ScaleDirection
from ..utils import scale
from ..display.cv import OpenCVDisplay, get_slider_config, add_config_input, reset_config, store_config, Key


class AI:

    def __init__(self, cap, dis, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cap = cap
        self.display = dis
        self.paused = False
        self.step = False
        self.calibration = self.kwargs.get('calibration')
        self._stopped = False
        self.ball_config = get_ball_config(self.kwargs.get('ball'))
        self.goals_config = get_goal_config()
        if self.calibration is not None:
            self.calibration_config = lambda: self.ball_config if self.calibration == 'ball' else self.goals_config
        self.detection_frame = None

        original = self.cap.dim()
        self.scale = self.kwargs.get('scale')
        scaled = self.scale_dim(original, self.scale)
        self.dims = FrameDimensions(original, scaled, self.scale)

        self.tracking = Tracking(self.dims, self.ball_config, self.goals_config, **self.kwargs)

        if self.calibration is not None:
            self.calibration_display = OpenCVDisplay(self.calibration, pos='br')
            # init slider window
            add_config_input(self.calibration, self.calibration_config())

    def stop(self):
        self._stopped = True

    def process_video(self):
        def reset_calibration():
            reset_config(self.calibration, self.calibration_config())
            return False

        def store_calibration():
            store_config(self.calibration, self.calibration_config())
            return False

        def pause():
            self.paused = not self.paused
            logging.info("PAUSE" if self.paused else "RESUME")
            return False

        def step_frame():
            if not self.step and self.paused:
                logging.info("STEP")
                self.step = True
            return False

        self.tracking.start()

        callbacks = {
            ord('q'): lambda: True,
            Key.SPACE.value: pause,
            ord('s'): store_calibration,
            ord('r'): reset_calibration,
            ord('n'): step_frame
        }

        fps = FPS()
        fps.start()
        f = None
        while not self._stopped:
            try:
                fps.update()
                process_frame = not self.paused or self.step
                if process_frame:
                    f = self.cap.next()
                    if f is not None:
                        f = scale(f, self.dims, ScaleDirection.DOWN)
                if f is not None:
                    if process_frame:
                        self.step = False
                        self.adjust_calibration()
                        self.tracking.track(f)
                        try:
                            f = self.tracking.output.get(block=True)
                            fps.stop()
                            frames_per_second = int(fps.fps())
                            if frames_per_second >= 90:
                                color = (0, 255, 0)
                            elif frames_per_second >= 75:
                                color = (0, 255, 127)
                            else:
                                color = (100, 0, 255)
                            r_text(f, f"FPS: {frames_per_second}", self.dims.scaled[0] - 60, self.dims.scaled[1] - 10,
                                   self.dims.scale, color)
                        except Empty:
                            # logging.debug("No new frame")
                            pass
                        self.display.show(f)
                        self.render_calibration()
                    if self.display.render(callbacks=callbacks):
                        break
                else:
                    break
            except Exception as e:
                logging.error(f"Error in stream {e}")
                traceback.print_exc()

        self.cap.stop()
        self.tracking.stop()
        self.display.stop()
        if self.calibration is not None:
            self.calibration_display.stop()

    def render_calibration(self):
        if self.calibration is not None:
            try:
                self.detection_frame = self.tracking.calibration_output.get_nowait()
                self.calibration_display.show(self.detection_frame)
            except Empty:
                pass

    def adjust_calibration(self):
        # see if some sliders changed
        if self.calibration in ["goal", "ball"]:
            self.tracking.config_input(get_slider_config(self.calibration))

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
