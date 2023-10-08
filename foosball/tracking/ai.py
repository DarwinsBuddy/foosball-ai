import logging
import traceback
from queue import Empty

from imutils.video import FPS

from . import Tracking, get_ball_config, get_goal_config
from .render import r_text, BLACK
from ..capture.Stream import Stream
from ..display.cv import OpenCVDisplay, get_slider_config, add_config_input, reset_config, Key
from ..models import FrameDimensions, Frame

BLANKS = (' ' * 80)


class AI:

    def __init__(self, stream: Stream, dis, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.logger = logging.getLogger("AI")
        self.stream = stream
        self.headless = kwargs.get('headless')
        self.display = dis if not self.headless else None
        self.paused = False
        self.calibration = self.kwargs.get('calibration')
        self._stopped = False
        self.ball_config = get_ball_config(self.kwargs.get('ball'))
        self.goals_config = get_goal_config()

        if self.calibration is not None:
            self.calibration_config = lambda: self.ball_config if self.calibration == 'ball' else self.goals_config
        self.detection_frame = None

        original = self.stream.dim()
        self.scale = self.kwargs.get('scale')
        scaled = self.scale_dim(original, self.scale)
        self.dims = FrameDimensions(original, scaled, self.scale)

        self.tracking = Tracking(self.stream, self.dims, self.ball_config, self.goals_config, **self.kwargs)

        if not self.headless and self.calibration is not None:
            self.calibration_display = OpenCVDisplay(self.calibration, pos='br')
            # init slider window
            add_config_input(self.calibration, self.calibration_config())

        self.fps = FPS()

    def stop(self):
        self._stopped = True

    def process_video(self):
        def reset_calibration():
            if self.calibration is not None:
                reset_config(self.calibration, self.calibration_config())
            return False

        def store_calibration():
            if self.calibration is not None:
                self.calibration_config().store()
            return False

        def pause():
            if self.paused:
                self.tracking.resume()
            else:
                self.tracking.pause()

            self.paused = not self.paused
            self.logger.info("PAUSE" if self.paused else "RESUME")
            return False

        def step_frame():
            if self.paused:
                self.logger.info("STEP")
                self.tracking.step()
            return False

        self.tracking.start()

        callbacks = {
            ord('q'): lambda: True,
            Key.SPACE.value: pause,
            ord('s'): store_calibration,
            ord('r'): reset_calibration,
            ord('n'): step_frame
        }

        self.fps.start()
        while not self._stopped:
            try:
                try:
                    self.adjust_calibration()
                    msg = self.tracking.output.get(block=False)
                    if msg is None:
                        self.logger.debug("received SENTINEL")
                        break
                    self.fps.update()
                    f = msg.kwargs['result']
                    self.fps.stop()
                    fps = int(self.fps.fps())
                    if not self.headless:
                        self.render_fps(f, fps)
                        self.display.show(f)
                        if self.calibration is not None:
                            self.render_calibration()
                        if self.display.render(callbacks=callbacks):
                            break
                    else:
                        print(f"{f} - FPS: {fps} {BLANKS}", end="\r")
                except Empty:
                    # logger.debug("No new frame")
                    pass
                if not self.headless and self.display.render(callbacks=callbacks):
                    break
            except Exception as e:
                self.logger.error(f"Error in stream {e}")
                traceback.print_exc()

        if not self.headless:
            self.display.stop()
        if self.calibration is not None:
            self.calibration_display.stop()
        self.tracking.stop()
        logging.debug("ai stopped")


    def render_fps(self, frame: Frame, fps: int):
        frames_per_second = fps
        if frames_per_second >= 90:
            color = (0, 255, 0)
        elif frames_per_second >= 75:
            color = (0, 255, 127)
        else:
            color = (100, 0, 255)
        r_text(frame, f"FPS: {frames_per_second}", frame.shape[1], 0, color, background=BLACK, text_scale=0.5, thickness=1, padding=(20, 20), ground_zero='tr')

    def render_calibration(self):
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
