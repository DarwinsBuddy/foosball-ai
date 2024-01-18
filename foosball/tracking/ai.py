import logging
import traceback
from queue import Empty

from imutils.video import FPS
from vidgear.gears import WriteGear

from . import Tracking, get_ball_config, get_goal_config
from .render import r_text, BLACK
from ..source import Source
from ..sink.opencv import DisplaySink, get_slider_config, add_config_input, reset_config, Key
from ..models import FrameDimensions, Frame

BLANKS = (' ' * 80)


class AI:

    def __init__(self, source: Source, dis, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.logger = logging.getLogger("AI")
        self.source = source
        self.headless = kwargs.get('headless')
        self.sink = dis if not self.headless else None
        self.paused = False
        self.calibration = self.kwargs.get('calibration')
        self._stopped = False
        self.ball_config = get_ball_config(self.kwargs.get('ball'))
        self.goals_config = get_goal_config()

        self.output = None if kwargs.get('output') is None else WriteGear(kwargs.get('output'), logging=True)

        if self.calibration is not None:
            self.calibration_config = lambda: self.ball_config if self.calibration == 'ball' else self.goals_config
        self.detection_frame = None

        original = self.source.dim()
        self.scale = self.kwargs.get('scale')
        scaled = self.scale_dim(original, self.scale)
        self.dims = FrameDimensions(original, scaled, self.scale)

        self.tracking = Tracking(self.source, self.dims, self.ball_config, self.goals_config, **self.kwargs)

        if not self.headless and self.calibration is not None:
            self.calibration_display = DisplaySink(self.calibration, pos='br')
            # init slider window
            add_config_input(self.calibration, self.calibration_config())

        self.fps = FPS()

    def set_calibration_config(self, config: dict):
        if self.calibration is not None:
            if self.calibration == 'ball':
                self.ball_config = config
            else:
                self.goals_config = config

    def stop(self):
        self._stopped = True

    def process_video(self):
        def reset_calibration():
            if self.calibration is not None:
                reset_config(self.calibration, self.calibration_config())
            return False

        def reset_score():
            self.tracking.reset_score()
            return False

        def store_calibration():
            if self.calibration is not None:
                self.calibration_config().store()
            else:
                logging.info("calibration not found. config not stored")
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
            ord('r'): reset_calibration if self.calibration else reset_score,
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
                        self.sink.show(f)
                        if self.output is not None:
                            self.output.write(f)
                        if self.calibration is not None:
                            self.render_calibration()
                        if self.sink.render(callbacks=callbacks):
                            break
                    else:
                        print(f"{f} - FPS: {fps} {BLANKS}", end="\r")
                except Empty:
                    # logger.debug("No new frame")
                    pass
                if not self.headless and self.sink.render(callbacks=callbacks):
                    break
            except Exception as e:
                self.logger.error(f"Error in stream {e}")
                traceback.print_exc()

        if not self.headless:
            self.sink.stop()
        if self.calibration is not None:
            self.calibration_display.stop()
        self.tracking.stop()
        logging.debug("ai stopped")

    @staticmethod
    def render_fps(frame: Frame, fps: int):
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
            new_config = get_slider_config(self.calibration)
            if new_config != self.calibration_config():
                self.set_calibration_config(new_config)
                self.tracking.config_input(self.calibration_config())

    @staticmethod
    def scale_dim(dim, scale_percent):

        # calculate the percent of original dimensions
        width = int(dim[0] * scale_percent)
        height = int(dim[1] * scale_percent)
        return [width, height]
