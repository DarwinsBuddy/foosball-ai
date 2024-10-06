import logging
import traceback
from queue import Empty

from imutils.video import FPS
from vidgear.gears import WriteGear

from const import HEADLESS, CALIBRATION_MODE, INFO_VERBOSITY, OUTPUT, CalibrationMode, SCALE, BALL
from . import Tracking
from .render import r_text, BLACK
from ..detectors.color import GoalColorConfig, BallColorDetector, BallColorConfig, GoalColorDetector
from ..models import FrameDimensions, Frame, InfoLog, Verbosity
from ..sink.opencv import DisplaySink, Key, BallColorCalibration, GoalColorCalibration
from ..source import Source

BLANKS = (' ' * 80)


class AI:

    def __init__(self, source: Source, dis, *args, **kwargs):
        self.args = args
        self.logger = logging.getLogger("AI")
        self.source = source
        self.headless = kwargs.get(HEADLESS)
        self.sink = dis if not self.headless else None
        self.paused = False
        self.calibrationMode = kwargs.get(CALIBRATION_MODE)
        self._stopped = False
        self.infoVerbosity = Verbosity(kwargs.get(INFO_VERBOSITY)) if kwargs.get(INFO_VERBOSITY) else None

        self.output = None if kwargs.get(OUTPUT) is None else WriteGear(kwargs.get(OUTPUT), logging=True)
        self.detection_frame = None

        original = self.source.dim()
        self.scale = kwargs.get(SCALE)
        scaled = self.scale_dim(original, self.scale)
        self.dims = FrameDimensions(original, scaled, self.scale)

        self.goal_detector = GoalColorDetector(GoalColorConfig.preset())
        self.ball_detector = BallColorDetector(BallColorConfig.preset(kwargs.get(BALL)))

        self.tracking = Tracking(self.source, self.dims, self.goal_detector, self.ball_detector, **kwargs)
        if not self.headless and self.calibrationMode is not None:
            # init calibration window
            self.calibration_display = DisplaySink(self.calibrationMode, pos='br')
        match self.calibrationMode:
            case CalibrationMode.BALL:
                self.calibration = BallColorCalibration(self.ball_detector.config)
            case CalibrationMode.GOAL:
                self.calibration = GoalColorCalibration(self.goal_detector.config)
            case _:
                self.calibration = None

        self.fps = FPS()

    def stop(self):
        self._stopped = True

    def process_video(self):
        def reset_calibration():
            self.calibration.reset()
            return False

        def reset_score():
            self.tracking.reset_score()
            return False

        def store_calibration():
            self.calibration.store()
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
                    frame = msg.kwargs.get('Renderer', None)
                    info: InfoLog = msg.info
                    self.fps.stop()
                    fps = int(self.fps.fps())
                    if not self.headless:
                        self.render_fps(frame, fps)
                        self.sink.show(frame)
                        if self.output is not None:
                            self.output.write(frame)
                        if self.calibration is not None:
                            self.render_calibration()
                        if self.sink.render(callbacks=callbacks):
                            break
                    else:
                        print(
                            f"{info.filter(self.infoVerbosity).to_string() if self.infoVerbosity is not None else ''} - FPS: {fps} {BLANKS}",
                            end="\r")
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
        r_text(frame, f"FPS: {frames_per_second}", frame.shape[1], 0, color, background=BLACK, text_scale=0.5,
               thickness=1, padding=(20, 20), ground_zero='tr')

    def render_calibration(self):
        try:
            self.detection_frame = self.tracking.calibration_output.get_nowait()
            self.calibration_display.show(self.detection_frame)
        except Empty:
            pass

    def adjust_calibration(self):
        # see if some sliders changed
        if self.calibrationMode in [CalibrationMode.GOAL, CalibrationMode.BALL]:
            new_config = self.calibration.get_slider_config()
            if new_config != self.calibration.config:
                self.calibration.config = new_config
                self.tracking.config_input(self.calibration.config)

    @staticmethod
    def scale_dim(dim, scale_percent):

        # calculate the percent of original dimensions
        width = int(dim[0] * scale_percent)
        height = int(dim[1] * scale_percent)
        return [width, height]
