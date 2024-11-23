import logging
import traceback
from multiprocessing import Queue
from queue import Empty

from const import CalibrationMode
from .models import TrackerResult
from ..preprocess import WarpMode, project_blob
from ...detectors.color import BallColorDetector, BallColorConfig
from ...models import Track, Info, Blob, Goals, Verbosity
from ...pipe.BaseProcess import BaseProcess, Msg
from ...pipe.Pipe import clear
from ...utils import generate_processor_switches
logger = logging.getLogger(__name__)


class Tracker(BaseProcess):

    def __init__(self, ball_detector: BallColorDetector, useGPU: bool = False, buffer=16, off=False, verbose=False,
                 calibrationMode=None, **kwargs):
        super().__init__(name="Tracker")
        self.ball_track = Track(maxlen=buffer)
        self.off = off
        self.verbose = verbose
        self.calibrationMode = calibrationMode
        [self.proc, self.iproc] = generate_processor_switches(useGPU)
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.calibration = self.calibrationMode == CalibrationMode.BALL
        self.ball_detector = ball_detector
        self.bounds_in = Queue() if self.calibration else None
        self.calibration_out = Queue() if self.calibration else None

    def close(self) -> None:
        if self.calibration:
            clear(self.bounds_in)
            self.bounds_in.close()
            clear(self.calibration_out)
            self.calibration_out.close()

    def update_ball_track(self, detected_ball: Blob) -> Track | None:
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def get_info(self, ball_track: Track | None) -> [Info]:
        info = [
            Info(verbosity=Verbosity.DEBUG, title="Track length", value=f"{str(sum([1 for p in ball_track or [] if p is not None])).rjust(2, ' ')}"),
            Info(verbosity=Verbosity.TRACE, title="Calibration", value=f"{self.calibrationMode if self.calibrationMode is not None else 'off'}"),
            Info(verbosity=Verbosity.TRACE, title="Tracker", value=f"{'off' if self.off else 'on'}")
        ]
        if self.calibration:
            [lower, upper] = self.ball_detector.config.bounds
            info.append(Info(verbosity=Verbosity.TRACE, title="lower", value=f'({",".join(map(str,lower))})'))
            info.append(Info(verbosity=Verbosity.TRACE, title="upper", value=f'({",".join(map(str,upper))})'))
            info.append(Info(verbosity=Verbosity.TRACE, title="invert frame", value=f'{self.ball_detector.config.invert_frame}'))
            info.append(Info(verbosity=Verbosity.TRACE, title="invert mask", value=f'{self.ball_detector.config.invert_mask}'))
        return info

    @property
    def calibration_output(self) -> Queue:
        return self.calibration_out

    def config_input(self, config: BallColorConfig) -> None:
        if self.calibration:
            self.bounds_in.put_nowait(config)

    def process(self, msg: Msg) -> Msg:
        preprocess_result = msg.data['Preprocessor']
        data = preprocess_result
        ball = None
        goals = data.goals
        ball_track = None
        viewbox = data.viewbox
        tracker_info = []
        try:
            if not self.off:
                if self.calibration:
                    try:
                        self.ball_detector.config = self.bounds_in.get_nowait()
                    except Empty:
                        pass
                f = self.proc(data.preprocessed if data.preprocessed is not None else data.original)
                ball_detection_result = self.ball_detector.detect(f)
                ball = ball_detection_result.ball
                # do not forget to project detected points onto the original frame on rendering
                if not self.verbose:
                    if ball is not None and data.homography_matrix is not None:
                        ball = project_blob(ball, data.homography_matrix, WarpMode.DEWARP)
                    if goals is not None and data.homography_matrix is not None:
                        goals = Goals(
                            left=project_blob(goals.left, data.homography_matrix, WarpMode.DEWARP),
                            right=project_blob(goals.right, data.homography_matrix, WarpMode.DEWARP)
                        )
                if self.calibration:
                    self.calibration_out.put_nowait(self.iproc(ball_detection_result.frame))
                    # copy deque, since we otherwise run into odd tracks displayed
                ball_track = self.update_ball_track(ball).copy()
            tracker_info = self.get_info(ball_track)
        except Exception as e:
            logger.error(f"Error in track {e}")
            traceback.print_exc()
        # Not passing original msg due to performance impact (copying whole frames, etc.)
        if not self.verbose:
            return Msg(info=tracker_info, data={"Tracker": TrackerResult(frame=data.original, goals=goals, ball_track=ball_track, ball=ball, viewbox=viewbox)})
        else:
            return Msg(info=tracker_info, data={"Tracker": TrackerResult(frame=data.preprocessed, goals=goals, ball_track=ball_track, ball=ball, viewbox=viewbox)})
