import logging
import traceback
from multiprocessing import Queue
from queue import Empty

from const import CalibrationMode
from .colordetection import detect_ball
from .preprocess import WarpMode, project_blob
from ..models import TrackResult, Track, BallConfig, Info, Blob, Goals, InfoLog
from ..pipe.BaseProcess import BaseProcess, Msg
from ..pipe.Pipe import clear
from ..utils import generate_processor_switches
logger = logging.getLogger(__name__)


def log(result: TrackResult) -> None:
    logger.debug(result.info)


class Tracker(BaseProcess):

    def __init__(self, ball_bounds: BallConfig, useGPU: bool = False, buffer=16, off=False, verbose=False,
                 calibrationMode=None, **kwargs):
        super().__init__(name="Tracker")
        self.ball_track = Track(maxlen=buffer)
        self.off = off
        self.verbose = verbose
        self.calibrationMode = calibrationMode
        [self.proc, self.iproc] = generate_processor_switches(useGPU)
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.ball_bounds = ball_bounds
        self.ball_calibration = self.calibrationMode == CalibrationMode.BALL
        self.calibration_bounds = lambda: self.ball_bounds if self.ball_calibration else None
        self.bounds_in = Queue() if self.ball_calibration else None
        self.calibration_out = Queue() if self.ball_calibration else None

    def close(self) -> None:
        if self.ball_calibration:
            clear(self.bounds_in)
            self.bounds_in.close()
            clear(self.calibration_out)
            self.calibration_out.close()

    def update_ball_track(self, detected_ball: Blob) -> Track:
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def get_info(self, ball_track: Track) -> InfoLog:
        info = InfoLog(infos=[
            Info(verbosity=1, title="Track length", value=f"{str(sum([1 for p in ball_track or [] if p is not None])).rjust(2, ' ')}"),
            Info(verbosity=0, title="Calibration", value=f"{self.calibrationMode if self.calibrationMode is not None else 'off'}"),
            Info(verbosity=0, title="Tracker", value=f"{'off' if self.off else 'on'}")
        ])
        if self.ball_calibration:
            [lower, upper] = self.calibration_bounds().bounds
            info.append(Info(verbosity=0, title=f"lower", value=f'({",".join(map(str,lower))})'))
            info.append(Info(verbosity=0, title=f"upper", value=f'({",".join(map(str,upper))})'))
            info.append(Info(verbosity=0, title=f"invert frame", value=f'{self.calibration_bounds().invert_frame}'))
            info.append(Info(verbosity=0, title=f"invert mask", value=f'{self.calibration_bounds().invert_mask}'))
        return info

    @property
    def calibration_output(self) -> Queue:
        return self.calibration_out

    def config_input(self, config: BallConfig) -> None:
        if self.ball_calibration:
            self.bounds_in.put_nowait(config)

    def process(self, msg: Msg) -> Msg:
        preprocess_result = msg.kwargs['result']
        ball = None
        goals = preprocess_result.goals
        ball_track = None
        info = None
        try:
            if not self.off:
                if self.ball_calibration:
                    try:
                        self.ball_bounds = self.bounds_in.get_nowait()
                    except Empty:
                        pass
                f = self.proc(preprocess_result.preprocessed if preprocess_result.preprocessed is not None else preprocess_result.original)
                ball_detection_result = detect_ball(f, self.ball_bounds)
                ball = ball_detection_result.ball
                # do not forget to project detected points onto the original frame on rendering
                if not self.verbose:
                    if ball is not None and preprocess_result.homography_matrix is not None:
                        ball = project_blob(ball, preprocess_result.homography_matrix, WarpMode.DEWARP)
                    if goals is not None and preprocess_result.homography_matrix is not None:
                        goals = Goals(
                            left=project_blob(goals.left, preprocess_result.homography_matrix, WarpMode.DEWARP),
                            right=project_blob(goals.right, preprocess_result.homography_matrix, WarpMode.DEWARP)
                        )
                if self.ball_calibration:
                    self.calibration_out.put_nowait(self.iproc(ball_detection_result.frame))
                    # copy deque, since we otherwise run into odd tracks displayed
                ball_track = self.update_ball_track(ball).copy()
            info = preprocess_result.info
            info.concat(self.get_info(ball_track))
        except Exception as e:
            logger.error(f"Error in track {e}")
            traceback.print_exc()
        if not self.verbose:
            return Msg(kwargs={"result": TrackResult(preprocess_result.original, goals, ball_track, ball, info)})
        else:
            return Msg(kwargs={"result": TrackResult(preprocess_result.preprocessed, goals, ball_track, ball, info)})
