import logging
import traceback
from queue import Empty

import pypeln as pl

from .colordetection import detect_ball
from .preprocess import WarpMode, project_blob
from ..models import TrackResult, Track, BallConfig, Info, Blob, PreprocessResult, Goals
from ..utils import generate_processor_switches


def log(result: TrackResult) -> None:
    logging.debug(result.info)


class Tracker:

    def __init__(self, ball_bounds: BallConfig, useGPU: bool = False, **kwargs):
        self.kwargs = kwargs
        self.ball_track = Track(maxlen=kwargs.get('buffer'))
        self.off = kwargs.get('off')
        self.verbose = kwargs.get("verbose")
        self.calibration = kwargs.get("calibration")
        [self.proc, self.iproc] = generate_processor_switches(useGPU)
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.ball_bounds = ball_bounds
        self.ball_calibration = self.calibration == "ball"
        self.calibration_bounds = lambda: self.ball_bounds if self.ball_calibration else None
        self.bounds_in = pl.process.IterableQueue() if self.ball_calibration else None
        self.calibration_out = pl.process.IterableQueue() if self.ball_calibration else None

    def stop(self) -> None:
        if self.ball_calibration:
            self.bounds_in.stop()
            self.calibration_out.stop()

    def update_ball_track(self, detected_ball: Blob) -> Track:
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def get_info(self, ball_track: Track) -> Info:
        info = [
            ("Track length", f"{sum([1 for p in ball_track or [] if p is not None])}"),
            ("Calibration", f"{self.calibration if self.calibration is not None else 'off'}"),
            ("Tracker", f"{'off' if self.off else 'on'}")
        ]
        if self.ball_calibration:
            [lower_rgb, upper_rgb] = self.calibration_bounds().bounds("rgb")
            info.append((f"lower", f'({",".join(map(str,lower_rgb))})'))
            info.append((f"upper", f'({",".join(map(str,upper_rgb))})'))
            info.append((f"invert frame", f'{self.calibration_bounds().invert_frame}'))
            info.append((f"invert mask", f'{self.calibration_bounds().invert_mask}'))
        return info

    @property
    def calibration_output(self) -> pl.process.IterableQueue:
        return self.calibration_out

    def config_input(self, config: BallConfig) -> None:
        if self.ball_calibration:
            self.bounds_in.put_nowait(config)

    def track(self, preprocess_result: PreprocessResult) -> TrackResult:
        ball = None
        goals = preprocess_result.goals
        ball_track = None
        info = []
        try:
            if not self.off:
                if self.ball_calibration:
                    try:
                        self.ball_bounds = self.bounds_in.get_nowait()
                    except Empty:
                        pass
                f = self.proc(preprocess_result.preprocessed if preprocess_result.preprocessed is not None else preprocess_result.original)
                # TODO: research this opencl T-API call for moving things into Shared Virtual Memory f = cv2.UMat(f)
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
                ball_track = self.update_ball_track(ball)
            info = preprocess_result.info + self.get_info(ball_track)
        except Exception as e:
            logging.error(f"Error in track {e}")
            traceback.print_exc()
        if not self.verbose:
            return TrackResult(preprocess_result.original, goals, ball_track, ball, info)
        else:
            return TrackResult(preprocess_result.preprocessed, goals, ball_track, ball, info)
