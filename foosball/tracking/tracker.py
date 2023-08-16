import logging
from queue import Empty

import pypeln as pl

from .colordetection import detect_ball
from .models import TrackResult, Track, BallConfig, Info, Blob, PreprocessResult
from .preprocess import generate_projection, WarpMode


def log(result: TrackResult) -> None:
    logging.debug(result.info)


class Tracker:

    def __init__(self, ball_bounds: BallConfig, off=False, verbose=False, **kwargs):
        self.kwargs = kwargs
        self.ball_track = Track(maxlen=kwargs.get('buffer'))
        self.off = off
        self.verbose = verbose
        self.calibration = kwargs.get("calibration")
        # define the lower_ball and upper_ball boundaries of the
        # ball in the HSV color space, then initialize the
        self.ball_bounds = ball_bounds
        self.ball_calibration = self.calibration == "ball"
        if self.calibration is not None:
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
            ("Track length", f"{sum([1 for p in ball_track if p is not None])}"),
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

    def track(self, preprocess_result: PreprocessResult, debug=False) -> TrackResult:
        ball = None
        ball_track = None
        if not self.off:
            if self.ball_calibration:
                try:
                    self.ball_bounds = self.bounds_in.get_nowait()
                except Empty:
                    pass
            f = preprocess_result.preprocessed if preprocess_result.preprocessed is not None else preprocess_result.original
            ball_detection_result = detect_ball(f, self.ball_bounds)
            ball = ball_detection_result.ball

            # do not forget to project detected points onto the original frame on rendering
            if not debug:
                if ball is not None and preprocess_result.homography_matrix is not None:
                    dewarp = generate_projection(preprocess_result.homography_matrix, WarpMode.DEWARP)
                    x0, y0 = dewarp((ball.bbox[0], ball.bbox[1]))
                    x1, y1 = dewarp((ball.bbox[0] + ball.bbox[2], ball.bbox[1] + ball.bbox[3]))
                    ball = Blob(dewarp(ball.center), (x0, y0, x1-x0, y1-y0))
            if self.ball_calibration:
                self.calibration_out.put_nowait(ball_detection_result.frame)
            ball_track = self.update_ball_track(ball)
        info = preprocess_result.info + self.get_info(ball_track)
        if not debug:
            return TrackResult(preprocess_result.original, ball_track, ball, info)
        else:
            return TrackResult(preprocess_result.preprocessed, ball_track, ball, info)
