import logging
import traceback
from multiprocessing import Queue
from queue import Empty

import cv2
import numpy as np

from const import CalibrationMode
from cv2.typing import Point3d

from .preprocess import WarpMode, project_blob, PositionEstimationInputs
from ..arUcos.camera_calibration import CameraCalibration
from ..detectors.color import BallColorDetector, BallColorConfig
from ..models import TrackResult, Track, Info, Blob, Goals, InfoLog, Verbosity, Point, Point3D
from ..pipe.BaseProcess import BaseProcess, Msg
from ..pipe.Pipe import clear
from ..utils import generate_processor_switches

logger = logging.getLogger(__name__)


def log(result: TrackResult) -> None:
    logger.debug(result.info)


class Tracker(BaseProcess):

    def __init__(self, ball_detector: BallColorDetector, useGPU: bool = False, buffer=16, off=False, verbose=False,
                 calibrationMode=None, **kwargs):
        super().__init__(name="Tracker")
        self.ball_track = Track(maxlen=buffer)
        self.ball_track_3d = Track(maxlen=buffer)
        self.off = off
        self.verbose = verbose
        self.calibrationMode = calibrationMode
        self.camera_matrix = CameraCalibration().load().camera_matrix
        self.last_timestamp = None
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

    def update_3d_ball_track(self, ball3d: Point3d) -> Track:
        self.ball_track_3d.appendleft(ball3d)
        return self.ball_track_3d

    def update_2d_ball_track(self, detected_ball: Blob) -> Track:
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def calc_speed(self, timestamp) -> float | None:
        v_mps = None
        if len(self.ball_track_3d) > 1 and self.ball_track_3d[-2] is not None and self.ball_track_3d[-1] is not None:
            last_position = self.ball_track_3d[-2]
            current_position = self.ball_track_3d[-1]
            distance_cm = cv2.norm(last_position - current_position, cv2.NORM_L2)  # TODO check if its really cm
            if distance_cm > 100:
                print("DISTANCE: ", distance_cm)
                print(f"POINT A: 3D= {self.ball_track_3d[-2]}    2D={self.ball_track[-2]}")
                print(f"POINT B: 3D= {self.ball_track_3d[-1]}    2D={self.ball_track[-1]}")
            if self.last_timestamp is not None:
                elapsed_time_ms = 0.000001 * (timestamp - self.last_timestamp)
                v_mps = (distance_cm * 10 / elapsed_time_ms)
        return v_mps

    def get_info(self, v_mps: float) -> InfoLog:
        info = InfoLog(infos=[
            Info(verbosity=Verbosity.INFO, title="Speed", value=f"{f'{v_mps:.2f}'.rjust(6) if v_mps is not None else '-'.rjust(6)} m/sec"),
            Info(verbosity=Verbosity.DEBUG, title="Track length", value=f"{str(sum([1 for p in self.ball_track or [] if p is not None])).rjust(2, ' ')}"),
            Info(verbosity=Verbosity.TRACE, title="Calibration", value=f"{self.calibrationMode if self.calibrationMode is not None else 'off'}"),
            Info(verbosity=Verbosity.TRACE, title="Tracker", value=f"{'off' if self.off else 'on'}")
        ])
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
        timestamp = msg.kwargs['time']
        preprocessed = msg.kwargs['preprocessed']
        original = msg.kwargs['original']
        arucos = msg.kwargs['arucos']
        homography_matrix = msg.kwargs['homography_matrix']
        position_estimation_inputs = msg.kwargs['positionEstimationInputs']
        ball = None
        goals = msg.kwargs['goals']
        ball_track = None
        info = InfoLog(infos=[])
        speed = None
        try:
            if not self.off:
                if self.calibration:
                    try:
                        self.ball_detector.config = self.bounds_in.get_nowait()
                    except Empty:
                        pass
                f = self.proc(preprocessed if preprocessed is not None else original)
                ball_detection_result = self.ball_detector.detect(f)
                ball = ball_detection_result.ball
                # if we have some markers detected in preprocess step, we can determine the 3d position of the ball
                ball3d = None
                if ball is not None and arucos is not None:
                    ball3d = self.get_3d_position(ball.center, position_estimation_inputs)
                # do not forget to project detected points onto the original frame on rendering
                if not self.verbose:
                    if ball is not None and homography_matrix is not None:
                        ball = project_blob(ball, homography_matrix, WarpMode.DEWARP)
                    if goals is not None and homography_matrix is not None:
                        goals = Goals(
                            left=project_blob(goals.left, homography_matrix, WarpMode.DEWARP),
                            right=project_blob(goals.right, homography_matrix, WarpMode.DEWARP)
                        )
                if self.calibration:
                    self.calibration_out.put_nowait(self.iproc(ball_detection_result.frame))
                    # copy deque, since we otherwise run into odd tracks displayed
                self.update_3d_ball_track(ball3d)
                ball_track = self.update_2d_ball_track(ball).copy()
            speed = self.calc_speed(timestamp)
            info.concat(self.get_info(speed))
            self.last_timestamp = timestamp
        except Exception as e:
            logger.error(f"Error in track {e}")
            traceback.print_exc()
        if not self.verbose:
            return Msg(kwargs={**msg.kwargs,
                               "time": timestamp,
                               "result": TrackResult(original, goals, ball_track, ball, info),
                               "speed": speed
                               })
        else:
            return Msg(kwargs={**msg.kwargs,
                               "time": timestamp,
                               "result": TrackResult(preprocessed, goals, ball_track, ball, info),
                               "speed": speed
                               })

    @staticmethod
    def get_3d_position(point2d: Point, position_estimation_inputs: PositionEstimationInputs) -> Point3D:
        """
        Calculate the 3D position of the point within the area covered by the ArUco markers
        """
        if position_estimation_inputs is not None:
            point = np.array([*point2d, 1])
            # Calculate the 3D position of the input point
            point_homogeneous = np.hstack((point, 1))
            estimated_point = np.zeros(3)
            for i in range(len(position_estimation_inputs.transformation_matrices)):
                inv_extrinsic = np.linalg.inv(
                    np.vstack((position_estimation_inputs.transformation_matrices[i], [0, 0, 0, 1])))
                result = np.dot(inv_extrinsic, point_homogeneous)
                estimated_point += result[:3] * result[3] / position_estimation_inputs.marker_positions_3d[i][2]

            estimated_point /= 4  # Average the estimated positions from all 4 markers

            return estimated_point
        return None
