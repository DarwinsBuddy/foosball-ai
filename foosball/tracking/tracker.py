import logging
import traceback
from multiprocessing import Queue
from queue import Empty

from const import CalibrationMode
from .preprocess import WarpMode, project_blob
from ..detectors.color import BallColorDetector, BallColorConfig
from ..models import TrackResult, Track, Info, Blob, Goals, InfoLog, Verbosity
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
        self.camera_matrix = Calibration().load().camera_matrix
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

    def update_ball_tracks(self, detected_ball: Blob, ball3d: Point3d) -> Track:
        # TODO: refactor this
        self.ball_track_3d.appendleft(ball3d)
        if len(self.ball_track_3d) > 1 and self.ball_track_3d[-2] is not None and self.ball_track_3d[-1] is not None:
            # TODO: catch on here
            print("Distance travelled ", self.ball_track_3d[-2] - self.ball_track_3d[-1])
        if detected_ball is not None:
            center = detected_ball.center
            # update the points queue (track history)
            self.ball_track.appendleft(center)
        else:
            self.ball_track.appendleft(None)
        return self.ball_track

    def get_info(self, ball_track: Track) -> InfoLog:
        info = InfoLog(infos=[
            Info(verbosity=Verbosity.DEBUG, title="Track length", value=f"{str(sum([1 for p in ball_track or [] if p is not None])).rjust(2, ' ')}"),
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
        preprocess_result = msg.kwargs['result']
        timestamp = msg.kwargs['time']
        ball = None
        goals = preprocess_result.goals
        ball_track = None
        info = None
        try:
            if not self.off:
                if self.calibration:
                    try:
                        self.ball_detector.config = self.bounds_in.get_nowait()
                    except Empty:
                        pass
                f = self.proc(preprocess_result.preprocessed if preprocess_result.preprocessed is not None else preprocess_result.original)
                ball_detection_result = self.ball_detector.detect(f)
                ball = ball_detection_result.ball
                # if we have some markers detected in preprocess step, we can determine the 3d position of the ball
                ball3d = None
                if ball is not None and preprocess_result.arucos is not None:
                    ball3d = get_3d_position(ball.center, preprocess_result.arucos, self.camera_matrix)
                # do not forget to project detected points onto the original frame on rendering
                if not self.verbose:
                    if ball is not None and preprocess_result.homography_matrix is not None:
                        ball = project_blob(ball, preprocess_result.homography_matrix, WarpMode.DEWARP)
                    if goals is not None and preprocess_result.homography_matrix is not None:
                        goals = Goals(
                            left=project_blob(goals.left, preprocess_result.homography_matrix, WarpMode.DEWARP),
                            right=project_blob(goals.right, preprocess_result.homography_matrix, WarpMode.DEWARP)
                        )
                if self.calibration:
                    self.calibration_out.put_nowait(self.iproc(ball_detection_result.frame))
                    # copy deque, since we otherwise run into odd tracks displayed
                ball_track = self.update_ball_tracks(ball, ball3d).copy()
            info = preprocess_result.info
            info.concat(self.get_info(ball_track))
        except Exception as e:
            logger.error(f"Error in track {e}")
            traceback.print_exc()
        if not self.verbose:
            return Msg(kwargs={"time": timestamp, "result": TrackResult(preprocess_result.original, goals, ball_track, ball, info)})
        else:
            return Msg(kwargs={"time": timestamp, "result": TrackResult(preprocess_result.preprocessed, goals, ball_track, ball, info)})


def get_3d_position(point2d: Point, arucos: list[Aruco], camera_matrix) -> Point3D:
    """
    Calculate the 3D position of the point within the area covered by the ArUco markers
    """
    # TODO: refactor this to be more efficient
    try:
        aruco_marker_size_cm = 5.0  # TODO let this come from somewhere fixed or arguments
        aruco_points = [corners2pt(a.corners) for a in arucos]
        point = np.array([*point2d, 1])
        scale_factors = [aruco_marker_size_cm / np.linalg.norm(t) for t in [a.translation_vector for a in arucos]]
        rvecs = [a.rotation_vector * scale_factors[i] for i, a in enumerate(arucos)]
        tvecs = [a.translation_vector * scale_factors[i] for i, a in enumerate(arucos)]
        if arucos is not None:
            # Define the transformation matrices for the ArUco markers
            transformation_matrices = []
            for i in range(4):
                R, _ = cv2.Rodrigues(rvecs[i])
                T = tvecs[i]
                extrinsic_matrix = np.hstack((R, T.reshape(3, 1)))
                transformation_matrices.append(extrinsic_matrix)

            # Define the 3D positions of the ArUco markers
            marker_positions = []
            for i in range(4):
                homogeneous_marker = np.hstack((aruco_points[i], 1))
                marker_position = np.dot(np.linalg.inv(camera_matrix), homogeneous_marker)
                marker_positions.append(marker_position)

            # Calculate the 3D position of the input point
            point_homogeneous = np.hstack((point, 1))
            estimated_point = np.zeros(3)
            for i in range(4):
                inv_extrinsic = np.linalg.inv(np.vstack((transformation_matrices[i], [0, 0, 0, 1])))
                result = np.dot(inv_extrinsic, point_homogeneous)
                estimated_point += result[:3] * result[3] / marker_positions[i][2]

            estimated_point /= 4  # Average the estimated positions from all 4 markers

            return estimated_point
    except Exception as e:
        logging.error("Dang it!", e)
        traceback.print_exc()
    return None
