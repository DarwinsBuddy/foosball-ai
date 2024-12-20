import traceback
from enum import Enum
from multiprocessing import Queue
from queue import Empty

import cv2
import numpy as np

from const import CalibrationMode, OFF
from .models import PreprocessorResult
from ...arUcos import calibration
from ...arUcos.models import Aruco
from ...detectors.color import GoalDetector, GoalColorConfig
from ...models import Frame, Point, Rect, Blob, Goals, FrameDimensions, ScaleDirection, \
    Info, Verbosity
from ...pipe.BaseProcess import BaseProcess, Msg
from ...pipe.Pipe import clear
from ...utils import ensure_cpu, generate_processor_switches, relative_change, scale

TEXT_SCALE = 0.8
TEXT_COLOR = (0, 255, 0)


class WarpMode(Enum):
    WARP = 1
    DEWARP = 2


def pad_pt(pt: Point, xpad, ypad) -> Point:
    return [max(0, pt[0] + xpad), max(0, pt[1] + ypad)]


def pad_rect(rectangle: Rect, xpad: int, ypad: int) -> Rect:
    (tl, tr, br, bl) = rectangle
    return (
        pad_pt(tl, -xpad, -ypad),
        pad_pt(tr, xpad, -ypad),
        pad_pt(br, xpad, ypad),
        pad_pt(bl, -xpad, ypad)
    )


class PreProcessor(BaseProcess):
    def __init__(self, dims: FrameDimensions, goal_detector: GoalDetector, headless=True, mask=None, used_markers=None,
                 redetect_markers_frames: int = 60, aruco_dictionary=cv2.aruco.DICT_4X4_1000,
                 aruco_params=cv2.aruco.DetectorParameters(), xpad: int = 50, ypad: int = 20,
                 goal_change_threshold: float = 0.10, useGPU: bool = False, calibrationMode=None, verbose=False, **kwargs):
        super().__init__(name="Preprocessor")
        self.dims = dims
        self.goal_change_threshold = goal_change_threshold
        self.redetect_markers_frame_threshold = redetect_markers_frames
        if used_markers is None:
            used_markers = [0, 1, 2, 3, 4]
        self.used_markers = used_markers
        self.headless = headless
        self.mask = mask
        self.xpad = xpad
        self.ypad = ypad
        [self.proc, self.iproc] = generate_processor_switches(useGPU)
        self.detector, _ = calibration.init_aruco_detector(aruco_dictionary, aruco_params)
        self.markers = []
        self.homography_matrix = None
        self.frames_since_last_marker_detection = 0
        self.goals = None
        self.calibrationMode = calibrationMode
        self.verbose = verbose
        self.goals_calibration = self.calibrationMode == CalibrationMode.GOAL
        self.calibration_out = Queue() if self.goals_calibration else None
        self.config_in = Queue() if self.goals_calibration else None
        self.goal_detector = goal_detector

    def config_input(self, config: GoalColorConfig) -> None:
        if self.goals_calibration:
            self.config_in.put_nowait(config)

    def close(self) -> None:
        if self.goals_calibration:
            clear(self.config_in)
            self.config_in.close()
            clear(self.calibration_out)
            self.calibration_out.close()

    def detect_markers(self, frame: Frame) -> list[Aruco]:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return calibration.detect_markers(img_gray, self.detector)

    def mask_frame(self, frame: Frame) -> Frame:
        """
        legacy
        """
        return frame if self.mask is None else cv2.bitwise_and(frame, frame, mask=self.mask)

    def process(self, msg: Msg) -> Msg:
        frame = msg.data['frame']
        frame = self.proc(frame)
        frame = scale(frame, self.dims, ScaleDirection.DOWN)
        preprocessed = frame
        info: [Info] = []
        try:
            if self.goals_calibration:
                try:
                    self.goal_detector.config = self.config_in.get_nowait()
                except Empty:
                    pass

            trigger_marker_detection = self.frames_since_last_marker_detection == 0 or len(self.markers) == 0
            goal_info = None
            marker_info = Info(verbosity=Verbosity.TRACE, title=f'{"? " if trigger_marker_detection else ""}Markers', value=f'{len(self.markers)}'.ljust(10, ' '))
            if not self.kwargs.get(OFF):
                if trigger_marker_detection:
                    # detect markers
                    markers = self.detect_markers(frame)
                    # check if there are exactly 4 markers present
                    markers = [m for m in markers if m.id in self.used_markers]
                    marker_info = Info(verbosity=Verbosity.TRACE, title=f'{"! " if len(markers) != 4 else ""}Markers', value=f'{len(markers)}'.ljust(10, ' '))
                    # logging.debug(f"markers {[list(m.id)[0] for m in markers]}")
                    if len(markers) == 4:
                        self.markers = markers

                self.frames_since_last_marker_detection = (self.frames_since_last_marker_detection + 1) % self.redetect_markers_frame_threshold
                if len(self.markers) == 4:
                    # crop and warp
                    preprocessed, self.homography_matrix = self.four_point_transform(frame, self.markers)
                    if trigger_marker_detection:
                        # detect goals anew
                        goals_detection_result = self.goal_detector.detect(preprocessed)
                        if self.goals_calibration:
                            self.calibration_out.put_nowait(ensure_cpu(goals_detection_result.frame))
                        # check if goals are not significantly smaller than before
                        new_goals = goals_detection_result.goals
                        if self.goals is None:
                            self.goals = new_goals
                        elif new_goals is not None:
                            left_change = relative_change(self.goals.left.area(), new_goals.left.area())
                            right_change = relative_change(self.goals.right.area(), new_goals.right.area())
                            self.goals = Goals(
                                left=self.goals.left if abs(left_change) < self.goal_change_threshold else new_goals.left,
                                right=self.goals.right if abs(right_change) < self.goal_change_threshold else new_goals.right
                            )
                        # TODO: distinguish between red or blue goal (instead of left and right)
                    goal_info = Info(verbosity=Verbosity.DEBUG, title='goals', value=f'{"detected" if self.goals is not None else "fail"}')
                else:
                    goal_info = Info(verbosity=Verbosity.DEBUG, title='goals', value='none'.rjust(8, ' '))
                    preprocessed = self.mask_frame(frame)
            info.append(marker_info)
            if goal_info is not None:
                info.append(goal_info)
        except Exception as e:
            self.logger.error(f"Error in preprocess {e}")
            traceback.print_exc()
        # Not passing original msg due to performance impact (copying whole frames, etc.)
        return Msg(info=info, data={
            "Preprocessor": PreprocessorResult(original=self.iproc(frame), preprocessed=self.iproc(preprocessed), homography_matrix=self.homography_matrix, goals=self.goals)
        })

    @staticmethod
    def corners2pt(corners) -> [int, int]:
        moments = cv2.moments(corners)
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        return [c_x, c_y]

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    def four_point_transform(self, frame: Frame, markers: list[Aruco]) -> tuple[Frame, [int, int]]:
        pts = np.array([self.corners2pt(marker.corners) for marker in markers])
        # obtain a consistent order of the points and unpack them
        # individually
        src_pts = self.order_points(pts)
        # pad
        src_pts = pad_rect(src_pts, self.xpad, self.ypad)
        (tl, tr, br, bl) = src_pts
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        # compute the perspective transform matrix and then apply it
        [width, height] = [max_width, max_height]
        x0, y0 = 0, 0  # that's the zero point (we could also shift the image around)
        dst_pts = np.array([
            [x0, y0],
            [x0 + width - 1, y0],
            [x0 + width - 1, y0 + height - 1],
            [x0, y0 + height - 1]], dtype="float32")
        src_pts = np.array(src_pts, dtype=np.float32)

        # homography_matrix = cv2.getPerspectiveTransform(pts, dst)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        # image = undistort(image, k, d)
        warped = cv2.warpPerspective(frame, homography_matrix, (width + (x0 * 2), height + (y0 * 2)))

        # warped = cv2.warpPerspective(image, M, (width, width))

        # since we want to draw on the original image (later on), which is still originally not warped,
        # we want to have a function set to project from/onto the warped/un-warped version of the frame
        # for future reference. so we return the warped image and the used homography matrix
        return warped, homography_matrix


def project_point(pt: Point, homography_matrix: np.array, mode: WarpMode):
    H = homography_matrix if mode == WarpMode.WARP else np.linalg.inv(homography_matrix)
    src = np.array([pt[0], pt[1], 1])  # add a dimension to convert into homogenous coordinates
    src = src.reshape(3, 1)  # reshape to line vector for proper matrix multiplication
    dest = np.dot(H, src)  # warp point (with potential perspective projection)
    dest = dest / dest[2]  # project back onto cartesian coordinates
    return int(dest[0]), int(dest[1])


def project_blob(blob: Blob, homography_matrix: np.array, mode: WarpMode):
    x0, y0 = project_point((blob.bbox[0], blob.bbox[1]), homography_matrix, mode)
    x1, y1 = project_point((blob.bbox[0] + blob.bbox[2], blob.bbox[1] + blob.bbox[3]), homography_matrix, mode)
    return Blob(project_point(blob.center, homography_matrix, mode), (x0, y0, x1 - x0, y1 - y0))
