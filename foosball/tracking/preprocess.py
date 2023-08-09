from enum import Enum

import cv2
import numpy as np

from ..arUcos import calibration, Aruco
from ..tracking.models import Frame, PreprocessResult

TEXT_SCALE = 0.8
TEXT_COLOR = (0, 255, 0)


class WarpMode(Enum):
    WARP = 1
    DEWARP = 2


class PreProcessor:
    def __init__(self, headless=True, mask=None, aruco_dictionary=cv2.aruco.DICT_4X4_1000,
                 aruco_params=cv2.aruco.DetectorParameters(), **kwargs):
        self.headless = headless
        self.mask = mask
        self.kwargs = kwargs
        self.detector, _ = calibration.init_aruco_detector(aruco_dictionary, aruco_params)
        # self.out = pl.process.IterableQueue()

    def stop(self) -> None:
        # self.out.stop()
        pass

    def detect_markers(self, frame: Frame) -> list[Aruco]:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return calibration.detect_markers(img_gray, self.detector)

    def mask_frame(self, frame: Frame) -> Frame:
        """
        legacy
        """
        return frame if self.mask is None else cv2.bitwise_and(frame, frame, mask=self.mask)

    def process(self, frame: Frame) -> PreprocessResult:
        # detect markers
        markers = self.detect_markers(frame)
        homography_matrix = None
        # check if there are exactly 4 markers present
        info = [(f'{"!" if len(markers) != 4 else ""}Corners', f'{len(markers)}')]
        if len(markers) == 4:
            # crop and warp
            preprocessed, homography_matrix = self.four_point_transform(frame, markers)
        else:
            preprocessed = self.mask_frame(frame)

        return PreprocessResult(frame, preprocessed, homography_matrix, info)

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
        print(src_pts)
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


def generate_projection(homography_matrix, mode: WarpMode):
    H = homography_matrix if mode == WarpMode.WARP else np.linalg.inv(homography_matrix)

    def _warp_func(src_pt):
        src = np.array([src_pt[0], src_pt[1], 1]) # add a dimension to convert into homogenous coordinates
        src = src.reshape(3, 1) # reshape to line vector for proper matrix multiplication
        dest = np.dot(H, src)  # warp point (with potential perspective projection)
        dest = dest / dest[2]  # project back onto cartesian coordinates
        return int(dest[0]), int(dest[1])
    return _warp_func
