"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
"""
import glob
import logging
import os.path
import time
from random import sample
from typing import Optional, List

import cv2
from cv2 import aruco
import numpy as np
import yaml
from tqdm import tqdm

from .models import Aruco

Frame = np.array
DIST_COEFF_PATH = 'dist_coeff.npy'
CAMERA_MATRIX_PATH = 'camera_matrix.npy'
CALIBRATION_YAML = 'calibration.yaml'


class Calibration:
    camera_matrix: np.array = None
    dist_coefficients: np.array = None
    _image_markers: List[List[Aruco]] = []

    def __init__(self, board: aruco.GridBoard, camera_matrix: np.array = None, dist_coefficients: np.array = None):
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coefficients
        self.board = board

    def get_sample(self, sample_size: int):
        image_markers_sorted_by_detected_sum = sorted(self._image_markers, key=lambda ms: len(ms))
        if len(image_markers_sorted_by_detected_sum) > 0:
            maximum = len(image_markers_sorted_by_detected_sum[-1])
            filtered_images = [markers for markers in image_markers_sorted_by_detected_sum if len(markers) == maximum]
            return filtered_images if len(filtered_images) < sample_size else sample(filtered_images, sample_size)
        else:
            return self._image_markers

    def recalibrate(self, shape: np.array, sample_size: Optional[int] = None) -> bool:
        counter, corners_list, id_list = [], [], []
        # if sample size is set only take a sample of all of those detected images' markers
        filtered_img_markers = self._image_markers if sample_size is None else self.get_sample(sample_size)
        for markers in filtered_img_markers:
            corners = np.array([m.corners for m in markers])
            ids = np.array([[m.id] for m in markers])
            print(corners_list, corners.shape)
            corners_list = corners if len(corners_list) == 0 else np.vstack((corners_list, corners))
            id_list = ids if len(id_list) == 0 else np.vstack((id_list, ids))
            counter.append(len(markers))
        logging.debug("Calibrating camera ....")
        if len(corners_list) > 0:
            try:
                ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, np.array(counter),
                                                                          self.board, shape, None, None)
                self.camera_matrix = mtx
                self.dist_coefficients = dist
                return True
            except cv2.error:
                logging.error("Could not calibrate camera. Exiting...")
        else:
            logging.error("No arUcos detected in any image. Exiting...")
        return False

    def add_image_markers(self, arucos: List[Aruco]):
        if arucos is not None and len(arucos) > 0:
            self._image_markers.append(arucos)

    def store(self):
        print(f"Camera matrix stored in {CAMERA_MATRIX_PATH}")
        print(self.camera_matrix)
        np.save(CAMERA_MATRIX_PATH, self.camera_matrix)
        print(f"Distortion coefficients in {DIST_COEFF_PATH}")
        print(self.dist_coefficients)
        np.save(DIST_COEFF_PATH, self.dist_coefficients)
        print(f"Both stored in {CALIBRATION_YAML}")
        data = {'camera_matrix': np.asarray(self.camera_matrix).tolist(),
                'dist_coeff': np.asarray(self.dist_coefficients).tolist()}
        with open(CALIBRATION_YAML, "w") as f:
            yaml.dump(data, f)
        return self

    def load(self):
        if os.path.exists(CALIBRATION_YAML):
            with open(CALIBRATION_YAML, 'r') as f:
                loaded_dict = yaml.unsafe_load(f)
            mtx = loaded_dict.get('camera_matrix')
            dist = loaded_dict.get('dist_coeff')
            self.camera_matrix = np.array(mtx)
            self.dist_coefficients = np.array(dist)
        elif os.path.exists(CAMERA_MATRIX_PATH) and os.path.exists(DIST_COEFF_PATH):
            self.camera_matrix = np.load(CAMERA_MATRIX_PATH)
            self.dist_coefficients = np.load(DIST_COEFF_PATH)
        else:
            raise Exception(
                f"Couldn't find calibration data. Either {CALIBRATION_YAML} or both {CAMERA_MATRIX_PATH} and {DIST_COEFF_PATH}")
        return self

    @property
    def as_string(self):
        return f"camera_matrix={self.camera_matrix}\ndist_coeff={self.dist_coefficients}\n# marker_images={len(self._image_markers)}"


def draw_markers(img: Frame, arucos: list[Aruco], calib: Calibration = None) -> Frame:
    if len(arucos) > 0:
        corners = np.array([a.corners for a in arucos])
        ids = np.array([a.id for a in arucos])
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
        if calib is not None:
            for a in arucos:
                if a.rotation_vector is not None and a.translation_vector is not None:
                    img = cv2.drawFrameAxes(img, calib.camera_matrix, calib.dist_coefficients,
                                            a.rotation_vector,
                                            a.translation_vector,
                                            length=.005, thickness=1)
    return img


def detect_markers(image, detector: cv2.aruco.ArucoDetector) -> list[Aruco]:
    corners, ids, rejected_img_points = detector.detectMarkers(image)
    # if rejected_img_points is not None:
    #     logging.debug(f"Marker detection rejected {len(rejected_img_points)}")
    if ids is not None:
        return [Aruco(np.array(i[0]), c) for i, c in list(zip(ids, corners))]
    else:
        return []


def estimate_markers_poses(arucos: List[Aruco], calib: Calibration):
    corners = np.array([a.corners for a in arucos])
    # Estimate pose of each marker and return the values rotation_vector and translation_vector
    # (different from those of camera coefficients)
    rotation_vectors, translation_vectors, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02,
                                                                                               calib.camera_matrix,
                                                                                               calib.dist_coefficients)
    if rotation_vectors is not None and translation_vectors is not None:
        for i in range(0, len(arucos)):
            arucos[i].rotation_vector = np.array([rotation_vectors[i]])
            arucos[i].translation_vector = np.array([translation_vectors[i]])
    return arucos


def init_aruco_detector(aruco_dictionary, aruco_params):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector, aruco_dict


def calibrate_camera(camera_id=None, calibration_video_path=None, calibration_images_path=None, headless=False,
                     aruco_dictionary=cv2.aruco.DICT_4X4_1000, marker_length_cm=5.0, marker_separation_cm=1.0,
                     aruco_params=cv2.aruco.DetectorParameters(), recording_time=5, sample_size=None):
    print("CAMERA: ", camera_id)
    print("images: ", calibration_images_path)
    # For validating results, show aruco board to camera.
    detector, aruco_dict = init_aruco_detector(aruco_dictionary=aruco_dictionary, aruco_params=aruco_params)
    board = cv2.aruco.GridBoard((4, 5), marker_length_cm, marker_separation_cm, aruco_dict)
    board_img = cv2.aruco.drawPlanarBoard(board, (864, 1080), marginSize=0, borderBits=1)
    if not headless:
        cv2.imshow("board", board_img)
    shape = None
    # if calibration_images_path calibrate with images
    if calibration_images_path is not None:
        path = os.path.abspath(calibration_images_path)
        img_list = []
        exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        fns = [f for ext in exts for f in glob.glob(os.path.join(path, ext))]
        for idx, fn in enumerate(fns):
            img = cv2.imread(str(os.path.join(path, fn)))
            img_list.append(img)
            # h, w, c = img.shape
        logging.debug(f'Calibrating {len(img_list)} images')
        calib = Calibration(board)
        for idx, im in enumerate(tqdm(img_list)):
            logging.debug(f"Calibrating {idx} {fns[idx]}")
            img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            shape = img_gray.shape
            if not headless:
                cv2.imshow("Calibration", img_gray)
                cv2.waitKey(1)
            arucos = detect_markers(img_gray, detector)
            calib.add_image_markers(arucos)
        if calib.recalibrate(shape, sample_size):
            calib.store()
    # if camera_id given calibrate with live footage
    elif calibration_video_path is not None or camera_id is not None:
        calib = Calibration(board).load()
        camera = cv2.VideoCapture(calibration_video_path if calibration_video_path is not None else camera_id)
        start = time.time()
        ret, img = camera.read()
        while (camera_id is None or (time.time() - start) < recording_time) and ret:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            shape = img_gray.shape
            arucos = detect_markers(img_gray, detector)
            if not headless:
                arucos = estimate_markers_poses(arucos, calib)
                img = draw_markers(img, arucos, calib)
            calib.add_image_markers(arucos)
            if not headless:
                cv2.imshow("Calibration", img)
                cv2.waitKey(1)
            ret, img = camera.read()
        if calib.recalibrate(shape, sample_size):
            calib.store()
            logging.debug(calib.as_string)
    else:
        logging.error("Please specify calibration options. Neither image path nor camera_id specified")

    cv2.destroyAllWindows()
