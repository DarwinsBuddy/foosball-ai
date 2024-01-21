from enum import Enum

CALIBRATION_MODE = "calibrationMode"
CALIBRATION_IMG_PATH = "calibrationImagePath"
CALIBRATION_VIDEO = "calibrationVideo"
CALIBRATION_SAMPLE_SIZE = "calibrationSampleSize"
ARUCO_BOARD = "arucoBoard"
FILE = "file"
CAMERA_ID = "cameraId"
FRAMERATE = "framerate"
OUTPUT = "output"
CAPTURE = "capture"
DISPLAY = "display"
BUFFER = "buffer"
BALL = "ball"
XPAD = "xpad"
YPAD = "ypad"
SCALE = "scale"
VERBOSE = "verbose"
HEADLESS = "headless"
OFF = "off"
MAX_PIPE_SIZE = "maxPipeSize"
INFO_VERBOSITY = "infoVerbosity"
GPU = "gpu"
AUDIO = "audio"
WEBHOOK = "webhook"


class CalibrationMode:
    BALL = "ball"
    GOAL = "goal"
    CAM = "cam"


class BallPresets:
    YAML = "yaml"
    YELLOW = "yellow"
    ORANGE = "orange"
