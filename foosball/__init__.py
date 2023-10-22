import argparse
import json
import logging.config
import os
import signal

from const import CALIBRATION_MODE, CALIBRATION_IMG_PATH, CALIBRATION_VIDEO, CALIBRATION_SAMPLE_SIZE, ARUCO_BOARD, \
    FILE, CAMERA_ID, FRAMERATE, OUTPUT, CAPTURE, DISPLAY, BALL, XPAD, YPAD, SCALE, VERBOSE, HEADLESS, OFF, \
    MAX_PIPE_SIZE, INFO_VERBOSITY, GPU, AUDIO, WEBHOOK, BUFFER, BallPresets, CalibrationMode
from foosball.arUcos.camera_calibration import print_aruco_board, calibrate_camera
from foosball.tracking.ai import AI

logging.config.fileConfig("logging.ini")


# import multiprocessing
# logger = multiprocessing.log_to_stderr()

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, conf_file_path, option_string=None):
        parser.parse_args(namespace=argparse.Namespace(**json.load(conf_file_path)))

def get_argparse():
    ap = argparse.ArgumentParser(prog="foosball-ai")
    ap.add_argument("-conf", "--config", type=file_path, default=None)

    calibration = ap.add_argument_group(title="Calibration", description="Options for camera or setup calibration")
    calibration.add_argument("-c", f"--{CALIBRATION_MODE}", choices=[CalibrationMode.BALL, CalibrationMode.GOAL, CalibrationMode.CAM], help="Calibration mode")
    calibration.add_argument("-ci", f"--{CALIBRATION_IMG_PATH}", type=dir_path, default=None,
                             help="Images path for calibration mode. If not given switching to live calibration")
    calibration.add_argument("-cv", f"--{CALIBRATION_VIDEO}", type=file_path, default=None,
                             help="Path to video file for calibration mode. If not given switching to live calibration")
    calibration.add_argument("-cs", f"--{CALIBRATION_SAMPLE_SIZE}", type=int, default=50,
                             help="Sample size for calibration mode. If not given all detected image markers will be taken into account")
    calibration.add_argument("-a", f"--{ARUCO_BOARD}", action='store_true', help="Output aruco board as arucos.pdf")

    io = ap.add_argument_group(title="Input/Output", description="Options for input and output streams")
    io.add_argument("-f", f"--{FILE}", help="path to the (optional) video file")
    io.add_argument("-cam", f"--{CAMERA_ID}", type=int, default=None, help="Camera id to be used")
    io.add_argument("-fps", f"--{FRAMERATE}", type=int, help="Framerate for camera src", default=60)
    io.add_argument("-out", f"--{OUTPUT}", default=None, help="path to store (optional) a rendered video")
    io.add_argument("-cap", f"--{CAPTURE}", choices=['cv', 'gear'], default='gear', help="capture backend")
    io.add_argument("-d", f"--{DISPLAY}", choices=['cv', 'gear'], default='cv', help="display backend cv=direct display, gear=stream")

    tracker = ap.add_argument_group(title="Tracker", description="Options for the ball/goal tracker")
    tracker.add_argument("-ba", f"--{BALL}", choices=[BallPresets.YAML, BallPresets.ORANGE, BallPresets.YELLOW], default=BallPresets.YAML,
                    help="Pre-configured ball color bounds. If 'yaml' is selected, a file called 'ball.yaml' "
                         "(stored by hitting 's' in ball calibration mode) will be loaded as a preset."
                         "If no file present fallback to 'yellow'")
    tracker.add_argument("-b", f"--{BUFFER}", type=int, default=16, help="max track buffer size")

    preprocess = ap.add_argument_group(title="Preprocessor", description="Options for the preprocessing step")
    preprocess.add_argument("-dis", "--distance", type=float, default=125.0, help="Distance between ArucoMarker top left (1) and top right (2)")
    preprocess.add_argument("-xp", f"--{XPAD}", type=int, default=50,
                    help="Horizontal padding applied to ROI detected by aruco markers")
    preprocess.add_argument("-yp", f"--{YPAD}", type=int, default=20,
                    help="Vertical padding applied to ROI detected by aruco markers")
    preprocess.add_argument("-s", f"--{SCALE}", type=float, default=0.4, help="Scale stream")

    general = ap.add_argument_group(title="General", description="General options")
    general.add_argument("-v", f"--{VERBOSE}", action='store_true', help="Verbose")
    general.add_argument("-q", f"--{HEADLESS}", action='store_true', help="Disable visualizations")
    general.add_argument("-o", f"--{OFF}", action='store_true', help="Disable ai")
    general.add_argument("-p", f"--{MAX_PIPE_SIZE}", type=int, default=128, help="max pipe buffer size")
    general.add_argument("-i", f"--{INFO_VERBOSITY}", type=int, help="Verbosity level of gui info box (default: None)", default=None)
    general.add_argument("-g", f"--{GPU}", choices=['preprocess', 'tracker', 'render'], nargs='+', default=["render"], help="use GPU")
    general.add_argument("-A", f"--{AUDIO}", action='store_true', help="Enable audio")
    general.add_argument("-W", f"--{WEBHOOK}", action='store_true', help="Enable webhook")
    return ap


def usage_and_exit():
    ap = argparse.ArgumentParser()
    print(ap.format_help())
    return 1


def main(kwargs):
    logging.debug(kwargs)
    dis = None
    if kwargs.get(ARUCO_BOARD):
        print_aruco_board()
        logging.info("Aruco board printed")
    elif kwargs.get(CALIBRATION_MODE) == 'cam':
        calibrate_camera(camera_id=kwargs.get(CAMERA_ID), calibration_video_path=kwargs.get(CALIBRATION_VIDEO),
                         calibration_images_path=kwargs.get(CALIBRATION_IMG_PATH), headless=kwargs.get(HEADLESS),
                         sample_size=kwargs.get(CALIBRATION_SAMPLE_SIZE))
    elif kwargs.get(FILE) or kwargs.get(CAMERA_ID) is not None:
        if not kwargs.get(HEADLESS):
            match kwargs.get(DISPLAY):
                case 'cv':
                    from .sink.opencv import DisplaySink
                    dis = DisplaySink()
                case 'gear':
                    from .sink.gear import StreamSink
                    dis = StreamSink()
                case _:
                    return usage_and_exit()

        source = kwargs.get(FILE) or kwargs.get(CAMERA_ID)
        match kwargs.get(CAPTURE):
            case 'gear':
                from .source.gear import GearSource
                cap = GearSource(source, framerate=kwargs.get(FRAMERATE), resolution=(1280, 720))
            case 'cv':
                from .source.opencv import OpenCVSource
                cap = OpenCVSource(source)
            case _:
                return usage_and_exit()
        ai = AI(cap, dis, **kwargs)

        def signal_handler(sig, frame):
            ai.stop()

        signal.signal(signal.SIGINT, signal_handler)
        ai.process_video()
    else:
        return usage_and_exit()
