import argparse
import json
import logging.config
import os
import signal
from itertools import chain

from foosball.arUcos.calibration import print_aruco_board, calibrate_camera
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
    calibration.add_argument("-c", "--calibrationMode", choices=['ball', 'goal', 'cam'], help="Calibration mode")
    calibration.add_argument("-ci", "--calibrationImagePath", type=dir_path, default=None,
                             help="Images path for calibration mode. If not given switching to live calibration")
    calibration.add_argument("-cv", "--calibrationVideo", type=file_path, default=None,
                             help="Path to video file for calibration mode. If not given switching to live calibration")
    calibration.add_argument("-cs", "--calibrationSampleSize", type=int, default=50,
                             help="Sample size for calibration mode. If not given all detected image markers will be taken into account")
    calibration.add_argument("-a", "--arucoBoard", action='store_true', help="Output aruco board as arucos.pdf")

    io = ap.add_argument_group(title="Input/Output", description="Options for input and output streams")
    io.add_argument("-f", "--file", help="path to the (optional) video file")
    io.add_argument("-cam", "--cameraId", type=int, default=None, help="Camera id to be used")
    io.add_argument("-fps", "--framerate", type=int, help="Framerate for camera src", default=60)
    io.add_argument("-out", "--output", default=None, help="path to store (optional) a rendered video")
    io.add_argument("-cap", "--capture", choices=['cv', 'gear'], default='gear', help="capture backend")
    io.add_argument("-d", "--display", choices=['cv', 'gear'], default='cv', help="display backend cv=direct display, gear=stream")

    tracker = ap.add_argument_group(title="Tracker", description="Options for the ball/goal tracker")
    tracker.add_argument("-ba", "--ball", choices=['yaml', 'orange', 'yellow'], default='yaml',
                    help="Pre-configured ball color bounds. If 'yaml' is selected, a file called 'ball.yaml' "
                         "(stored by hitting 's' in ball calibration mode) will be loaded as a preset."
                         "If no file present fallback to 'yellow'")
    tracker.add_argument("-b", "--buffer", type=int, default=16, help="max track buffer size")

    preprocess = ap.add_argument_group(title="Preprocessor", description="Options for the preprocessing step")
    preprocess.add_argument("-xp", "--xpad", type=int, default=50,
                    help="Horizontal padding applied to ROI detected by aruco markers")
    preprocess.add_argument("-yp", "--ypad", type=int, default=20,
                    help="Vertical padding applied to ROI detected by aruco markers")
    preprocess.add_argument("-s", "--scale", type=float, default=0.4, help="Scale stream")

    general = ap.add_argument_group(title="General", description="General options")
    general.add_argument("-v", "--verbose", action='store_true', help="Verbose")
    general.add_argument("-q", "--headless", action='store_true', help="Disable visualizations")
    general.add_argument("-o", "--off", action='store_true', help="Disable ai")
    general.add_argument("-p", "--maxPipeSize", type=int, default=128, help="max pipe buffer size")
    general.add_argument("-i", "--info_verbosity", type=int, help="Verbosity level of gui info box (default: None)", default=None)
    general.add_argument("-g", "--gpu", choices=['preprocess', 'tracker', 'render'], nargs='+', default=["render"], help="use GPU")
    general.add_argument("-A", "--audio", action='store_true', help="Enable audio")
    general.add_argument("-W", "--webhook", action='store_true', help="Enable webhook")
    return ap


def usage_and_exit():
    ap = argparse.ArgumentParser()
    print(ap.format_help())
    return 1


def main(kwargs):
    logging.debug(kwargs)
    dis = None
    if kwargs.get('arucoBoard'):
        print_aruco_board()
        logging.info("Aruco board printed")
    elif kwargs.get('calibrationMode') == 'cam':
        calibrate_camera(camera_id=kwargs.get('cameraId'), calibration_video_path=kwargs.get('calibrationVideo'),
                         calibration_images_path=kwargs.get('calibrationImagePath'), headless=False,
                         sample_size=kwargs.get('calibrationSampleSize'))
    elif kwargs.get('file') or kwargs.get('cameraId') is not None:
        if not kwargs.get('headless'):
            if kwargs.get('display') == 'cv':
                from .sink.opencv import DisplaySink

                dis = DisplaySink()
            elif kwargs.get('display') == 'gear':
                from .sink.gear import StreamSink
                dis = StreamSink()
            else:
                return usage_and_exit()

        source = kwargs.get('file') or kwargs.get('cameraId')
        if kwargs.get('capture') == 'gear':
            from .source.gear import GearSource
            cap = GearSource(source, framerate=kwargs.get('framerate'), resolution=(1280, 720))
        elif kwargs.get('capture') == 'cv':
            from .source.opencv import OpenCVSource
            cap = OpenCVSource(source)
        else:
            return usage_and_exit()
        ai = AI(cap, dis, **kwargs)

        def signal_handler(sig, frame):
            ai.stop()

        signal.signal(signal.SIGINT, signal_handler)
        ai.process_video()
    else:
        return usage_and_exit()
