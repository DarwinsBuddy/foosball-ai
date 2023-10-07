import argparse
import os
import signal

from foosball.arUcos.calibration import calibrate_camera, print_aruco_board
from .tracking.ai import AI


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


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the (optional) video file")
ap.add_argument("-a", "--arucoBoard", action='store_true', help="Output aruco board as arucos.pdf")
ap.add_argument("-c", "--calibration", choices=['ball', 'goal', 'cam'], help="Calibration mode")
ap.add_argument("-ci", "--calibrationImagePath", type=dir_path, default=None,
                help="Images path for calibration mode. If not given switching to live calibration")
ap.add_argument("-cv", "--calibrationVideo", type=file_path, default=None,
                help="Path to video file for calibration mode. If not given switching to live calibration")
ap.add_argument("-cs", "--calibrationSampleSize", type=int, default=50,
                help="Sample size for calibration mode. If not given all detected image markers will be taken into "
                     "account")
ap.add_argument("-cam", "--cameraId", type=int, default=None, help="Camera id to be used")
ap.add_argument("-ba", "--ball", choices=['yaml', 'orange', 'yellow'], default='yaml',
                help="Pre-configured ball color bounds. If 'yaml' is selected, a file called 'ball.yaml' (stored by "
                     "hitting 's' in ball calibration mode) will be loaded as a preset. If no file present fallback to "
                     "'yellow'")
ap.add_argument("-v", "--verbose", action='store_true', help="Verbose")
ap.add_argument("-o", "--off", action='store_true', help="Disable ai")
ap.add_argument("-q", "--headless", action='store_true', help="Disable visualizations")
ap.add_argument("-b", "--buffer", type=int, default=16, help="max track buffer size")
ap.add_argument("-xp", "--xpad", type=int, default=50,
                help="Horizontal padding applied to ROI detected by aruco markers")
ap.add_argument("-yp", "--ypad", type=int, default=20,
                help="Vertical padding applied to ROI detected by aruco markers")
ap.add_argument("-s", "--scale", type=float, default=0.4, help="Scale stream")
ap.add_argument("-cap", "--capture", choices=['cv', 'gear'], default='gear', help="capture backend")
ap.add_argument("-d", "--display", choices=['cv', 'gear'], default='cv', help="display backend")
ap.add_argument("-g", "--gpu", choices=['preprocess', 'tracker', 'render'], nargs='+', default=["render"], help="use GPU")
ap.add_argument("-A", "--audio", action='store_true', help="Enable audio")
ap.add_argument("-W", "--webhook", action='store_true', help="Enable webhook")
kwargs = vars(ap.parse_args())


def usage_and_exit():
    print(ap.format_help())
    exit(1)


if __name__ == '__main__':
    cap = None
    dis = None
    if kwargs.get('arucoBoard'):
        print_aruco_board()
        print("Aruco board printed")
    elif kwargs.get('calibration') == 'cam':
        calibrate_camera(camera_id=kwargs.get('cameraId'), calibration_video_path=kwargs.get('calibrationVideo'),
                         calibration_images_path=kwargs.get('calibrationImagePath'), headless=False,
                         sample_size=kwargs.get('calibrationSampleSize'))
    elif kwargs.get('file') or kwargs.get('cameraId') is not None:
        if not kwargs.get('headless'):
            if kwargs.get('display') == 'cv':
                from .display.cv import OpenCVDisplay

                dis = OpenCVDisplay()
            elif kwargs.get('display') == 'gear':
                print("[ALPHA] Feature - Streaming not fully supported")
                from .display.gear import StreamDisplay

                dis = StreamDisplay()
            else:
                usage_and_exit()

        source = kwargs.get('file') or kwargs.get('cameraId')
        if kwargs.get('capture') == 'gear':
            from .capture.GearStream import GearStream
            cap = GearStream(source, framerate=32, resolution=(1280, 720))
        elif kwargs.get('capture') == 'cv':
            from .capture.OpenCVStream import OpenCVStream
            cap = OpenCVStream(source)
        else:
            usage_and_exit()
        print(kwargs)
        ai = AI(cap, dis, **kwargs)


        def signal_handler(sig, frame):
            ai.stop()


        signal.signal(signal.SIGINT, signal_handler)
        ai.process_video()
    else:
        usage_and_exit()
