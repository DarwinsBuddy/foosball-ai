import argparse
import os
import signal

from foosball.arUcos.calibration import calibrate_camera
from .tracking.ai import AI

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the (optional) video file")
ap.add_argument("-c",  "--calibration", action='store_true', help="Calibration mode")
ap.add_argument("-ci", "--calibrationImagePath", type=dir_path, default=None, help="Image path for calibration mode. If not given switching to live calibration")
ap.add_argument("-cam", "--cameraId", type=int, default=None, help="Camera id to be used")
ap.add_argument("-cc", "--colorCalibration", choices=['ball', 'all'], help="Color calibration mode")
ap.add_argument("-v", "--verbose", action='store_true', help="Verbose")
ap.add_argument("-o", "--off", action='store_true', help="Disable ai")
ap.add_argument("-q", "--headless", action='store_true', help="Disable visualizations")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max track buffer size")
ap.add_argument("-cap", "--capture", choices=['gear', 'imutils'], default='gear', help="capture backend")
ap.add_argument("-d", "--display", choices=['cv', 'gear'], default='cv', help="display backend")
kwargs = vars(ap.parse_args())


def usage_and_exit():
    print(ap.format_help())
    exit(1)


if __name__ == '__main__':
    cap = None
    dis = None
    if kwargs.get('file'):
        if kwargs.get('display') == 'cv':
            from .display.cv import OpenCVDisplay
            dis = OpenCVDisplay()
        elif kwargs.get('display') == 'gear':
            print("[ALPHA] Feature - Streaming not fully supported")
            from .display.gear import StreamDisplay
            dis = StreamDisplay()
        else:
            usage_and_exit()

        if kwargs.get('capture') == 'gear':
            from .capture.gearcapture import GearCapture
            cap = GearCapture(kwargs.get('file'))
        elif kwargs.get('capture') == 'imutils':
            from .capture.filecapture import FileCapture
            cap = FileCapture(kwargs.get('file'))
        else:
            usage_and_exit()
        print(kwargs)
        ai = AI(cap, dis, **kwargs)


        def signal_handler(sig, frame):
            print('\n\nExiting...')
            ai.stop()


        signal.signal(signal.SIGINT, signal_handler)
        ai.process_video()
    elif kwargs.get('calibration'):
        calibrate_camera(camera_id=kwargs.get('cameraId'), calibration_images_path=kwargs.get('calibrationImagePath'), headless=False)
    else:
        usage_and_exit()
