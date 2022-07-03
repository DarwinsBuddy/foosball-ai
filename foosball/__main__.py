import argparse
import signal

from foosball.display.gear import StreamDisplay
from .ai import AI

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the (optional) video file")
ap.add_argument("-c", "--calibration", choices=['ball', 'all'], help="Calibration mode")
ap.add_argument("-v", "--verbose", action='store_true', help="Verbose")
ap.add_argument("-o", "--off", action='store_true', help="Disable ai")
ap.add_argument("-q", "--headless", action='store_true', help="Disable visualizations")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max track buffer size")
ap.add_argument("-cap", "--capture", choices=['gear', 'imutils'], default='gear', help="capture backend")
ap.add_argument("-d", "--display", choices=['cv', 'gear'], default='cv', help="display backend")
args = vars(ap.parse_args())


def usage_and_exit():
    print(ap.format_help())
    exit(1)


if __name__ == '__main__':
    cap = None
    display = None
    calibration_mode = args.get('calibration') is not None
    if args.get('file'):
        if args.get('display') == 'cv':
            from .display.cv import OpenCVDisplay
            display = OpenCVDisplay()
        elif args.get('display') == 'gear':
            print("[ALPHA] Feature - Streaming not fully supported")
            from .display.cv import OpenCVDisplay
            display = StreamDisplay()
        else:
            usage_and_exit()

        if args.get('capture') == 'gear':
            from .capture.gearcapture import GearCapture
            cap = GearCapture(args.get('file'))
        elif args.get('capture') == 'imutils':
            from .capture.filecapture import FileCapture
            cap = FileCapture(args.get('file'))
        else:
            usage_and_exit()

        ai = AI(args, cap=cap, display=display)

        def signal_handler(sig, frame):
            print('\n\nExiting...')
            ai.stop()

        signal.signal(signal.SIGINT, signal_handler)
        ai.process_video()
    else:
        usage_and_exit()
