import argparse

from foosball.ai import process_video
from foosball.capture import Capture

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the (optional) video file")
ap.add_argument("-c", "--calibration", action='store_true', help="Calibration mode")
ap.add_argument("-v", "--verbose", action='store_true', help="Verbose")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())


if __name__ == '__main__':
    # if args.get('calibration') and args.get('file'):
    #     calibration(args, cap=Capture(args.get('file')))
    # elif args.get('file'):
    #     process_video(args, cap=Capture(args.get('file')))
    if args.get('file'):
        process_video(args, cap=Capture(args.get('file')))
    else:
        print(ap.format_help())
