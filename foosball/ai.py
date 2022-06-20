# TODO: #1 solve fps issue (we are not capturing high speed) maybe buffer it?

import cv2

from .tracker import Tracker
from .utils import destroy_all_windows, show


def process_video(args, cap):
    calibration_mode = args.get('calibration') is not None
    goals_calibration = args.get('calibration') in ['all', 'goals']
    ball_calibration = args.get('calibration') in ['all', 'ball']
    verbose = args.get('verbose')
    frame_dimensions = cap.dim()

    tracker = Tracker(frame_dimensions, ball_calibration, goals_calibration, verbose, args.get("buffer"))

    while True:
        frame = cap.next()
        if frame is None:
            break

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Calibration", f"{'on' if args.get('calibration') else 'off'}"),
            ("Tracker", f"{'off' if args.get('off') else 'on'}"),
            ("FPS", f"{int(cap.fps_real())} / {int(cap.fps_stream())}")
        ]

        if not args.get('off'):
            detected_ball, tracker_info = tracker.track(frame)
            info += tracker_info
            if detected_ball is not None:
                [center, bbox] = detected_ball
                mark_ball(frame, center, bbox)
            frame = tracker.render_track(frame, render_mask=True)

        render_info(frame, info)

        show("Frame", frame, 'tl')

        if poll_key(calibration_mode, tracker):
            break

    cap.stop()
    destroy_all_windows()


def render_info(frame, info):
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        x = int(i / 2) * 300
        y = (i % 2) * 20
        cv2.putText(frame, text, (x, 40 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


def mark_ball(frame, center, bbox):
    [x, y, w, h] = bbox
    # only proceed if the radius meets a minimum size
    if 9 < w < 33 and 9 < h < 33:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)


def poll_key(calibration_mode, tracker, interval=1):
    return wait(calibration_mode, tracker, loop=False, interval=interval)


def wait(calibration_mode, tracker, loop=False, interval=0.1):
    while True:
        key = cv2.waitKey(interval) & 0xFF
        # if the expected key is pressed, return
        if key == ord('q'):
            return True
        if key == ord('r') and calibration_mode:
            tracker.reset_bounds()
            return False

        if not loop:
            break
    return False
