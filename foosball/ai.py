# TODO: #1 solve fps issue (we are not capturing high speed) maybe buffer it?

import cv2
from imutils.video import FPS

from .tracker import Tracker


def process_video(args, cap, display):
    ball_calibration = args.get('calibration') in ['all', 'ball']
    verbose = args.get('verbose')
    frame_dimensions = cap.dim()

    tracker = Tracker(frame_dimensions, display, ball_calibration, verbose, args.get("buffer"))
    display.set_tracker(tracker)
    fps = FPS()

    fps.start()
    while True:
        fps.update()
        frame = cap.next()
        if frame is None:
            break

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Calibration", f"{'on' if args.get('calibration') else 'off'}"),
            ("Tracker", f"{'off' if args.get('off') else 'on'}")
        ]

        if not args.get('off'):
            detected_ball, tracker_info = tracker.track(frame)
            info += tracker_info
            if detected_ball is not None:
                [center, bbox] = detected_ball
                mark_ball(frame, center, bbox)
            frame = tracker.render_track(frame, render_mask=True)
        fps.stop()
        current_fps = int(fps.fps())
        info += [("FPS", f"{current_fps}")]

        if not args.get("headless"):
            render_info(frame, info)
            # frame = scale(frame, 0.5)
            display.show("Frame", frame, 'tl')
        print(" - ".join([f"{label}: {text}" for label, text in info]) + (" " * 20), end="\r")

        if display.poll_key():
            break

    cap.stop()
    display.stop()


def scale(src, scale_percent):

    # calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent)
    height = int(src.shape[0] * scale_percent)

    # dsize
    dsize = (width, height)
    return cv2.resize(src, dsize)


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
