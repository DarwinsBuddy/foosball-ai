# TODO: #1 solve fps issue (we are not capturing high speed) maybe buffer it?

import cv2
from imutils.video import FPS

from .display.cv import reset_bounds
from .tracker import Tracker


class AI:

    def __init__(self, args, cap, display):
        self.args = args
        self.cap = cap
        self.display = display
        self._stopped = False

    def stop(self):
        self._stopped = True

    def process_video(self):
        ball_calibration = self.args.get('calibration') in ['all', 'ball']
        verbose = self.args.get('verbose')
        frame_dimensions = self.cap.dim()

        tracker = Tracker(frame_dimensions, ball_calibration, verbose, self.args.get("buffer"))

        def reset_cb():
            if ball_calibration:
                tracker.reset()
        fps = FPS()

        fps.start()
        while not self._stopped:
            fps.update()
            frame = self.cap.next()
            if frame is None:
                break

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Calibration", f"{'on' if self.args.get('calibration') else 'off'}"),
                ("Tracker", f"{'off' if self.args.get('off') else 'on'}")
            ]

            if not self.args.get('off'):
                detected_ball, tracker_info = tracker.track(frame)
                info += tracker_info
                if detected_ball is not None:
                    [center, bbox] = detected_ball
                    self.mark_ball(frame, center, bbox)
                frame = tracker.render_track(frame, render_mask=True)
            fps.stop()
            current_fps = int(fps.fps())
            info += [("FPS", f"{current_fps}")]

            if not self.args.get("headless"):
                self.render_info(frame, info)
                # frame = self.scale(frame, 0.5)
                self.display.show(frame)
            print(" - ".join([f"{label}: {text}" for label, text in info]) + (" " * 20), end="\r")

            if self.display.render(reset_cb=reset_cb):
                break

        self.cap.stop()
        self.display.stop()

    @staticmethod
    def scale(src, scale_percent):

        # calculate the 50 percent of original dimensions
        width = int(src.shape[1] * scale_percent)
        height = int(src.shape[0] * scale_percent)

        # dsize
        dsize = (width, height)
        return cv2.resize(src, dsize)

    @staticmethod
    def render_info(frame, info):
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            x = int(i / 2) * 300
            y = (i % 2) * 20
            cv2.putText(frame, text, (x, 40 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    @staticmethod
    def mark_ball(frame, center, bbox):
        [x, y, w, h] = bbox
        # only proceed if the radius meets a minimum size
        if 9 < w < 33 and 9 < h < 33:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
