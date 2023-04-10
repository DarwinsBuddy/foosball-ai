import cv2
import pypeln as pl
import numpy as np

from . import FrameDimensions, DetectionResult

TEXT_SCALE = 0.8
TEXT_COLOR = (0, 255, 0)

def r_info(frame, dims: FrameDimensions, info):
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        txt = "{}: {}".format(k, v)
        x = (int(i / 2) * 100)
        y = dims.scaled[1] - ((i % 2) * 20)
        r_text(frame, txt, x + 10, y - int(TEXT_SCALE * 20), dims.scale)

def r_text(frame, text: str, x: int, y: int, scale: float):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * TEXT_SCALE, TEXT_COLOR, 1)
def r_ball(frame, b, scale):
    [center, bbox] = b
    [x, y, w, h] = bbox

    minimum = scale * 9
    maximum = scale * 33
    # only proceed if the radius meets a minimum size
    if minimum < w < maximum and minimum < h < maximum:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        # cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
def r_track(frame, ball_track, scale):
    # loop over the set of tracked points
    for i in range(1, len(ball_track)):
        # if either of the tracked points are None, ignore
        # them
        if ball_track[i - 1] is None or ball_track[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        r = 255 - (255 * (i / ball_track.maxlen))
        g = 255 * (i / ball_track.maxlen)
        b = 255 - (255 * (i / ball_track.maxlen))
        thickness = max(1, int(int(np.sqrt(ball_track.maxlen / float(i + 1)) * 2) * scale))
        cv2.line(frame, ball_track[i - 1], ball_track[i], (b, g, r), thickness)

class Renderer:
    def __init__(self, dims: FrameDimensions, headless=False, **kwargs):
        self.dims = dims
        self.headless = headless
        self.kwargs = kwargs
        self.out = pl.process.IterableQueue()

    def stop(self):
        self.out.stop()

    def render(self, detection_result: DetectionResult) -> DetectionResult:
        f = detection_result.rendered_frame
        ball = detection_result.ball
        track = detection_result.ball_track
        info = detection_result.info

        try:
            if ball is not None:
                r_ball(f, ball, self.dims.scale)

            r_track(f, track, self.dims.scale)
            if not self.headless:
                r_info(f, self.dims, info)
            print(" - ".join([f"{label}: {text}" for label, text in info]) + (" " * 20), end="\r")

            self.out.put_nowait(f)
        except Exception as e:
            print("Error in renderer ", e)
        return DetectionResult(detection_result.frame, f, detection_result.ball_track, detection_result.ball, detection_result.info)
