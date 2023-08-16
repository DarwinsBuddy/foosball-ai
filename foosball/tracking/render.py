import cv2
import pypeln as pl
import numpy as np

from . import FrameDimensions
from .colordetection import Blob
from .models import TrackResult, Info

TEXT_SCALE = 1.2
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
ORANGE = (0, 143, 252)


def text_color(key, value):
    if value == "off" or key.startswith("!"):
        return GRAY
    elif key.startswith("?"):
        return ORANGE
    else:
        return GREEN


def r_info(frame, dims: FrameDimensions, info: Info) -> None:
    # loop over the info tuples and draw them on our frame
    height = int(len(info) / 2) * 25
    cv2.rectangle(frame, (0, dims.scaled[1] - height), (dims.scaled[0], dims.scaled[1]), (0, 0, 0), -1)
    for (i, (k, v)) in enumerate(info):
        txt = "{}: {}".format(k, v)
        x = (int(i / 2) * 170) + 10
        y = dims.scaled[1] - ((i % 2) * 20)
        r_text(frame, txt, x, y - int(TEXT_SCALE * 20), dims.scale, text_color(txt, v))


def r_text(frame, text: str, x: int, y: int, scale: float, color=GREEN):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * TEXT_SCALE, color, 1)


def r_ball(frame, b: Blob, scale) -> None:
    [x, y, w, h] = b.bbox

    minimum = scale * 9
    maximum = scale * 33
    # only proceed if the radius meets a minimum size
    if minimum < w < maximum and minimum < h < maximum:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        # cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)


def r_track(frame, ball_track, scale) -> None:
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

    def stop(self) -> None:
        self.out.stop()

    def render(self, track_result: TrackResult) -> TrackResult:
        f = track_result.frame
        ball = track_result.ball
        track = track_result.ball_track
        info = track_result.info

        try:
            if ball is not None:
                r_ball(f, ball, self.dims.scale)

            r_track(f, track, self.dims.scale)
            if not self.headless:
                r_info(f, self.dims, info)
            print(" - ".join([f"{label}: {text}" for label, text in info]) + (" " * 80), end="\r")
            self.out.put_nowait(f)
        except Exception as e:
            print("Error in renderer ", e)
        return TrackResult(f, track_result.ball_track, track_result.ball,
                           track_result.info)
