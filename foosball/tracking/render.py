import logging
import traceback

import cv2
import numpy as np
import pypeln as pl

from . import FrameDimensions
from .colordetection import Blob
from .models import Info, Goal, AnalyzeResult, Score

TEXT_SCALE = 1.2

# BGR
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
WHITE = (0, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 143, 252)


def text_color(key, value):
    if value == "off" or value == "fail" or key.startswith("!"):
        return GRAY
    elif key.startswith("?"):
        return ORANGE
    else:
        return GREEN


def r_info(frame, dims: FrameDimensions, info: Info) -> None:
    # loop over the info tuples and draw them on our frame
    height = int(len(info) / 2) * 30
    cv2.rectangle(frame, (0, dims.scaled[1] - height), (dims.scaled[0], dims.scaled[1]), (0, 0, 0), -1)
    for (i, (k, v)) in enumerate(info):
        txt = "{}: {}".format(k, v)
        x = (int(i / 2) * 170) + 10
        y = dims.scaled[1] - ((i % 2) * 20)
        r_text(frame, txt, x, y - int(TEXT_SCALE * 20), dims.scale, text_color(txt, v))


def r_score(frame, score: Score, dims: FrameDimensions) -> None:
    r_text(frame, f"{score.blue} : {score.red}", int(dims.scaled[0] / 2), 30, dims.scale, GREEN)


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
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 1)
        # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)


def r_goal(frame, g: Goal) -> None:
    [x, y, w, h] = g.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)


def r_track(frame, ball_track, scale) -> None:
    # loop over the set of tracked points
    for i in range(1, len(ball_track or [])):
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

    def render(self, analyze_result: AnalyzeResult) -> AnalyzeResult:
        f = analyze_result.frame
        ball = analyze_result.ball
        goals = analyze_result.goals
        track = analyze_result.ball_track
        info = analyze_result.info
        score = analyze_result.score

        try:
            if ball is not None:
                r_ball(f, ball, self.dims.scale)
            if goals is not None:
                r_goal(f, goals.left)
                r_goal(f, goals.right)
            r_track(f, track, self.dims.scale)
            r_score(f, score, self.dims)
            if not self.headless:
                r_info(f, self.dims, info)
            else:
                print(" - ".join([f"{label}: {text}" for label, text in info]) + (" " * 50), end="\r")
            self.out.put_nowait(f)
        except Exception as e:
            logging.error("Error in renderer {e}")
            traceback.print_exc()
        return analyze_result
