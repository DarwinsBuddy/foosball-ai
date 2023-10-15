import logging
import traceback

import cv2
import numpy as np

from const import INFO_VERBOSITY
from .preprocess import corners2pt, PositionEstimationInputs
from ..arUcos.calibration import draw_markers
from ..models import Goal, Score, FrameDimensions, Blob, InfoLog, Verbosity
from ..pipe.BaseProcess import Msg, BaseProcess
from ..utils import generate_processor_switches
logger = logging.getLogger(__name__)


TEXT_SCALE = 0.8
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 143, 252)


def text_color(key, value):
    if value.strip() in ["off", "fail", "none"] or key.startswith("!"):
        return GRAY
    elif key.startswith("?"):
        return ORANGE
    else:
        return GREEN


def r_info(frame, shape: tuple[int, int, int], info: InfoLog, text_scale=1.0, thickness=1) -> None:
    [height, width, channels] = shape
    # loop over the info tuples and draw them on our frame
    x = 0
    y = height
    h = 0
    w = 0
    for i in info:
        text = i.to_string()
        if w + x > width:
            cv2.rectangle(frame, (x, y), (width, y - h), BLACK, -1) # fill
            x = 0
            y = y - h
        [x0, _, w, h] = r_text(frame, text, x, y, text_color(text, i.value), text_scale=text_scale, thickness=thickness, background=BLACK, padding=(20, 20), ground_zero='bl')
        x = x0 + w
    cv2.rectangle(frame, (x, y), (width, y - h), BLACK, -1)  # fill


def r_text(frame, text: str, x: int, y: int, color=GREEN, text_scale=1.0, thickness=1, background=None, padding=(0, 0), ground_zero='bl'):
    horizontal = ground_zero[1]
    vertical = ground_zero[0]
    [text_width, text_height] = cv2.getTextSize(text, FONT, text_scale, thickness)[0]
    x0 = x - text_width if horizontal == 'r' else x
    y0 = y + text_height if vertical == 't' else y - text_height
    roi = (x0, y0, text_width, text_height)
    if background is not None:
        y0 = y0 - padding[1]
        x0 = x0 - padding[0] if horizontal == 'r' else x0
        x1 = x0 + text_width + padding[0]
        y1 = y0 + text_height + padding[1]
        cv2.rectangle(frame, (x0, y0), (x1, y1), BLACK, -1)
        roi = (x0, y0, text_width + padding[0], text_height + padding[1])
    cv2.putText(frame, text, (x0 + int(padding[0]/2), y0 + text_height + int(padding[1]/2)), FONT, text_scale, color, thickness)
    return roi


def r_score(frame, score: Score, text_scale=1, thickness=1) -> None:
    r_text(frame, score.to_string(),  0, 0, GREEN, background=BLACK, padding=(5, 20), text_scale=text_scale, thickness=thickness, ground_zero='tl')


def r_ball(frame, b: Blob) -> None:
    [x, y, w, h] = b.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 1)


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


class Renderer(BaseProcess):
    def close(self):
        pass

    def __init__(self, dims: FrameDimensions, headless=False, useGPU: bool = False, *args, **kwargs):
        super().__init__(name="Renderer")
        self.dims = dims
        self.headless = headless
        self.infoVerbosity = Verbosity(kwargs.get(INFO_VERBOSITY)) if kwargs.get(INFO_VERBOSITY) else None
        [self.proc, self.iproc] = generate_processor_switches(useGPU)

    def r_distance(self, frame: Frame, arucos: list[Aruco], position_estimation_inputs: PositionEstimationInputs):
        if arucos is not None:
            (i1, a1), (i2,a2) = [(i, a) for i, a in enumerate(arucos) if a.id in [0, 3]]
            apt1 = np.array(corners2pt(a1.corners))
            apt2 = np.array(corners2pt(a2.corners))
            cv2.line(frame, apt1, apt2, ORANGE, 2, 1)
            # TODO: somehow this shows, that the markers are always equally wrong distant from each other, no matter
            #       the given marker length
            # print(position_estimation_inputs.marker_positions_3d)
            d = cv2.norm(position_estimation_inputs.marker_positions_3d[i1], position_estimation_inputs.marker_positions_3d[i2], cv2.NORM_L2)
            pos = [int(x) for x in (apt1 + apt2) / 2]
            r_text(frame, f'{d:2f} cm', pos[0], pos[1], ORANGE, thickness=2)


    def process(self, msg: Msg) -> Msg:
        analyze_result = msg.kwargs['result']
        arucos = msg.kwargs.get('arucos')
        position_estimation_inputs = msg.kwargs.get('positionEstimationInputs')
        calibration = msg.kwargs.get('calibration')
        timestamp = msg.kwargs['time']
        info: InfoLog = analyze_result.info
        try:
            if not self.headless:
                shape = analyze_result.frame.shape
                f = self.proc(analyze_result.frame)
                ball = analyze_result.ball
                goals = analyze_result.goals
                track = analyze_result.ball_track
                score = analyze_result.score

                draw_markers(f, arucos, calibration)
                self.r_distance(f, arucos, position_estimation_inputs)
                if ball is not None:
                    r_ball(f, ball)
                if goals is not None:
                    r_goal(f, goals.left)
                    r_goal(f, goals.right)
                r_track(f, track, self.dims.scale)
                r_score(f, score, text_scale=1, thickness=4)
                if self.infoVerbosity is not None:
                    r_info(f, shape, info.filter(self.infoVerbosity), text_scale=0.5, thickness=1)
                return Msg(kwargs={**msg.kwargs, "time": timestamp, "result": self.iproc(f), 'info': info, "speed": msg.kwargs['speed']})
            else:
                return Msg(kwargs={**msg.kwargs, "time": timestamp, "result": None, 'info': info, "speed": msg.kwargs['speed']})

        except Exception as e:
            logger.error(f"Error in renderer {e}")
            traceback.print_exc()
        return Msg(analyze_result)
