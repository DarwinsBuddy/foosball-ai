import cv2
import numpy as np
import yaml

from foosball.tracking import BallConfig
from foosball.tracking.models import rgb2hsv, hsv2rgb, GoalConfig

GOAL = "goal"
BALL = "ball"


class OpenCVDisplay:

    def __init__(self, name='frame', pos='tl'):
        self.name = name
        cv2.namedWindow(self.name)
        [x, y] = self._position(pos)
        cv2.moveWindow(self.name, x, y)

    @staticmethod
    def _position(pos):
        return {
            'tl': [10, 0],
            'tr': [1310, 0],
            'bl': [10, 900],
            'br': [1310, 900]
        }[pos]

    @staticmethod
    def title(s):
        print(f'{"=" * 8} {s} {"=" * 8}')

    def stop(self):
        cv2.destroyWindow(self.name)

    def show(self, frame):
        if frame is not None:
            cv2.imshow(self.name, frame)

    @staticmethod
    def render(reset_cb=None, store_cb=None):
        return wait(loop=False, interval=1, reset_cb=reset_cb, store_cb=store_cb)


def wait(loop=False, interval=0.1, reset_cb=None, store_cb=None):
    while True:
        key = cv2.waitKey(interval) & 0xFF
        # if the expected key is pressed, return
        if key == ord('q'):
            return True
        if key == ord('r') and reset_cb is not None:
            reset_cb()
            return False
        if key == ord('s') and store_cb is not None:
            store_cb()
            return False

        if not loop:
            break
    return False


def slider_label(rgb, bound):
    return f"{rgb} ({bound})"


def add_config_input(calibration, config):
    if calibration == GOAL:
        add_goals_config_input(config)
    elif calibration == BALL:
        add_ball_config_input(config)


def add_ball_config_input(bounds: BallConfig):
    [lower_hsv, upper_hsv] = bounds.bounds_hsv
    lower_rgb = hsv2rgb(lower_hsv)
    upper_rgb = hsv2rgb(upper_hsv)
    cv2.createTrackbar(f'invert_frame', BALL, 1 if bounds.invert_frame else 0, 1, lambda v: None)
    cv2.createTrackbar(f'invert_mask', BALL, 1 if bounds.invert_mask else 0, 1, lambda v: None)
    # create trackbars for color change
    cv2.createTrackbar(slider_label('R', 'low'), BALL, lower_rgb[0], 255, lambda v: None)
    cv2.createTrackbar(slider_label('G', 'low'), BALL, lower_rgb[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('B', 'low'), BALL, lower_rgb[2], 255, lambda v: None)
    cv2.createTrackbar(slider_label('R', 'high'), BALL, upper_rgb[0], 255, lambda v: None)
    cv2.createTrackbar(slider_label('G', 'high'), BALL, upper_rgb[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('B', 'high'), BALL, upper_rgb[2], 255, lambda v: None)
    # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))


def add_goals_config_input(config: GoalConfig):
    [lower, upper] = config.bounds
    cv2.createTrackbar(f'invert_frame', GOAL, 1 if config.invert_frame else 0, 1, lambda v: None)
    cv2.createTrackbar(f'invert_mask', GOAL, 1 if config.invert_mask else 0, 1, lambda v: None)
    # create trackbars for color change
    cv2.createTrackbar("lower", GOAL, lower, 255, lambda v: None)
    cv2.createTrackbar("upper", GOAL, upper, 255, lambda v: None)
    # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))


def reset_config(calibration, config):
    if calibration == GOAL:
        reset_goal_config(config)
    elif calibration == BALL:
        reset_ball_config(config)


def reset_ball_config(bounds: BallConfig):
    [lower_hsv, upper_hsv] = bounds.bounds_hsv
    print(f"Reset config {BALL}", end="\n\n\n")
    lower_rgb = hsv2rgb(lower_hsv)
    upper_rgb = hsv2rgb(upper_hsv)

    cv2.setTrackbarPos('invert_frame', BALL, 1 if bounds.invert_frame else 0)
    cv2.setTrackbarPos('invert_mask', BALL, 1 if bounds.invert_mask else 0)

    cv2.setTrackbarPos(slider_label('R', 'low'), BALL, lower_rgb[0])
    cv2.setTrackbarPos(slider_label('G', 'low'), BALL, lower_rgb[1])
    cv2.setTrackbarPos(slider_label('B', 'low'), BALL, lower_rgb[2])
    cv2.setTrackbarPos(slider_label('R', 'high'), BALL, upper_rgb[0])
    cv2.setTrackbarPos(slider_label('G', 'high'), BALL, upper_rgb[1])
    cv2.setTrackbarPos(slider_label('B', 'high'), BALL, upper_rgb[2])


def reset_goal_config(config: GoalConfig):
    [lower, upper] = config.bounds_hsv
    print(f"Reset config {GOAL}", end="\n\n\n")

    cv2.setTrackbarPos('invert_frame', GOAL, 1 if config.invert_frame else 0)
    cv2.setTrackbarPos('invert_mask', GOAL, 1 if config.invert_mask else 0)

    cv2.setTrackbarPos('lower', GOAL, lower)
    cv2.setTrackbarPos('upper', GOAL, upper)


def store_config(calibration, bounds):
    if calibration == GOAL:
        store_goals_config(bounds)
    elif calibration == BALL:
        store_ball_config(bounds)


def store_ball_config(config: BallConfig):
    filename = f"ball.yaml"
    [lower_hsv, upper_hsv] = config.bounds_hsv
    print(f"Store config {filename}" + (" " * 50), end="\n\n")
    lower_rgb = hsv2rgb(lower_hsv)
    upper_rgb = hsv2rgb(upper_hsv)
    with open(filename, "w") as f:
        yaml.dump({
            "lower_rgb": lower_rgb.tolist(),
            "upper_rgb": upper_rgb.tolist(),
            "invert_frame": config.invert_frame,
            "invert_mask": config.invert_mask
        }, f)


def store_goals_config(config: GoalConfig):
    filename = f"goal.yaml"
    [lower, upper] = config.bounds
    print(f"Store config {filename}" + (" " * 50), end="\n\n")
    with open(filename, "w") as f:
        yaml.dump({
            "lower": lower,
            "upper": upper,
            "invert_frame": config.invert_frame,
            "invert_mask": config.invert_mask
        }, f)


def get_slider_config(calibration):
    if calibration == GOAL:
        return get_slider_goals_config()
    elif calibration == BALL:
        return get_slider_ball_config()


def get_slider_ball_config():
    # get current positions of four trackbars
    invert_frame = cv2.getTrackbarPos('invert_frame', BALL)
    invert_mask = cv2.getTrackbarPos('invert_mask', BALL)

    rl = cv2.getTrackbarPos(slider_label('R', 'low'), BALL)
    rh = cv2.getTrackbarPos(slider_label('R', 'high'), BALL)

    gl = cv2.getTrackbarPos(slider_label('G', 'low'), BALL)
    gh = cv2.getTrackbarPos(slider_label('G', 'high'), BALL)

    bl = cv2.getTrackbarPos(slider_label('B', 'low'), BALL)
    bh = cv2.getTrackbarPos(slider_label('B', 'high'), BALL)
    lower = rgb2hsv(np.array([rl, gl, bl]))
    upper = rgb2hsv(np.array([rh, gh, bh]))
    return BallConfig(bounds_hsv=[lower, upper], invert_mask=invert_mask, invert_frame=invert_frame)


def get_slider_goals_config():
    # get current positions of four trackbars
    invert_frame = cv2.getTrackbarPos('invert_frame', GOAL)
    invert_mask = cv2.getTrackbarPos('invert_mask', GOAL)

    lower = cv2.getTrackbarPos('lower', GOAL)
    upper = cv2.getTrackbarPos('upper', GOAL)

    return GoalConfig(bounds=[lower, upper], invert_mask=invert_mask, invert_frame=invert_frame)
