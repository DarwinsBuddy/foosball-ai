import cv2

from foosball.utils import rgb2hsv, hsv2rgb


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
    def render(reset_cb=None):
        return wait(loop=False, interval=1, reset_cb=reset_cb)


def wait(loop=False, interval=0.1, reset_cb=None):
    while True:
        key = cv2.waitKey(interval) & 0xFF
        # if the expected key is pressed, return
        if key == ord('q'):
            return True
        if key == ord('r') and reset_cb is not None:
            reset_cb()
            return False

        if not loop:
            break
    return False
    

def slider_label(rgb, bound):
    return f"{rgb} ({bound})"


def add_calibration_input(name, lower_hsv, upper_hsv):
    print("lower, upper ", lower_hsv, upper_hsv)
    lower_rgb = hsv2rgb(lower_hsv)
    upper_rgb = hsv2rgb(upper_hsv)
    # create trackbars for color change
    cv2.createTrackbar(slider_label('R', 'low'), name, lower_rgb[0], 255, lambda v: None)
    cv2.createTrackbar(slider_label('G', 'low'), name, lower_rgb[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('B', 'low'), name, lower_rgb[2], 255, lambda v: None)
    cv2.createTrackbar(slider_label('R', 'high'), name, upper_rgb[0], 255, lambda v: None)
    cv2.createTrackbar(slider_label('G', 'high'), name, upper_rgb[1], 255, lambda v: None)
    cv2.createTrackbar(slider_label('B', 'high'), name, upper_rgb[2], 255, lambda v: None)
    # cv2.createButton("Reset", reset_bounds, (name, lower_rgb, upper_rgb))


def reset_bounds(name, lower_hsv, upper_hsv):
    print(f"Reset color bounds {name}")
    lower_rgb = hsv2rgb(lower_hsv)
    upper_rgb = hsv2rgb(upper_hsv)

    cv2.setTrackbarPos(slider_label('R', 'low'), name, lower_rgb[0])
    cv2.setTrackbarPos(slider_label('G', 'low'), name, lower_rgb[1])
    cv2.setTrackbarPos(slider_label('B', 'low'), name, lower_rgb[2])
    cv2.setTrackbarPos(slider_label('R', 'high'), name, upper_rgb[0])
    cv2.setTrackbarPos(slider_label('G', 'high'), name, upper_rgb[1])
    cv2.setTrackbarPos(slider_label('B', 'high'), name, upper_rgb[2])


def get_slider_bounds(name, mode="hsv"):
    # get current positions of four trackbars
    rl = cv2.getTrackbarPos(slider_label('R', 'low'), name)
    rh = cv2.getTrackbarPos(slider_label('R', 'high'), name)

    gl = cv2.getTrackbarPos(slider_label('G', 'low'), name)
    gh = cv2.getTrackbarPos(slider_label('G', 'high'), name)

    bl = cv2.getTrackbarPos(slider_label('B', 'low'), name)
    bh = cv2.getTrackbarPos(slider_label('B', 'high'), name)
    if mode == 'hsv':
        lower = rgb2hsv((rl, gl, bl))
        upper = rgb2hsv((rh, gh, bh))
    else:
        lower = (rl, gl, bl)
        upper = (rh, gh, bh)
    return [lower, upper]
