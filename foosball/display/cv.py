import cv2


class OpenCVDisplay:

    def __init__(self, calibration_mode):
        self.all_windows = []
        self.tracker = None
        self.calibration_mode = calibration_mode

    def set_tracker(self, tracker):
        self.tracker = tracker

    @staticmethod
    def title(s):
        print(f'{"=" * 8} {s} {"=" * 8}')

    def stop(self):
        cv2.destroyAllWindows()
        self.all_windows.clear()

    def show(self, name, frame, pos='tl'):
        [x, y] = {
            'tl': [10, 0],
            'tr': [1310, 0],
            'bl': [10, 900],
            'br': [1310, 900]
        }[pos]

        if name not in self.all_windows:
            cv2.namedWindow(name)
            cv2.moveWindow(name, x, y)
            self.all_windows.append(name)
        if frame is not None:
            cv2.imshow(name, frame)

    def poll_key(self):
        return self._wait(loop=False, interval=1)

    def _wait(self, loop=False, interval=0.1):
        while True:
            key = cv2.waitKey(interval) & 0xFF
            # if the expected key is pressed, return
            if key == ord('q'):
                return True
            if key == ord('r') and self.calibration_mode:
                self.tracker.reset()
                return False

            if not loop:
                break
        return False
