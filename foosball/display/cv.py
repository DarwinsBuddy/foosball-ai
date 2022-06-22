import cv2


class OpenCVDisplay:

    def __init__(self):
        self.all_windows = []

    @staticmethod
    def title(s):
        print(f'{"=" * 8} {s} {"=" * 8}')

    def destroy_all_windows(self):
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
        cv2.imshow(name, frame)

    def poll_key(self, calibration_mode, tracker, interval=1):
        return self.wait(calibration_mode, tracker, loop=False, interval=interval)

    @staticmethod
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
