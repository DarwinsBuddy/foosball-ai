import cv2

from foosball import title, show, wait


def rgb(img, x, y):
    return f"[{x},{y}], RGB = {img[y, x, 2]} {img[y, x, 1]} {img[y, x, 0]}"


def cb(img):
    def f(event, x, y, flags, params):
        if event == 1:
            print(rgb(img, x, y))
    return f


def calibration_legacy(args, cap):
    title("CALIBRATION MODE")
    frame = cap.next()
    show('calibration', frame)
    cv2.setMouseCallback("calibration", cb(frame))
    wait(loop=True)
    cap.stop()
