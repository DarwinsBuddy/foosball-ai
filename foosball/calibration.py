import cv2

from foosball import title, show, wait


def calibration(args, cap):
    title("CALIBRATION MODE")
    frame = cap.next()
    show('calibration', frame)

    wait(loop=True)
    cap.stop()
