# TODO: solve fps issue (we are not capturing high speed) maybe buffer it?

from collections import deque
import numpy as np
import cv2
import imutils

from . import destroy_all_windows, show, poll_key
from .capture import EndOfStreamException


def process_video(args, cap):
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points

    greenLower = (0, 140, 170)
    greenUpper = (80, 255, 255)

    # greenLower = (0, 143, 175)
    # greenUpper = (89, 255, 255)

    pts = deque(maxlen=args["buffer"])

    try:
        while True:
            frame = cap.next()
            # print(f"frame {int(next_frame)}/{total_frames}")

            # handle the frame from VideoCapture or VideoStream
            # frame = frame[1] if args.get("video", False) else frame

            # if we are viewing a video and we did not grab a frame,

            kernel1 = np.ones((1, 5), np.uint8)
            kernel2 = np.ones((5, 5), np.uint8)
            kernel3 = np.ones((3, 3), np.uint8)

            frame = imutils.resize(frame, width=1024)

            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # hsv = cv2.dilate(hsv, kernel, iterations=1)
            mask3 = cv2.inRange(hsv, greenLower, greenUpper)
            mask2 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel3)

            mask1 = cv2.dilate(mask2, kernel2, iterations=1)

            mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel3)
            mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel3)

            # # resize the frame, blur it, and convert it to the HSV
            # # color space
            # #frame = imutils.resize(frame, width=600)
            # #blurred = cv2.GaussianBlur(frame, (1, 1), 0)
            # #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # # construct a mask for the color "green", then perform
            # # a series of dilations and erosions to remove any small
            # # blobs left in the mask
            # #mask = cv2.inRange(hsv, greenLower, greenUpper)
            # #mask = cv2.erode(mask, None, iterations=2)
            # #mask = cv2.dilate(mask, None, iterations=2)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                # ((x, y), radius) = cv2.minEnclosingCircle(c)
                x, y, w, h = cv2.boundingRect(c)

                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if 9 < w < 33 and 9 < h < 33:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)
            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # show the frame to our screen
            # cv2.imshow("Frame", frame)
            # cv2.imshow("Mask", mask)

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("FPS", "{:.2f}".format(cap.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, 50 - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # The frame is ready and already captured
            # cv2.imshow('video', frame)
            show("Frame", frame, 0, 0)
            # show("Blur", blurred, 500, 0)
            # show("Filtrado", mask3, 1000, 0)
            # show("Erosionado", mask2, 500, 500)
            # show("Dilatado", mask1, 500, 500)
            # show("Rellenado", mask, 500, 1000)

            if poll_key():
                break
    except EndOfStreamException:
        print("End of Stream")

    destroy_all_windows()
