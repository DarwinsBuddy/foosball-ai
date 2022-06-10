# TODO: #1 solve fps issue (we are not capturing high speed) maybe buffer it?

from collections import deque
import numpy as np
import cv2
import imutils

from . import destroy_all_windows, show, poll_key
from .capture import EndOfStreamException


def get_ball_bounds_hsv():
    # greenLower = (0, 140, 170)
    # greenUpper = (80, 255, 255)

    # green_lower = (0, 143, 175)
    # green_upper = (89, 255, 255)

    # green = [green_lower, green_upper]

    # orange_lower_rgb = (115, 70, 80)
    # orange_upper_rgb = (190, 120, 25)
    # orange_hsv = [rgb2hsv(orange_lower_rgb), rgb2hsv(orange_upper_rgb)]

    # TODO: #2 calibration for the demo footage (other ball => other values)
    lower = rgb2hsv((166, 94, 72))
    upper = rgb2hsv((0, 249, 199))

    return [lower, upper]


def rgb2hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]


def hsv2rgb(hsv):
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]


def generate_bar_mask(cap):
    bar_positions = [250, 350, 450, 550]

    bar_width = 20
    bar_color = 0
    bg = 255
    [width, height] = cap.dim()
    bar_mask = np.full((height, width), bg, np.uint8)
    for x_pos in bar_positions:
        [x, y] = [x_pos, 0]
        bar_dim = [x+bar_width, y+height]
        bar_mask = cv2.rectangle(bar_mask, (x, y), bar_dim, bar_color, -1)
    return bar_mask


def generate_frame_mask(cap):
    bar_color = 255
    bg = 0
    [width, height] = cap.dim()
    mask = np.full((height, width), bg, np.uint8)
    frame_mask = cv2.rectangle(mask, (100, 70), (1100, 620), bar_color, -1)
    return frame_mask


def mask_img(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def add_slider(window_name, lowerRGB, upperRGB):
    # create trackbars for color change
    cv2.createTrackbar('R-low', window_name, lowerRGB[0], 255, lambda v: None)
    cv2.createTrackbar('R-high', window_name, upperRGB[0], 255, lambda v: None)

    cv2.createTrackbar('G-low', window_name, lowerRGB[1], 255, lambda v: None)
    cv2.createTrackbar('G-high', window_name, upperRGB[1], 255, lambda v: None)

    cv2.createTrackbar('B-low', window_name, lowerRGB[2], 255, lambda v: None)
    cv2.createTrackbar('B-high', window_name, upperRGB[2], 255, lambda v: None)


def get_slider_bounds(window_name):
    # get current positions of four trackbars
    rl = cv2.getTrackbarPos('R-low', window_name)
    rh = cv2.getTrackbarPos('R-high', window_name)

    gl = cv2.getTrackbarPos('G-low', window_name)
    gh = cv2.getTrackbarPos('G-high', window_name)

    bl = cv2.getTrackbarPos('B-low', window_name)
    bh = cv2.getTrackbarPos('B-high', window_name)
    lower = rgb2hsv((rl, gl, bl))
    upper = rgb2hsv((rh, gh, bh))
    return [lower, upper]


def draw_bounds(img, lower, upper):
    cv2.putText(img=img, text=f'Lower RGB={hsv2rgb(lower)}', org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0, 255, 0), thickness=1)
    cv2.putText(img=img, text=f'Upper RGB={hsv2rgb(upper)}', org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0, 255, 0), thickness=1)


def filter_ball(frame, lower, upper, verbose=False):
    blurred = cv2.GaussianBlur(frame, (1, 1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the simple frame
    simple = cv2.inRange(hsv, lower, upper)
    # show("mask", mask, 100, 500)
    simple = cv2.erode(simple, None, iterations=2)
    # show("erode", mask, 500, 500)
    simple = cv2.dilate(simple, None, iterations=2)
    if verbose:
       show("dilate", simple, 'bl')

    # ## for masking
    # cleaned = mask_img(simple, mask=bar_mask)
    # contrast = mask_img(frame, cleaned)
    # show("contrast", contrast, 'br')
    # show("frame", mask_img(frame, bar_mask), 'tl')

    return cv2.bitwise_and(frame, frame, mask=simple)


def process_video(args, cap):
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points

    calibration_mode = args.get('calibration')
    verbose = args.get('verbose')
    pts = deque(maxlen=args["buffer"])

    [lower, upper] = get_ball_bounds_hsv()

    # init slider window
    if calibration_mode:
        cv2.namedWindow('img')
        add_slider('img', hsv2rgb(lower), hsv2rgb(upper))

    try:
        while True:
            frame = cap.next()
            frame = mask_img(frame, generate_frame_mask(cap))
            if calibration_mode:
                [lower, upper] = get_slider_bounds('img')

            img = filter_ball(frame, lower, upper, verbose)
            if calibration_mode:
                draw_bounds(img, lower, upper)
            if verbose:
                show("img", img, 'br')

            # kernel1 = np.ones((1, 5), np.uint8)
            # kernel2 = np.ones((5, 5), np.uint8)
            # kernel3 = np.ones((3, 3), np.uint8)

            # frame = imutils.resize(frame, width=1024)

            # blurred = cv2.GaussianBlur(img, (3, 3), 0)
            # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # hsv = cv2.dilate(hsv, kernel1, iterations=1)
            # mask3 = cv2.inRange(hsv, lower, upper)
            # mask2 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel3)

            # mask1 = cv2.dilate(mask2, kernel2, iterations=1)

            # mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel3)
            # mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel3)

            # resize the frame, blur it, and convert it to the HSV
            # color space

            # show("hsvBlurred", hsv, 500, 100)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            # cnts = cv2.findContours(simple.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            show("Frame", frame, 'tl')
            # show("Blur", blurred, 'tl')
            # show("Filtrado", mask3, 'tr')
            # show("Erosionado", mask2, 'bl')
            # show("Dilatado", mask1, 'br')
            # show("Rellenado", mask, 'br')

            if poll_key():
                break
    except EndOfStreamException:
        print("End of Stream")

    destroy_all_windows()
