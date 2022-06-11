import cv2

UNABLE_TO_READ_NEXT_FRAME = 100

all_windows = []


def title(s):
    print(f'{"=" * 8} {s} {"=" * 8}')


def destroy_all_windows():
    cv2.destroyAllWindows()
    all_windows.clear()


def show(name, frame, pos='tl'):
    [x, y] = {
        'tl': [0, 0],
        'tr': [1280, 0],
        'bl': [0, 800],
        'br': [1280, 800]
    }[pos]

    if name not in all_windows:
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, y)
        all_windows.append(name)
    cv2.imshow(name, frame)
