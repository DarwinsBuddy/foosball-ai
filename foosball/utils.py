import cv2

all_windows = []


def title(s):
    print(f'{"=" * 8} {s} {"=" * 8}')


def destroy_all_windows():
    cv2.destroyAllWindows()
    all_windows.clear()


def show(name, frame, pos='tl'):
    [x, y] = {
        'tl': [10, 0],
        'tr': [1310, 0],
        'bl': [10, 900],
        'br': [1310, 900]
    }[pos]

    if name not in all_windows:
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, y)
        all_windows.append(name)
    cv2.imshow(name, frame)