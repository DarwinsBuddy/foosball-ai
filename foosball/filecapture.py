from imutils.video import VideoStream, FPS

from .FileVideoStream import FileVideoStream


class FileCapture:

    def __init__(self, video=None):
        # if a video path was not supplied, grab the reference
        # to the webcam
        if video is None:
            self.cap = VideoStream(src=1).start()
        # otherwise, grab a reference to the video file
        else:
            # #vs = FileVideoStream(args['file']).start()
            self.cap = FileVideoStream(video).start()

        self.fps_cap = FPS().start()
        self.is_file_capture = video is not None

    def next(self):
        self.fps_cap.update()
        self.fps_cap.stop()
        return self.cap.read()

    def stop(self):
        self.cap.stop()

    def dim(self):
        return self.cap.dim()

    def fps(self):
        return self.fps_cap.fps()
