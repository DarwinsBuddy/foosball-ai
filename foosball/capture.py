import cv2
from imutils.video import VideoStream, FPS


class EndOfStreamException(Exception):
    pass


class Capture:

    def __init__(self, video=None):
        # if a video path was not supplied, grab the reference
        # to the webcam
        if video is None:
            self.cap = VideoStream(src=1).start()
        # otherwise, grab a reference to the video file
        else:
            # #vs = FileVideoStream(args['file']).start()
            self.cap = cv2.VideoCapture(video)
            while not self.cap.isOpened():
                self.cap = cv2.VideoCapture(video)
                cv2.waitKey(1000)
                print("Wait for the header")

        self.fps_cap = FPS().start()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.skipped_frames = 0
        self.is_file_capture = video is not None

    def check_eos(self):
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            self.stop()
            raise EndOfStreamException("End of stream")

    def get_frame(self):
        # grab the current frame
        flag, frame = self.cap.read()

        while not flag:
            self.check_eos()
            if self.skipped_frames == 0:
                print(f"Skipping frame(s)")
            self.skipped_frames += 1
            flag, frame = self.cap.read()
        if self.skipped_frames > 0:
            print(f"Resuming video...")
            self.skipped_frames = 0
        pos_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        return pos_frame + 1, frame

    def next(self):
        self.fps_cap.update()
        self.fps_cap.stop()
        return self.get_frame()[1]

    def stop(self):
        # if we are not using a video file, stop the camera video stream
        if not self.is_file_capture:
            self.cap.stop()
        # otherwise, release the camera
        else:
            self.cap.release()

    def dim(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return [width, height]

    def fps(self):
        return self.fps_cap.fps()
