from abc import abstractmethod
from multiprocessing import Queue
from queue import Full
from threading import Thread

from foosball.pipe.BaseProcess import Msg


class Stream(Thread):

    def __init__(self, maxsize=128, skip_frames=True, timeout=2, *args, **kwargs):
        super().__init__(daemon=True, *args, **kwargs)
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.timeout = timeout
        self.skip_frames = skip_frames
        self.skipped_frames = 0
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=maxsize)

    @abstractmethod
    def is_eos(self):
        return False

    @abstractmethod
    def next_frame(self):
        pass

    def read_frame(self):
        # grab the current frame
        flag, frame = self.next_frame()

        while not flag and self.skip_frames:
            if self.is_eos():
                return None, None
            if self.skipped_frames == 0:
                print("Skipping frame(s)")
            self.skipped_frames += 1
            flag, frame = self.next_frame()
        if self.skipped_frames > 0:
            print("Resuming video...")
            self.skipped_frames = 0

        return flag, frame

    @property
    def output(self):
        return self.Q

    def send_frame(self, frame):
        msg = Msg(kwargs={'frame': frame}) if frame is not None else None
        while True:
            try:
                # try to put it into the queue
                self.Q.put(msg, True, 0.5)
                break
            except Full:
                print("Queue is full")
                if self.stopped:
                    break

    def run(self):
        while not self.stopped:
            # read the next frame from the file
            grabbed, frame = self.read_frame()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if grabbed is None or not grabbed:
                print("Could not grab")
                self.stopped = True
                frame = None
            self.send_frame(frame)
        print("Release")
        self.close_capture()

    @abstractmethod
    def close_capture(self):
        pass

    @abstractmethod
    def dim(self) -> [int, int]:
        pass

    def stop(self):
        self.stopped = True
        return self
