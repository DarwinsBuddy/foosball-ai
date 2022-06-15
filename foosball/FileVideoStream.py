from threading import Thread
import cv2
import time
from queue import Queue, Full


class EndOfStreamException(Exception):
    pass


class FileVideoStream:

    def __init__(self, path, transform=None, queue_size=128, skip_frames=True):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.skip_frames = skip_frames
        self.stream = cv2.VideoCapture(path)
        self.total_frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.skipped_frames = 0
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # initialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def check_eos(self):
        if self.stream.get(cv2.CAP_PROP_POS_FRAMES) == self.total_frames:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            self.stop()
            raise EndOfStreamException("End of stream")

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def get_frame(self):
        # grab the current frame
        flag, frame = self.stream.read()

        while not flag and self.skip_frames:
            self.check_eos()
            if self.skipped_frames == 0:
                print(f"Skipping frame(s)")
            self.skipped_frames += 1
            flag, frame = self.stream.read()
        if self.skipped_frames > 0:
            print(f"Resuming video...")
            self.skipped_frames = 0

        return flag, frame

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break
            try:
                # read the next frame from the file
                grabbed, frame = self.get_frame()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    print("Could not grab")
                    self.stopped = True
                    raise EndOfStreamException()

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. i.e. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are typically OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)
                while True:
                    try:
                        # try to put it into the queue
                        self.Q.put(frame, True, 0.5)
                        break
                    except Full:
                        print("Queue is full")
                        if self.stopped:
                            print("Stopped")
                            raise EndOfStreamException()
            except EndOfStreamException:
                print("End stream")
                self.stopped = True
                break
        print("Release")
        self.stream.release()

    def dim(self):
        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return [width, height]

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
