import abc
import datetime as dt
import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from queue import Empty, Full

from foosball.models import InfoLog
from foosball.pipe.Pipe import clear, SENTINEL


# TODO: check why merging into one Msg is having a huge impact on FPS
@dataclass
class Msg:
    args: list[any]
    kwargs: dict
    info: InfoLog = None
    timestamp: dt.datetime = dt.datetime.now()

    def add(self, name: str, data: any, info=InfoLog([])):
        self.kwargs[name] = data
        if self.info is not None:
            self.info.concat(info)
        else:
            self.info = InfoLog([])

    def remove(self, name) -> any:
        return self.kwargs.pop(name)

    def __init__(self, args=None, kwargs=None, timestamp=dt.datetime.now()):
        if kwargs is None:
            kwargs = dict()
        if args is None:
            args = list()
        self.kwargs = kwargs
        self.args = args
        self.timestamp = timestamp


class BaseProcess(multiprocessing.Process):
    def __init__(self, send_receive_timeout=0.5, *args, **kwargs):
        super().__init__(daemon=True, *args, **kwargs)
        self.args = args
        self.logger = logging.getLogger(kwargs.get('name') or __name__)
        self.kwargs = kwargs
        self.stopped: multiprocessing.Event = multiprocessing.Event()
        self.next_step: multiprocessing.Event = multiprocessing.Event()
        self.playing: multiprocessing.Event = multiprocessing.Event()
        self.playing.set()
        self.send_receive_timeout = send_receive_timeout
        self.inq = None
        self.outq = None

    def set_input(self, inq):
        self.inq = inq

    def set_output(self, outq):
        self.outq = outq

    @property
    def output(self):
        return self.outq

    @property
    def input(self):
        return self.inq

    @abc.abstractmethod
    def process(self, msg: Msg) -> any:
        pass

    @abc.abstractmethod
    def close(self):
        pass

    def pause(self):
        if self.playing.is_set():
            self.playing.clear()

    def resume(self):
        if not self.playing.is_set():
            self.playing.set()

    def step(self):
        self.next_step.set()

    def stop(self):
        self.stopped.set()

    def send(self, msg: Msg):
        while True:
            try:
                self.outq.put(msg, block=True, timeout=self.send_receive_timeout)
                break
            except Full:
                # print("Queue is full")
                if self.stopped.is_set():
                    break

    def receive(self) -> Msg:
        while True:
            try:
                return self.inq.get(block=True, timeout=self.send_receive_timeout)
            except Empty:
                # print("Queue is empty")
                if self.stopped.is_set():
                    break

    def run(self):
        assert self.inq is not None
        assert self.outq is not None
        self.logger.debug(f"Starting {self.name}")
        while not self.stopped.is_set():
            try:
                if self.playing.is_set() or self.next_step.is_set():
                    self.next_step.clear()
                    msg = self.inq.get_nowait()
                    if msg is SENTINEL:
                        self.logger.debug("received SENTINEL")
                        self.send(SENTINEL)
                        break
                    out = self.process(msg)
                    if out is None:
                        self.logger.debug("sending SENTINEL")
                    self.send(out)
                else:
                    self.playing.wait(timeout=0.5)
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in {self.name} - {e}")
                traceback.print_exc()
        self.logger.debug(f"Stopping {self.name}...")
        clear(self.inq)
        self.close()
        self.logger.debug(f"Stopped  {self.name}")
