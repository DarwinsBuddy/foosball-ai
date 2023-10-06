import abc
import dataclasses
import logging
import multiprocessing
import traceback
from queue import Empty, Full

from foosball.pipe.Pipe import clear, SENTINEL


@dataclasses.dataclass
class Msg:
    args: list[any]
    kwargs: dict

    def __init__(self, args=None, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        if args is None:
            args = list()
        self.kwargs = kwargs
        self.args = args

class BaseProcess(multiprocessing.Process):
    def __init__(self, send_receive_timeout=0.5, *args, **kwargs):
        super().__init__(daemon=True, *args, **kwargs)
        self.args = args
        self.logger = logging.getLogger(kwargs.get('name') or __name__)
        self.kwargs = kwargs
        self.stop_event: multiprocessing.Event = multiprocessing.Event()
        self.send_receive_timeout = send_receive_timeout

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

    def stop(self):
        self.stop_event.set()

    def send(self, msg: Msg):
        while True:
            try:
                self.outq.put(msg, block=True, timeout=self.send_receive_timeout)
                break
            except Full:
                print("Queue is full")
                if self.stop_event.is_set():
                    break

    def receive(self) -> Msg:
        while True:
            try:
                return self.inq.get(block=True, timeout=self.send_receive_timeout)
            except Empty:
                print("Queue is empty")
                if self.stop_event.is_set():
                    break

    def run(self):
        assert self.inq is not None
        assert self.outq is not None
        self.logger.debug(f"Starting {self._name}")
        while not self.stop_event.is_set():
            try:
                msg = self.inq.get_nowait()
                if msg is SENTINEL:
                    self.logger.debug("received SENTINEL")
                    self.send(SENTINEL)
                    break
                out = self.process(msg)
                if out is None:
                    self.logger.debug("sending SENTINEL")
                self.send(out)
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in {self._name} - {e}")
                traceback.print_exc()
        self.logger.debug(f"Stopping {self._name}...")
        clear(self.inq)
        self.close()
        self.logger.debug(f"Stopped  {self._name}")
