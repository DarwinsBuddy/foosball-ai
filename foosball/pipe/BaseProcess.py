import abc
import dataclasses
import logging
import multiprocessing
import time
import traceback
from queue import Empty

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
    def __init__(self, *args, **kwargs):
        super().__init__(daemon=True, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.stop_event: multiprocessing.Event = multiprocessing.Event()

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
        pass # self.stop_event.set()

    def run(self):
        assert self.inq is not None
        assert self.outq is not None
        logging.debug(f"Starting {self._name}")
        while not self.stop_event.is_set():
            try:
                msg = self.inq.get_nowait()
                if msg is SENTINEL:
                    break
                out = self.process(msg)
                self.outq.put_nowait(out)
            except Empty:
                pass
            except Exception as e:
                logging.error(f"Error in {self._name} - {e}")
                traceback.print_exc()
        logging.debug(f"Stopping {self._name}...")
        self.outq.put_nowait(SENTINEL)
        clear(self.inq)
        # TODO: come up with a better way of ensuring that downstream process
        #       drains this outq (their inq) after receiving the SENTINEL poison pill
        time.sleep(1)
        clear(self.outq)
        self.close()
        logging.debug(f"Stopped {self._name}")
