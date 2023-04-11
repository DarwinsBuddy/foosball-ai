import logging
import signal
from abc import abstractmethod
from multiprocessing import Process

import pypeln as pl

class Pipeline:
    def __init__(self):
        self.p = None
        self.running = False
        self.stopped = False
        def signal_handler(sig, frame):
            print('\n\nExiting...')
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)

    def stop(self, timeout=5):
        self._stop()
        if self.p is not None:
            logging.debug(f'Stopping {self.__class__.__name__}...')
            self.p.join(timeout=timeout)
            self.running = False
            self.stopped = True
            logging.debug(f'Stopped {self.__class__.__name__}')

    def build(self, *args, **kwargs):
        if self.p is None:
            logging.debug(f'Building {self.__class__.__name__}...')
            self.p = Process(target=self._build, args=args, kwargs=kwargs)
        else:
            print('Pipeline already built')
        return self.p

    def _start(self):
        pass
    def start(self):
        self._start()
        if not self.running and not self.stopped:
            logging.debug(f'Starting {self.__class__.__name__}...')
            self.p.start()
            self.running = True
        elif self.running:
            print(f'Cannot start {self.__class__.__name__} - call build() before invoking start()')
        else:
            print(f'Cannot start {self.__class__.__name__} twice')

    @abstractmethod
    def _build(self, *args, **kwargs) -> pl.utils.BaseStage:
        pass

    def _stop(self):
        pass

def dev_null():
    return pl.thread.filter(lambda: False)

def out(x):
    try:
        print(x)
        return x
    except Exception as e:
        logging.error(e)
        return None

def tap(c: callable):
    def f(x):
        c(x)
        return x
    return f

def tee(queue: pl.process.IterableQueue):
    return tap(lambda x: queue.put(x))

def split(n: int, ctor):
    def _split(x):
        outs = []
        for i in range(0, n):
            o = ctor()
            o.put(x)
            outs.append(o)
        return outs
    return _split
