import logging
import signal
from abc import abstractmethod
from multiprocessing import Process

import pypeln.process as pl
from pypeln.process.stage import Stage

class Pipeline:
    def __init__(self):
        self.p = None
        self.running = False
        self.stopped = False
        def signal_handler(sig, frame):
            print('\n\nExiting...')
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        self.build()

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

    def start(self):
        if not self.running and not self.stopped:
            logging.debug(f'Starting {self.__class__.__name__}...')
            self.p.start()
            self.running = True
        elif self.running:
            print(f'Cannot start {self.__class__.__name__} - call build() before invoking start()')
        else:
            print(f'Cannot start {self.__class__.__name__} twice')

    @abstractmethod
    def _build(self, *args, **kwargs) -> Stage:
        pass

    @abstractmethod
    def _stop(self):
        pass

    @staticmethod
    def dev_null(*args, **kwargs):
        return False

    @staticmethod
    def out(x):
        try:
            print(x)
        except Exception as e:
            logging.error(e)
            return None

class TestPipeline(Pipeline):
    input = pl.IterableQueue()
    # output = pl.IterableQueue()
    def __init__(self):
        super().__init__()

    @staticmethod
    def add1(x):
        try:
            return x + 1
        except Exception as e:
            logging.error(e)
            return None
    @staticmethod
    def times2(x):
        try:
            return x * 2
        except Exception as e:
            logging.error(e)
            return None

    def _stop(self):
        self.input.stop()
        self.input.close()
    def _build(self):
        pipe = (
                self.input
                | pl.map(self.add1, workers=1)
                | pl.map(self.times2, workers=1)
                | pl.map(self.out, workers=1)
                | pl.filter(self.dev_null, workers=1)
                | list
                )

        logging.debug(f"End of pipe: {pipe}")
        return pipe
