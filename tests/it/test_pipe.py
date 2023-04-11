import logging
from queue import Empty

import pypeln as pl
import time

from foosball.pipe import Pipeline, out


class TestPipeline(Pipeline):
    def _start(self):
        pass

    input = pl.process.IterableQueue()
    output = pl.process.IterableQueue()

    def __init__(self):
        super().__init__()

    @staticmethod
    def add(n):
        def f(x):
            try:
                return x + n
            except Exception as e:
                logging.error(e)
                return None

        return f

    @staticmethod
    def times(mul: int):
        def f(x):
            try:
                return x * mul
            except Exception as e:
                logging.error(e)
                return None

        return f

    def _stop(self):
        self.input.stop()
        self.output.stop()

    def out(self, x: int):
        out(x)
        self.output.put_nowait(x)
        return x

    def _build(self):
        added = pl.process.map(self.add(1), self.input)
        times2 = pl.process.map(self.times(-2), added, workers=1)
        times3 = pl.process.map(self.times(2), added, workers=1)
        times = pl.process.concat([times2, times3])
        as_list = list(times)
        pipe = list(pl.thread.map(self.out, as_list))


        logging.debug(f"End of pipe: {pipe}")
        return pipe


def test_pipeline():
    p = TestPipeline()
    p.build()
    p.start()
    p.input.put(2)
    p.input.put(3)
    time.sleep(2)
    p.stop()
    result = []
    while True:
        try:
            x = p.output.get_nowait()
            result.append(x)
        except Empty:
            break
    assert result == [6, -6, 8, -8]
