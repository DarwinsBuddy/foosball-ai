import logging
from multiprocessing import Queue
from queue import Empty

from foosball.pipe import BaseProcess

SENTINEL = None


def clear(q: Queue):
    try:
        while True:
            q.get(block=True, timeout=1)
    except Empty:
        pass


class Pipe:
    def __init__(self, stream, processes: list[BaseProcess], maxsize=128):
        assert len(processes) > 0
        self.logger = logging.getLogger("Pipe")
        self.processes: list[BaseProcess] = processes
        self.stream = stream
        self.queues = [self.stream.output] + [Queue(maxsize=maxsize) for i in range(0, len(processes))]
        self.build()

    def build(self):
        for i in range(0, len(self.processes)):
            self.processes[i].set_input(self.queues[i])
            self.processes[i].set_output(self.queues[i + 1])

    def start(self):
        self.logger.debug("Starting pipe...")
        for p in self.processes:
            p.start()
        self.stream.start()
        self.logger.debug("Started  pipe")

    @property
    def output(self):
        return self.queues[-1]

    @property
    def input(self):
        return self.queues[0]

    @property
    def qsizes(self):
        return [q.qsize() for q in self.queues]

    def stop(self):
        self.logger.debug("Stopping pipe...")
        self.stream.stop()
        for p in self.processes:
            p.stop()
        # empty the last queue

        self.logger.debug(f"Queue sizes: {' '.join([str(s) for s in self.qsizes])}")
        self.logger.debug("draining queues...")
        # draining all queues for good
        for q in self.queues:
            clear(q)
        self.logger.debug("joining...")
        for p in reversed(self.processes):
            p.join()
        self.logger.debug(f"Queue sizes: {' '.join([str(s) for s in self.qsizes])}")
        self.logger.debug("Stopped  pipe")
