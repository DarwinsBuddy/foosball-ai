import logging
from multiprocessing import Queue
from queue import Empty

from foosball.pipe import BaseProcess

SENTINEL = None


def clear(q: Queue):
    try:
        while True:
            q.get(block=True, timeout=0.01)
    except Empty:
        pass


class Pipe:
    def __init__(self, processes: list[BaseProcess]):
        assert len(processes) > 0
        self.logger = logging.getLogger("Pipe")
        self.processes: list[BaseProcess] = processes
        self.queues = [Queue() for i in range(0, len(processes) + 1)]
        self.build()

    def build(self):
        for i in range(0, len(self.processes)):
            self.processes[i].set_input(self.queues[i])
            self.processes[i].set_output(self.queues[i + 1])

    def start(self):
        self.logger.debug("Starting pipe...")
        for p in self.processes:
            p.start()
        self.logger.debug("Started  pipe")

    @property
    def output(self):
        return self.queues[-1]

    @property
    def input(self):
        return self.queues[0]

    def stop(self):
        self.logger.debug("Stopping pipe...")
        for p in self.processes:
            p.stop()
        # empty the last queue
        self.logger.debug("joining...")
        for p in reversed(self.processes):
            p.join()
        self.logger.debug("draining queues...")
        # draining all queues for good
        for q in self.queues:
            clear(q)
        self.logger.debug("Stopped  pipe")
