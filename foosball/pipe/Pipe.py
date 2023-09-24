import logging
from multiprocessing import Queue
from queue import Empty

from foosball.pipe import BaseProcess

SENTINEL = None


def clear(q: Queue):
    try:
        while True:
            q.get(block=True, timeout=0.1)
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
        self.queues[0].put_nowait(SENTINEL)
        self.logger.debug("joining...")
        for i in range(0, len(self.processes) - 1):
            p = self.processes[i]
            p.join()
            self.logger.debug(f"joined   {p.name}")
        # empty the last queue
        clear(self.queues[-1])
        # join the last process
        self.processes[-1].join()
        self.logger.debug(f"joined   {self.processes[-1].name}")
        self.logger.debug("Stopped  pipe")
