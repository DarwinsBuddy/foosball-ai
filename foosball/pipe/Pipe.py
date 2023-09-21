import logging
from multiprocessing import Queue
from queue import Empty

from foosball.pipe import BaseProcess


SENTINEL = None

def clear(q: Queue):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass


class Pipe:
    def __init__(self, processes: list[BaseProcess]):
        assert len(processes) > 0
        self.processes: list[BaseProcess] = processes
        self.queues = [Queue() for i in range(0, len(processes) + 1)]
        self.build()

    def build(self):
        for i in range(0, len(self.processes)):
            self.processes[i].set_input(self.queues[i])
            self.processes[i].set_output(self.queues[i + 1])

    def start(self):
        logging.debug("Starting pipe...")
        for p in self.processes:
            p.start()
        logging.debug("Started pipe")

    @property
    def output(self):
        return self.queues[-1]

    @property
    def input(self):
        return self.queues[0]

    def stop(self):
        logging.debug("Stopping pipe...")
        logging.debug("stopping...")
        #for p in self.processes:
        #    p.stop()
        self.queues[0].put_nowait(SENTINEL)
        logging.debug("draining...")
        #for q in self.queues:
        #    clear(q)
        #    q.close()
        logging.debug("joining...")
        clear(self.queues[-1])
        for p in self.processes:
            p.join()

        logging.debug("Stopped pipe")
