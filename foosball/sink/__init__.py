from abc import abstractmethod
from typing import Callable


class Sink:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def show(self, frame):
        pass

    @abstractmethod
    def render(self, callbacks: dict[int, Callable] = None):
        pass
