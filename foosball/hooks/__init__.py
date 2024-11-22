from abc import ABC, abstractmethod


class Hook(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self, *args, **kwargs):
        pass
