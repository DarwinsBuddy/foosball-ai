from abc import ABC, abstractmethod


class Hook(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass
