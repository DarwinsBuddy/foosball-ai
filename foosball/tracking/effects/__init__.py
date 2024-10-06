from abc import abstractmethod, ABC


class Effect(ABC):

    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass
