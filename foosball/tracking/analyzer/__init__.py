import logging
from abc import ABC, abstractmethod
from datetime import datetime

from foosball.models import Info
from foosball.pipe.BaseProcess import Msg


class AbstractAnalyzer(ABC):

    def __init__(self, name: str = "UnknownAnalyzer", **kwargs):
        self.name = name
        self.logger = logging.getLogger(name)

    @abstractmethod
    def analyze(self, msg: Msg, timestamp: datetime) -> [dict, [Info]]:
        pass

    @abstractmethod
    def reset(self):
        pass
