from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from foosball.models import Frame

DetectorResult = TypeVar('DetectorResult')


class Detector(ABC, Generic[DetectorResult]):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def detect(self, frame: Frame) -> DetectorResult | None:
        pass
