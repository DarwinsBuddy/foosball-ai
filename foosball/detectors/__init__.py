from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from foosball.models import Frame, DetectedBall, DetectedGoals

DetectorResult = TypeVar('DetectorResult')
DetectorConfig = TypeVar('DetectorConfig')
BallConfig = TypeVar('BallConfig')
GoalConfig = TypeVar('GoalConfig')


class Detector(ABC, Generic[DetectorConfig, DetectorResult]):
    def __init__(self, config: DetectorConfig, *args, **kwargs):
        self.config = config

    @abstractmethod
    def detect(self, frame: Frame) -> DetectorResult | None:
        pass


class BallDetector(Generic[BallConfig], Detector[BallConfig, DetectedBall], ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GoalDetector(Generic[GoalConfig], Detector[GoalConfig, DetectedGoals], ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)