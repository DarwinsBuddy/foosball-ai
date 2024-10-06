import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from foosball.models import TrackerResult, Score, Team, Result

@dataclass
class ScoreAnalyzerResultData:
    score: Score
    team_scored: Team


ScoreAnalyzerResult = Result[ScoreAnalyzerResultData]


class AbstractAnalyzer(ABC):

    def __init__(self, name: str = "UnknownAnalyzer", **kwargs):
        self.name = name
        self.logger = logging.getLogger(name)

    @abstractmethod
    def analyze(self, track_result: TrackerResult, timestamp: datetime) -> dict:
        pass
