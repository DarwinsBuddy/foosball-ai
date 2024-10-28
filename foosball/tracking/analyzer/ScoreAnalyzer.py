import datetime as dt
import multiprocessing
import traceback
from dataclasses import dataclass
from typing import Optional

from . import AbstractAnalyzer
from ...hooks import AudioHook, Webhook
from ...models import Team, Goals, Score, Track, Info, Verbosity, Info
from ...pipe.BaseProcess import Msg
from ...utils import contains


@dataclass
class ScoreAnalyzerResult:
    score: Score
    team_scored: Team


class ScoreAnalyzer(AbstractAnalyzer):
    def close(self):
        pass

    def __init__(self, goal_grace_period_sec: float = 1.0, audio: bool = False, webhook: bool = False, *args, **kwargs):
        super().__init__(name="ScoreAnalyzer")
        self.kwargs = kwargs
        self.goal_grace_period_sec = goal_grace_period_sec
        self.score = Score()
        self.score_reset = multiprocessing.Event()
        self.last_track_sighting: dt.datetime | None = None
        self.last_track: Optional[Track] = None
        self.goal_candidate = None
        self.hooks = [AudioHook("goal"), Webhook.load_webhook('goal_webhook.yaml')]

    @staticmethod
    def is_track_empty(track: Track):
        return len([x for x in track if x is not None]) == 0

    @staticmethod
    def is_track_about_to_vanish(track: Track):
        return len([x for x in track if x is not None]) == 1

    def goal_shot(self, goals: Goals, track: Track) -> Optional[Team]:
        # current track is empty but last track had one single point left
        try:
            if self.is_track_empty(track) and self.is_track_about_to_vanish(self.last_track):
                if contains(goals.left.bbox, self.last_track[-1]):
                    return Team.BLUE
                elif contains(goals.right.bbox, self.last_track[-1]):
                    return Team.RED
        except Exception as e:
            self.logger.error(f"Error {e}")
        return None

    def analyze(self, msg: Msg, timestamp: dt.datetime) -> [ScoreAnalyzerResult, [Info]]:
        goals = msg.data["Tracker"].goals
        track = msg.data["Tracker"].ball_track
        team_scored = None
        try:
            self.check_reset_score()
            now = timestamp  # take frame timestamp as now instead of dt.datetime.now (to prevent drift due to pushing/dragging pipeline)
            no_track_sighting_in_grace_period = (now - self.last_track_sighting).total_seconds() >= self.goal_grace_period_sec if self.last_track_sighting is not None else None
            if not self.is_track_empty(track):
                # track is not empty, so we save our state and remove a potential goal (which was wrongly tracked)
                # case1: detected goals where not accurate => false positive
                # case2: ball jumped right back into field => false positive
                self.last_track_sighting = now
                self.goal_candidate = None
            else:
                # let's wait for track (s.a.), or we run out of grace period (down below)
                # whatever happens first
                if self.goal_candidate is not None and self.last_track_sighting is not None and no_track_sighting_in_grace_period:
                    self.count_goal(self.goal_candidate, timestamp)
                    team_scored = self.goal_candidate
                    self.goal_candidate = None
                elif self.goal_candidate is None:
                    # if track is empty, and we have no current goal candidate, check if there is one
                    self.goal_candidate = self.goal_shot(goals, track) if None not in [goals, track, self.last_track] else None
        except Exception as e:
            self.logger.error("Error in analyzer ", e)
            traceback.print_exc()
        self.last_track = track
        return [
            ScoreAnalyzerResult(score=self.score, team_scored=team_scored),
            [Info(verbosity=Verbosity.INFO, title="Score", value=self.score.to_string())]
        ]

    def check_reset_score(self):
        if self.score_reset.is_set():
            self.score.reset()
            self.score_reset.clear()

    def count_goal(self, team: Team, timestamp: dt.datetime):
        self.score.inc(team)
        for h in self.hooks:
            try:
                h.invoke({"team": team.value}, timestamp)
            except Exception as e:
                self.logger.error("Error in Analyzer - effects ")
                print(e)
                print(type(h))
                traceback.print_exc()

    def reset(self):
        self.score_reset.set()
