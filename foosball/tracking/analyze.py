from collections import deque
from typing import Optional

from .models import TrackResult, Team, Goals, Score, AnalyzeResult, Track
from .utils import contains


class Analyzer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.score = Score()
        self.last_track: Track = None

    def stop(self) -> None:
        pass

    def goal_shot(self, goals: Goals, track: Track) -> Optional[Team]:
        # current track is empty but last track had one single point left
        if track is not None and len([x for x in track if x is not None]) == 0 and self.last_track is not None and len([x for x in self.last_track if x is not None]) == 1:
            if contains(goals.left.bbox, self.last_track[-1]):
                return Team.BLUE
            elif contains(goals.right.bbox, self.last_track[-1]):
                return Team.RED
        return None

    def analyze(self, track_result: TrackResult) -> AnalyzeResult:
        goals = track_result.goals
        ball = track_result.ball
        track = track_result.ball_track
        frame = track_result.frame
        info = track_result.info
        try:

            team = self.goal_shot(goals, track)
            if team is not None:
                print(f"GOOOOOOAL!!! {team}", end="\n\n")
            self.score.inc(team)
            self.last_track = track
        except Exception as e:
            print("Error in analyzer ", e)
        return AnalyzeResult(score=self.score, ball=ball, goals=goals, frame=frame, info=info, ball_track=track)