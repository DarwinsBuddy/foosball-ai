import logging
import traceback
from typing import Optional

from ..models import Team, Goals, Score, AnalyzeResult, Track
from ..pipe.BaseProcess import BaseProcess, Msg
from ..utils import contains

class Analyzer(BaseProcess):
    def close(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(name="Analyzer")
        self.kwargs = kwargs
        self.score = Score()
        self.last_track: Optional[Track] = None

    def goal_shot(self, goals: Goals, track: Track) -> Optional[Team]:
        # current track is empty but last track had one single point left
        try:
            if len([x for x in track if x is not None]) == 0 and len([x for x in self.last_track if x is not None]) == 1:
                if contains(goals.left.bbox, self.last_track[-1]):
                    return Team.BLUE
                elif contains(goals.right.bbox, self.last_track[-1]):
                    return Team.RED
        except Exception as e:
            self.logger.error(f"Error {e}")
            self.logger.error(f"self.last_track: [{' '.join([f'{x}' for x in self.last_track])}]")
        return None

    def process(self, msg: Msg) -> Msg:
        track_result = msg.kwargs['result']
        goals = track_result.goals
        ball = track_result.ball
        track = track_result.ball_track
        frame = track_result.frame
        info = track_result.info
        try:
            team = self.goal_shot(goals, track) if None not in [goals, track, self.last_track] else None
            self.score.inc(team)
            if team is not None:
                self.logger.info(f"GOAL Team:{team} - {self.score.red} : {self.score.blue}")
        except Exception as e:
            self.logger.error("Error in analyzer ", e)
            traceback.print_exc()
        self.last_track = track
        return Msg(kwargs={"result": AnalyzeResult(score=self.score, ball=ball, goals=goals, frame=frame, info=info, ball_track=track)})
