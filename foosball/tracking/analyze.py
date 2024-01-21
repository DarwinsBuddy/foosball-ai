import multiprocessing
import traceback
from typing import Optional

from .. import hooks
from ..hooks import generate_goal_webhook
from ..models import Team, Goals, Score, AnalyzeResult, Track, Info, Verbosity
from ..pipe.BaseProcess import BaseProcess, Msg
from ..utils import contains


class Analyzer(BaseProcess):
    def close(self):
        pass

    def __init__(self, audio: bool = False, webhook: bool = False, *args, **kwargs):
        super().__init__(name="Analyzer")
        self.kwargs = kwargs
        self.score = Score()
        self.score_reset = multiprocessing.Event()
        self.audio = audio
        self.webhook = webhook
        self.last_track: Optional[Track] = None

    def goal_shot(self, goals: Goals, track: Track) -> Optional[Team]:
        # current track is empty but last track had one single point left
        try:
            if len([x for x in track if x is not None]) == 0 and len(
                    [x for x in self.last_track if x is not None]) == 1:
                if contains(goals.left.bbox, self.last_track[-1]):
                    return Team.BLUE
                elif contains(goals.right.bbox, self.last_track[-1]):
                    return Team.RED
        except Exception as e:
            self.logger.error(f"Error {e}")
        return None

    def call_hooks(self, team: Team) -> None:
        if self.audio:
            hooks.play_random_sound('goal')
        if self.webhook:
            hooks.webhook(generate_goal_webhook(team))

    def process(self, msg: Msg) -> Msg:
        track_result = msg.kwargs['result']
        goals = track_result.goals
        ball = track_result.ball
        track = track_result.ball_track
        frame = track_result.frame
        info = track_result.info
        if self.score_reset.is_set():
            self.score.reset()
            self.score_reset.clear()
        try:
            team: Team = self.goal_shot(goals, track) if None not in [goals, track, self.last_track] else None
            self.score.inc(team)
            if team is not None:
                self.logger.info(f"GOAL Team:{team} - {self.score.red} : {self.score.blue}")
                self.call_hooks(team)
        except Exception as e:
            self.logger.error("Error in analyzer ", e)
            traceback.print_exc()
        self.last_track = track
        info.append(Info(verbosity=Verbosity.INFO, title="Score", value=self.score.to_string()))
        return Msg(kwargs={"result": AnalyzeResult(score=self.score, ball=ball, goals=goals, frame=frame, info=info,
                                                   ball_track=track)})

    def reset_score(self):
        self.score_reset.set()
