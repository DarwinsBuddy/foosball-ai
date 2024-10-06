import multiprocessing
import traceback
from typing import Optional
import datetime as dt

from .. import hooks
from ..hooks import generate_goal_webhook
from ..models import Team, Goals, Score, AnalyzeResult, Track, Info, Verbosity
from ..pipe.BaseProcess import BaseProcess, Msg
from ..utils import contains


class Analyzer(BaseProcess):
    def close(self):
        pass

    def __init__(self, audio: bool = False, webhook: bool = False, goal_grace_period_sec: float = 1.0, *args, **kwargs):
        super().__init__(name="Analyzer")
        self.kwargs = kwargs
        self.goal_grace_period_sec = goal_grace_period_sec
        self.score = Score()
        self.score_reset = multiprocessing.Event()
        self.audio = audio
        self.webhook = webhook
        self.last_track_sighting: dt.datetime | None = None
        self.last_track: Optional[Track] = None
        self.goal_candidate = None

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
        try:
            self.check_reset_score()
            now = dt.datetime.now()
            # TODO: define goal grace period in seconds of runtime not seconds in rendering!
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
                    self.count_goal(self.goal_candidate)
                    self.goal_candidate = None
                elif self.goal_candidate is None:
                    # if track is empty, and we have no current goal candidate, check if there is one
                    self.goal_candidate = self.goal_shot(goals, track) if None not in [goals, track, self.last_track] else None
        except Exception as e:
            self.logger.error("Error in analyzer ", e)
            traceback.print_exc()
        self.last_track = track
        info.append(Info(verbosity=Verbosity.INFO, title="Score", value=self.score.to_string()))
        return Msg(kwargs={"result": AnalyzeResult(score=self.score, ball=ball, goals=goals, frame=frame, info=info,
                                                   ball_track=track)})

    def check_reset_score(self):
        if self.score_reset.is_set():
            self.score.reset()
            self.score_reset.clear()

    def count_goal(self, team: Team):
        self.score.inc(team)
        if team is not None:
            self.logger.info(f"GOAL Team:{team} - {self.score.red} : {self.score.blue}")
            self.call_hooks(team)

    def reset_score(self):
        self.score_reset.set()
