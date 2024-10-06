import traceback

from .ScoreAnalyzer import ScoreAnalyzer
from ... import hooks
from ...hooks import generate_goal_webhook
from ...models import Team, TrackerResult
from ...pipe.BaseProcess import BaseProcess, Msg


class Analyzer(BaseProcess):
    def close(self):
        pass

    def __init__(self, audio: bool = False, webhook: bool = False, goal_grace_period_sec: float = 1.0, *args, **kwargs):
        super().__init__(name="Analyzer")
        self.kwargs = kwargs
        self.analyzers = [ScoreAnalyzer(goal_grace_period_sec, args, kwargs)]
        # TODO: catch up here
        self.effects = []
        self.audio = audio
        self.webhook = webhook

    def call_hooks(self, team: Team) -> None:
        if self.audio:
            hooks.play_random_sound('goal')
        if self.webhook:
            hooks.webhook(generate_goal_webhook(team))

    def process(self, msg: Msg) -> Msg:
        track_result: TrackerResult = msg.kwargs['Tracker']
        data = track_result.data

        for a in self.analyzers:
            try:
                ar = a.analyze(track_result, msg.timestamp)
                msg.add(a.name, ar, info=ar.info)
            except Exception as e:
                self.logger.error("Error in Analyzer - analyzers ", e)
                traceback.print_exc()
        # TODO: catch up here
        for e in self.effects:
            try:
                e.invoke(track_result, msg.timestamp)
            except Exception as e:
                self.logger.error("Error in Analyzer - effects ", e)
                traceback.print_exc()
        return msg
