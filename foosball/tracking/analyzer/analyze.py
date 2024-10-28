import traceback

from .ScoreAnalyzer import ScoreAnalyzer
from ...pipe.BaseProcess import BaseProcess, Msg


class Analyzer(BaseProcess):
    def close(self):
        pass

    def __init__(self, audio: bool = False, webhook: bool = False, goal_grace_period_sec: float = 1.0, *args, **kwargs):
        super().__init__(name="Analyzer")
        self.kwargs = kwargs
        self.analyzers = [ScoreAnalyzer(goal_grace_period_sec, audio, webhook, args, kwargs)]
        self.audio = audio
        self.webhook = webhook

    def reset_score(self):
        for a in self.analyzers:
            if type(a) is ScoreAnalyzer:
                a.reset()

    def process(self, msg: Msg) -> Msg:
        results = {}
        infos = []
        for a in self.analyzers:
            try:
                [result, info] = a.analyze(msg, msg.timestamp)
                results[a.name] = result
                infos.extend(info)
            except Exception as e:
                self.logger.error("Error in Analyzer - analyzers ", e)
                traceback.print_exc()
        return Msg(msg=msg, info=infos, data=results)
