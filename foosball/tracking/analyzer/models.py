from dataclasses import dataclass

from foosball.models import Goals, Track, Blob, Score, FrameDimensions


@dataclass
class FrameStats:
    goals: Goals | None
    track: Track | None
    ball: Blob | None
    score: Score | None
    dims: FrameDimensions | None
    viewbox: list | None
    timestamp: str | None

    def to_dict(self) -> dict:
        return {
            "goals": self.goals.to_json() if self.goals else None,
            "track": self.track.to_json() if self.track else None,
            "ball": self.ball.to_json() if self.ball else None,
            "score": self.score.to_json() if self.score else None,
            "dims": self.dims.to_json() if self.dims else None,
            "viewbox": self.viewbox if self.viewbox else None,
            "timestamp": self.timestamp
        }