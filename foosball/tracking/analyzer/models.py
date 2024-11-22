from dataclasses import dataclass

from foosball.models import Goals, Track, Blob, Score, FrameDimensions


@dataclass
class FrameStats:
    goals: Goals | None
    track: Track | None
    ball: Blob | None
    score: Score | None
    dims: FrameDimensions | None
