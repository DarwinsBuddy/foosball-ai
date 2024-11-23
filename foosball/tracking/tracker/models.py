from dataclasses import dataclass

from foosball.models import Frame, Goals, Track, Blob


@dataclass
class TrackerResult:
    frame: Frame
    goals: Goals | None
    ball_track: Track | None
    ball: Blob | None
    viewbox: list | None
