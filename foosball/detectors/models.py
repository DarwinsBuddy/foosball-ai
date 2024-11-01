from dataclasses import dataclass
from typing import Optional

import numpy as np

from foosball.models import Goals, Blob


@dataclass
class DetectedGoals:
    goals: Optional[Goals]
    frame: np.array


@dataclass
class DetectedBall:
    ball: Optional[Blob]
    frame: np.array
