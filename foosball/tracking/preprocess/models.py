from dataclasses import dataclass
from typing import Optional

import numpy as np

from foosball.models import Frame, Goals


@dataclass
class PreprocessorResult:
    original: Frame
    preprocessed: Optional[Frame]
    homography_matrix: Optional[np.ndarray]  # 3x3 matrix used to warp the image and project points
    goals: Optional[Goals]
