from dataclasses import dataclass
import numpy as np

@dataclass
class Aruco:
    id: np.array
    corners: np.array
    rotation_vector: np.array = None
    translation_vector: np.array = None