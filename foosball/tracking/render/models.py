from dataclasses import dataclass
from typing import Optional

from foosball.models import Frame


@dataclass
class RendererResult:
    frame: Optional[Frame]
