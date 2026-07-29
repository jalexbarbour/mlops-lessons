from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"


# Positions that can fill a FLEX roster slot.
FLEX_ELIGIBLE = {Position.RB, Position.WR, Position.TE}


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    position: Position
    team: str
    bye_week: int
    rank: int  # overall expert rank, 1 = best player in the pool
    proj_points: float  # projected season fantasy points

    def __str__(self) -> str:
        return f"{self.name} ({self.position.value}, {self.team}, bye {self.bye_week})"
