from __future__ import annotations

from dataclasses import dataclass, field

from .models import Position


@dataclass
class LeagueSettings:
    """Roster shape for the league being drafted. Defaults to a standard
    12-team, single-QB league."""

    num_teams: int = 12
    starting_slots: dict[Position, int] = field(
        default_factory=lambda: {
            Position.QB: 1,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
            Position.K: 1,
            Position.DST: 1,
        }
    )
    flex_slots: int = 1
    bench_slots: int = 6

    @property
    def roster_size(self) -> int:
        return sum(self.starting_slots.values()) + self.flex_slots + self.bench_slots
