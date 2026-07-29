from __future__ import annotations

from dataclasses import dataclass, field

from .league import LeagueSettings
from .models import Player, Position


@dataclass
class Roster:
    team_id: int
    players: list[Player] = field(default_factory=list)

    def count(self, position: Position) -> int:
        return sum(1 for p in self.players if p.position == position)

    def count_on_bye(self, position: Position, bye_week: int) -> int:
        return sum(1 for p in self.players if p.position == position and p.bye_week == bye_week)


class DraftState:
    """Tracks a live snake draft: who is on the clock, what's been picked,
    and every team's roster so far."""

    def __init__(self, players: list[Player], league: LeagueSettings, my_team_id: int = 1):
        if not 1 <= my_team_id <= league.num_teams:
            raise ValueError(f"my_team_id must be between 1 and {league.num_teams}")
        self.league = league
        self.my_team_id = my_team_id
        self._available: dict[str, Player] = {p.id: p for p in players}
        self.rosters: dict[int, Roster] = {tid: Roster(tid) for tid in range(1, league.num_teams + 1)}
        self.pick_history: list[tuple[int, int, Player]] = []  # (pick_no, team_id, player)

    @property
    def pick_number(self) -> int:
        return len(self.pick_history) + 1

    @property
    def round_number(self) -> int:
        return (self.pick_number - 1) // self.league.num_teams + 1

    def current_team_id(self) -> int:
        """Team on the clock for the next pick, accounting for snake order."""
        n = self.league.num_teams
        round_idx = (self.pick_number - 1) // n
        slot_idx = (self.pick_number - 1) % n
        if round_idx % 2 == 1:
            slot_idx = n - 1 - slot_idx
        return slot_idx + 1

    def is_my_turn(self) -> bool:
        return self.current_team_id() == self.my_team_id

    def available_players(self) -> list[Player]:
        return list(self._available.values())

    def find(self, query: str) -> list[Player]:
        """Look up available players by name. Prefers an exact (case
        insensitive) match; otherwise returns all substring matches."""
        q = query.strip().lower()
        if not q:
            return []
        exact = [p for p in self._available.values() if p.name.lower() == q]
        if exact:
            return exact
        return [p for p in self._available.values() if q in p.name.lower()]

    def draft(self, player: Player, team_id: int | None = None) -> int:
        """Record a pick. Defaults to whichever team is on the clock."""
        if player.id not in self._available:
            raise ValueError(f"{player.name} has already been drafted")
        team_id = team_id if team_id is not None else self.current_team_id()
        if team_id not in self.rosters:
            raise ValueError(f"no such team: {team_id}")
        self.pick_history.append((self.pick_number, team_id, player))
        self.rosters[team_id].players.append(player)
        del self._available[player.id]
        return team_id

    def undo(self) -> None:
        if not self.pick_history:
            raise ValueError("no picks to undo")
        _, team_id, player = self.pick_history.pop()
        self.rosters[team_id].players.remove(player)
        self._available[player.id] = player

    def my_roster(self) -> Roster:
        return self.rosters[self.my_team_id]
