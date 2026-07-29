from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .draft import Roster
from .league import LeagueSettings
from .models import FLEX_ELIGIBLE, Player


@dataclass
class Suggestion:
    player: Player
    score: float
    reasons: list[str]


class SuggestionStrategy(ABC):
    """Pluggable pick-suggestion logic.

    Swap in a different subclass to change how picks are ranked (e.g. a
    VORP model, opponent-adjusted projections, or an ML-based scorer)
    without touching `DraftState` or the CLI -- both only depend on this
    interface.
    """

    @abstractmethod
    def score(self, player: Player, roster: Roster, league: LeagueSettings) -> tuple[float, list[str]]:
        """Return (score, human-readable reasons) for drafting `player`
        onto `roster`. Higher score = better suggestion."""

    def suggest(
        self,
        available: list[Player],
        roster: Roster,
        league: LeagueSettings,
        top_n: int = 5,
    ) -> list[Suggestion]:
        scored = [Suggestion(p, *self.score(p, roster, league)) for p in available]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]


class ValueNeedByeStrategy(SuggestionStrategy):
    """Baseline strategy: best-player-available, adjusted for roster need
    and bye-week stacking.

    - value: overall expert rank (lower rank number = more valuable)
    - need: bonus for starting slots not yet filled; a smaller bonus for
      flex-eligible depth once starters are set; no bonus once a position
      is well stocked
    - bye week: once you'd have as many same-position players on one bye
      week as you start at that position, you can't field a full lineup
      that week -- each additional stack past that point is penalized
    """

    def __init__(self, need_weight: float = 25.0, bye_penalty_weight: float = 20.0):
        self.need_weight = need_weight
        self.bye_penalty_weight = bye_penalty_weight

    def score(self, player: Player, roster: Roster, league: LeagueSettings) -> tuple[float, list[str]]:
        reasons = [f"rank {player.rank}, projected {player.proj_points:.0f} pts"]

        value = -float(player.rank)

        starters_needed = league.starting_slots.get(player.position, 0)
        have = roster.count(player.position)
        if have < starters_needed:
            need_bonus = self.need_weight * (starters_needed - have)
            reasons.append(f"fills starting need at {player.position.value} ({have}/{starters_needed})")
        elif player.position in FLEX_ELIGIBLE and have < starters_needed + league.flex_slots:
            need_bonus = self.need_weight * 0.5
            reasons.append(f"flex-eligible depth at {player.position.value}")
        else:
            need_bonus = 0.0

        bye_threshold = max(1, starters_needed)
        on_same_bye = roster.count_on_bye(player.position, player.bye_week)
        if on_same_bye >= bye_threshold:
            stack_over = on_same_bye - bye_threshold + 1
            bye_penalty = self.bye_penalty_weight * stack_over
            reasons.append(
                f"caution: {on_same_bye} other {player.position.value}(s) already on bye week {player.bye_week}"
            )
        else:
            bye_penalty = 0.0

        return value + need_bonus - bye_penalty, reasons
