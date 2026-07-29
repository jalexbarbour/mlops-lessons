from __future__ import annotations

import pytest

from draft_tool.draft import DraftState
from draft_tool.league import LeagueSettings
from draft_tool.models import Player, Position


def make_player(id, name, position, bye_week=7, rank=1, proj_points=100.0, team="XX"):
    return Player(id=id, name=name, position=Position(position), team=team, bye_week=bye_week, rank=rank, proj_points=proj_points)


@pytest.fixture
def players():
    return [
        make_player("1", "Alpha Runner", "RB", rank=1),
        make_player("2", "Bravo Receiver", "WR", rank=2),
        make_player("3", "Charlie Passer", "QB", rank=3),
        make_player("4", "Delta Runner", "RB", rank=4),
    ]


@pytest.fixture
def state(players):
    league = LeagueSettings(num_teams=4)
    return DraftState(players, league, my_team_id=1)


def test_snake_draft_order():
    # 4 teams, 2 rounds: round 1 forward (1,2,3,4), round 2 reversed (4,3,2,1)
    players = [make_player(str(i), f"Player {i}", "RB", rank=i) for i in range(1, 9)]
    league = LeagueSettings(num_teams=4)
    state = DraftState(players, league, my_team_id=1)

    expected = [1, 2, 3, 4, 4, 3, 2, 1]
    seen = []
    for _ in expected:
        seen.append(state.current_team_id())
        state.draft(state.available_players()[0])
    assert seen == expected


def test_draft_removes_from_available_and_adds_to_roster(state, players):
    target = players[0]
    team_id = state.draft(target)
    assert team_id == 1
    assert target not in state.available_players()
    assert target in state.rosters[1].players


def test_cannot_draft_same_player_twice(state, players):
    target = players[0]
    state.draft(target)
    with pytest.raises(ValueError):
        state.draft(target)


def test_undo_reverses_last_pick(state, players):
    target = players[0]
    state.draft(target)
    state.undo()
    assert target in state.available_players()
    assert target not in state.rosters[1].players
    assert state.pick_number == 1


def test_undo_with_no_picks_raises(state):
    with pytest.raises(ValueError):
        state.undo()


def test_find_prefers_exact_match(state):
    matches = state.find("Alpha Runner")
    assert [p.id for p in matches] == ["1"]


def test_find_substring_match(state):
    matches = state.find("runner")
    assert {p.id for p in matches} == {"1", "4"}


def test_find_no_match(state):
    assert state.find("nobody") == []


def test_is_my_turn(state):
    assert state.is_my_turn()
    state.draft(state.available_players()[0])
    assert not state.is_my_turn()
