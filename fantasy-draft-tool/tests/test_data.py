from __future__ import annotations

from pathlib import Path

from draft_tool.data import load_players
from draft_tool.models import Position

PLAYERS_CSV = Path(__file__).resolve().parent.parent / "data" / "players.csv"


def test_load_players_reads_all_rows_and_sorts_by_rank():
    players = load_players(PLAYERS_CSV)
    assert len(players) > 100
    ranks = [p.rank for p in players]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_load_players_covers_every_position():
    players = load_players(PLAYERS_CSV)
    positions = {p.position for p in players}
    assert positions == set(Position)


def test_load_players_parses_types():
    players = load_players(PLAYERS_CSV)
    p = players[0]
    assert isinstance(p.bye_week, int)
    assert isinstance(p.rank, int)
    assert isinstance(p.proj_points, float)
