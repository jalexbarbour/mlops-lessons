from __future__ import annotations

import csv
from pathlib import Path

from .models import Player, Position


def load_players(path: str | Path) -> list[Player]:
    """Load a player pool from a CSV with columns:
    id,name,position,team,bye_week,rank,proj_points
    """
    players = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            players.append(
                Player(
                    id=row["id"],
                    name=row["name"],
                    position=Position(row["position"].strip().upper()),
                    team=row["team"].strip().upper(),
                    bye_week=int(row["bye_week"]),
                    rank=int(row["rank"]),
                    proj_points=float(row["proj_points"]),
                )
            )
    players.sort(key=lambda p: p.rank)
    return players
