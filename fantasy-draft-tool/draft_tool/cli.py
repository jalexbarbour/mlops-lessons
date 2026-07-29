from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_players
from .draft import DraftState, Roster
from .league import LeagueSettings
from .strategy import Suggestion, SuggestionStrategy, ValueNeedByeStrategy

DEFAULT_PLAYERS = Path(__file__).resolve().parent.parent / "data" / "players.csv"

HELP = """\
Commands:
  pick <name>       record the next pick (whoever's turn it is) for the player matching <name>
  suggest [n]       show top n suggested picks for your team (default 5)
  roster [team]     show a team's roster (default: yours)
  available [pos]   list top available players, optionally filtered by position (e.g. RB)
  status            show pick number, round, and who's on the clock
  undo              undo the last pick
  help              show this message
  quit              exit
"""


def format_suggestion(rank: int, s: Suggestion) -> str:
    return f"{rank}. {s.player}  score={s.score:.1f}  [{'; '.join(s.reasons)}]"


def print_suggestions(state: DraftState, strategy: SuggestionStrategy, top_n: int = 5) -> None:
    suggestions = strategy.suggest(state.available_players(), state.my_roster(), state.league, top_n)
    if not suggestions:
        print("No players available.")
        return
    print(f"Top {len(suggestions)} suggestions for team {state.my_team_id}:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {format_suggestion(i, s)}")


def print_roster(roster: Roster) -> None:
    if not roster.players:
        print("  (empty)")
        return
    for p in roster.players:
        print(f"  {p}")


def resolve_player(state: DraftState, query: str):
    matches = state.find(query)
    if not matches:
        print(f"No available player matches '{query}'.")
        return None
    if len(matches) > 1:
        print(f"Multiple matches for '{query}', be more specific:")
        for p in sorted(matches, key=lambda p: p.rank)[:10]:
            print(f"  {p}")
        return None
    return matches[0]


def run(state: DraftState, strategy: SuggestionStrategy) -> None:
    print(HELP)
    while True:
        if state.is_my_turn():
            print(f"\n--- Pick {state.pick_number} (round {state.round_number}) -- YOUR TURN ---")
            print_suggestions(state, strategy)

        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "help":
            print(HELP)

        elif cmd == "status":
            on_clock = state.current_team_id()
            you = " (you)" if state.is_my_turn() else ""
            print(f"Pick {state.pick_number}, round {state.round_number}: team {on_clock} on the clock{you}.")

        elif cmd in ("pick", "p"):
            if not arg:
                print("Usage: pick <player name>")
                continue
            player = resolve_player(state, arg)
            if player is None:
                continue
            team_id = state.draft(player)
            print(f"Team {team_id} selects {player}.")

        elif cmd == "suggest":
            top_n = int(arg) if arg.isdigit() else 5
            print_suggestions(state, strategy, top_n)

        elif cmd == "roster":
            team_id = int(arg) if arg.isdigit() else state.my_team_id
            if team_id not in state.rosters:
                print(f"No such team: {team_id}")
                continue
            print(f"Team {team_id} roster:")
            print_roster(state.rosters[team_id])

        elif cmd == "available":
            pos = arg.upper() or None
            players = sorted(state.available_players(), key=lambda p: p.rank)
            if pos:
                players = [p for p in players if p.position.value == pos]
            for p in players[:25]:
                print(f"  {p}  rank={p.rank}")

        elif cmd == "undo":
            try:
                state.undo()
                print("Last pick undone.")
            except ValueError as exc:
                print(exc)

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for commands.")

    print("\nFinal roster:")
    print_roster(state.my_roster())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live fantasy football draft assistant.")
    parser.add_argument("--players", default=str(DEFAULT_PLAYERS), help="path to the players CSV")
    parser.add_argument("--teams", type=int, default=12, help="number of teams in the league")
    parser.add_argument("--my-slot", type=int, default=1, help="your draft position (1-indexed)")
    parser.add_argument("--bench", type=int, default=6, help="bench slots")
    args = parser.parse_args(argv)

    players = load_players(args.players)
    league = LeagueSettings(num_teams=args.teams, bench_slots=args.bench)
    state = DraftState(players, league, my_team_id=args.my_slot)
    strategy = ValueNeedByeStrategy()
    run(state, strategy)


if __name__ == "__main__":
    main()
