# fantasy-draft-tool

A live draft assistant: as picks happen (yours and everyone else's), you
type them in, and it suggests the best available player for your team --
weighing player value, your remaining roster needs, and bye-week stacking.

## Why this exists

Most "best available" cheat sheets stop at overall rank. In a live draft
that's not enough on its own: a slightly worse player who fills an empty
starting slot is often the right pick over a slightly better one at a
position you've already filled, and stacking three starters onto the same
bye week quietly costs you a week of your season. This tool folds both of
those into the suggestion instead of leaving you to do the math at the
table.

The scoring logic is a pluggable strategy (`SuggestionStrategy` in
`draft_tool/strategy.py`), not a hardcoded formula -- the baseline
`ValueNeedByeStrategy` is meant to be swapped out or subclassed as the
logic gets smarter (real ADP feeds, VORP, opponent-adjusted projections,
a trained model), without touching the draft state or CLI.

## Quickstart

```
uv sync
uv run python -m draft_tool.cli --teams 12 --my-slot 5
```

Then, as the draft happens, just narrate it:

```
> pick Christian McCaffrey     # whoever is on the clock, in real snake-draft order
> pick Justin Jefferson
> status                       # see whose turn it is
> suggest                      # top 5 picks for your team right now
> roster                       # your roster so far
> available WR                 # remaining players at a position
> undo                         # fix a misclick
```

When it's your turn, suggestions print automatically before the prompt.
`pick` always resolves to whoever the snake-draft order says is on the
clock -- you don't tell it which team, it tracks that itself from
`--teams`/`--my-slot` and the pick count.

## How suggestions are scored

`ValueNeedByeStrategy` (`draft_tool/strategy.py`) combines three signals
per available player:

1. **Value** -- overall expert rank (lower rank number = more valuable).
2. **Need** -- a bonus for filling an empty starting slot, a smaller bonus
   for flex-eligible depth once starters are set, and nothing once a
   position is already well stocked.
3. **Bye week** -- once a roster would have as many same-position players
   on one bye week as it *starts* at that position, it can't field a full
   lineup that week no matter how deep the bench is. Each additional
   player stacked onto that bye beyond the threshold is penalized, scaled
   by how many starting slots exist at that position (single-start spots
   like QB/TE/K/DST get flagged at the first repeat; RB/WR tolerate one
   more before it matters).

Every suggestion prints its score breakdown as plain-English reasons, not
just a number, so you can sanity-check or override it at the table.

## Adding smarter logic later

Subclass `SuggestionStrategy` and implement `score(player, roster,
league) -> (float, list[str])`; swap it into `cli.py` in place of
`ValueNeedByeStrategy()`. Nothing else needs to change -- `DraftState` and
the CLI only depend on the abstract interface, not the baseline
implementation. Ideas: pull live ADP/projections instead of the static
CSV, weight bye weeks by strength of your bench at that position instead
of a flat threshold, or replace the scoring function with a trained
ranking model.

## Data

`data/players.csv` is a **synthetic sample pool** (real player names,
made-up ranks/projections/bye weeks) so the tool runs out of the box --
it is not real 2025 rankings. Point `--players` at your own CSV with the
same columns to use real data:

```
id,name,position,team,bye_week,rank,proj_points
```

`position` must be one of `QB, RB, WR, TE, K, DST`.

## Layout

- `draft_tool/models.py` -- `Player`, `Position`
- `draft_tool/league.py` -- `LeagueSettings` (roster shape: starting slots, flex, bench, team count)
- `draft_tool/data.py` -- CSV loader
- `draft_tool/draft.py` -- `DraftState`/`Roster`: snake-draft turn order, picks, undo, per-team rosters
- `draft_tool/strategy.py` -- `SuggestionStrategy` interface + the baseline `ValueNeedByeStrategy`
- `draft_tool/cli.py` -- interactive live-draft REPL
- `tests/` -- pytest coverage for draft state, scoring, and data loading

## Tests

```
uv run pytest
```
