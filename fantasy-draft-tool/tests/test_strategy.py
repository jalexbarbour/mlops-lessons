from __future__ import annotations

from draft_tool.draft import Roster
from draft_tool.league import LeagueSettings
from draft_tool.models import Player, Position
from draft_tool.strategy import ValueNeedByeStrategy


def make_player(id, position, bye_week, rank, team="XX"):
    return Player(id=id, name=f"Player {id}", position=Position(position), team=team, bye_week=bye_week, rank=rank, proj_points=100.0)


def test_higher_rank_scores_lower_by_default():
    league = LeagueSettings()
    roster = Roster(team_id=1)
    strategy = ValueNeedByeStrategy()
    best = make_player("1", "RB", bye_week=7, rank=1)
    worse = make_player("2", "RB", bye_week=8, rank=50)
    score_best, _ = strategy.score(best, roster, league)
    score_worse, _ = strategy.score(worse, roster, league)
    assert score_best > score_worse


def test_need_bonus_favors_unfilled_position():
    league = LeagueSettings()  # needs 1 QB, 2 RB, 2 WR, 1 TE, 1 K, 1 DST
    # RB starting slots are already full (2/2); QB is still empty (0/1).
    roster = Roster(
        team_id=1,
        players=[
            make_player("r1", "RB", bye_week=7, rank=5),
            make_player("r2", "RB", bye_week=8, rank=6),
        ],
    )
    strategy = ValueNeedByeStrategy()

    # Same rank: QB fills an empty starting slot, RB only adds bench/flex depth.
    qb_candidate = make_player("q1", "QB", bye_week=9, rank=20)
    rb_candidate = make_player("r3", "RB", bye_week=9, rank=20)

    qb_score, qb_reasons = strategy.score(qb_candidate, roster, league)
    rb_score, rb_reasons = strategy.score(rb_candidate, roster, league)

    assert qb_score > rb_score
    assert any("fills starting need" in r for r in qb_reasons)


def test_bye_week_stack_is_penalized():
    league = LeagueSettings()  # starts 2 RB
    strategy = ValueNeedByeStrategy()

    # Roster already has 2 RBs (the full starting need) on bye week 9.
    roster = Roster(
        team_id=1,
        players=[
            make_player("r1", "RB", bye_week=9, rank=5),
            make_player("r2", "RB", bye_week=9, rank=6),
        ],
    )

    same_bye = make_player("r3", "RB", bye_week=9, rank=30)
    different_bye = make_player("r4", "RB", bye_week=10, rank=30)

    same_bye_score, same_bye_reasons = strategy.score(same_bye, roster, league)
    diff_bye_score, _ = strategy.score(different_bye, roster, league)

    assert diff_bye_score > same_bye_score
    assert any("caution" in r for r in same_bye_reasons)


def test_bye_week_penalty_ignored_below_threshold():
    league = LeagueSettings()  # starts 2 RB, so 1 existing RB on a bye is fine
    strategy = ValueNeedByeStrategy()
    roster = Roster(team_id=1, players=[make_player("r1", "RB", bye_week=9, rank=5)])

    same_bye = make_player("r2", "RB", bye_week=9, rank=30)
    _, reasons = strategy.score(same_bye, roster, league)
    assert not any("caution" in r for r in reasons)


def test_suggest_returns_top_n_sorted_desc():
    league = LeagueSettings()
    roster = Roster(team_id=1)
    strategy = ValueNeedByeStrategy()
    players = [make_player(str(i), "WR", bye_week=7, rank=i) for i in range(1, 11)]

    suggestions = strategy.suggest(players, roster, league, top_n=3)

    assert len(suggestions) == 3
    assert [s.player.id for s in suggestions] == ["1", "2", "3"]
    assert suggestions[0].score >= suggestions[1].score >= suggestions[2].score
