import json

import numpy as np

from barricade_rl.dashboard import (
    DASHBOARD_SCHEMA_VERSION,
    evaluation_to_dashboard_event,
    render_dashboard_html,
    write_dashboard_event,
)
from barricade_rl.evaluate import (
    GameRecord,
    MatchResult,
    estimate_elos,
    evaluate_ladder,
    play_game,
    play_match,
)
from barricade_rl.game import ACTION_COUNT, Game
from barricade_rl.opponents import RandomOpponent
from barricade_rl.small_board import SmallGame


class FirstLegalPolicy:
    def __init__(self, name="first-legal"):
        self.name = name

    def select_action(self, game, state, rng):
        del rng
        mask = game.legal_actions(state)
        return int(np.flatnonzero(mask)[0])


class ForwardPolicy:
    name = "forward"

    def select_action(self, game, state, rng):
        del rng
        if game.legal_actions(state)[0]:
            return 0
        return int(np.flatnonzero(game.legal_actions(state))[0])


def test_play_game_records_terminal_match_without_mutating_policy_contract():
    record = play_game(ForwardPolicy(), ForwardPolicy(), seed=7)

    assert isinstance(record, GameRecord)
    assert record.player0 == "forward"
    assert record.player1 == "forward"
    assert record.winner == 0
    assert record.termination == "win"
    assert record.plies == 15
    assert len(record.actions) == 15
    assert all(0 <= action < ACTION_COUNT for action in record.actions)


def test_play_match_balances_colours_and_scores_candidate():
    result = play_match(ForwardPolicy(), ForwardPolicy(), games_per_color=1, seed=11)

    assert isinstance(result, MatchResult)
    assert result.candidate == "forward"
    assert result.opponent == "forward"
    assert result.games == 2
    assert result.candidate_wins == 1
    assert result.opponent_wins == 1
    assert result.draws == 0
    assert result.candidate_score == 1.0
    assert result.score_rate == 0.5


def test_play_game_supports_m2_small_board_action_and_wall_counts():
    game = SmallGame()
    record = play_game(ForwardPolicy(), ForwardPolicy(), seed=5, game=game)

    assert all(0 <= action < game.action_count for action in record.actions)
    assert record.walls_placed == (0, 0)


def test_capped_games_are_recorded_as_draws_for_match_scoring():
    result = play_match(
        FirstLegalPolicy("candidate"),
        FirstLegalPolicy("opponent"),
        games_per_color=1,
        seed=3,
        game=Game(max_plies=2),
    )

    assert result.games == 2
    assert result.draws == 2
    assert result.candidate_score == 1.0
    assert result.score_rate == 0.5


def test_elo_estimator_is_anchored_at_random():
    ratings = estimate_elos(
        [
            MatchResult(
                candidate="candidate",
                opponent="random",
                candidate_wins=3,
                opponent_wins=1,
                draws=0,
                games_per_color=2,
                avg_plies=10.0,
                records=(),
            )
        ]
    )

    assert ratings["random"] == 0.0
    assert ratings["candidate"] > 0.0


def test_evaluate_ladder_returns_serializable_summary_with_elos():
    evaluation = evaluate_ladder(
        ForwardPolicy(),
        ladder=(RandomOpponent(),),
        games_per_color=1,
        seed=5,
    )

    payload = evaluation.to_dict()
    assert payload["candidate"] == "forward"
    assert payload["ladder_version"] == 1
    assert payload["seed"] == 5
    assert payload["matches"][0]["opponent"] == "random"
    assert "random" in payload["elo_ratings"]
    json.dumps(payload)


def test_dashboard_event_writer_and_html_renderer(tmp_path):
    evaluation = evaluate_ladder(
        ForwardPolicy(),
        ladder=(RandomOpponent(),),
        games_per_color=1,
        seed=9,
        run_id="m1-smoke",
    )
    event = evaluation_to_dashboard_event(evaluation)

    assert event["schema_version"] == DASHBOARD_SCHEMA_VERSION
    assert event["kind"] == "ladder_evaluation"
    assert event["run_id"] == "m1-smoke"
    assert event["metrics"]["ladder_elo"] == event["elo_ratings"]["forward"]

    events_path = tmp_path / "events.jsonl"
    write_dashboard_event(events_path, event)
    line = events_path.read_text().strip()
    assert json.loads(line)["candidate"] == "forward"

    html = render_dashboard_html(events_path)
    assert "m1-smoke" in html
    assert "forward" in html
    assert "ladder_elo" in html
