from dataclasses import dataclass

import numpy as np

from barricade_rl.az_self_play import (
    SelfPlayConfig,
    generate_self_play_games,
    play_self_play_game,
)
from barricade_rl.az_replay import AlphaZeroReplayBuffer
from barricade_rl.config import load_config
from barricade_rl.game import TerminalStatus
from barricade_rl.mcts import MCTSResult
from barricade_rl.small_board import SmallGame, SmallState


class ZeroEvaluator:
    def evaluate(self, game, state):
        return np.zeros(game.action_count, dtype=np.float32), 0.0


def test_self_play_config_matches_m2_handover_constants():
    config = SelfPlayConfig.from_project_config(load_config("configs/m2_5x5.json"))

    assert config.full_simulations == 200
    assert config.fast_simulations == 50
    assert config.cpuct == 1.6
    assert config.full_search_probability == 0.25
    assert config.temperature_moves == 16
    assert config.raw_policy_injection_probability == 0.04
    assert config.diversification_plies == 8
    assert config.root_dirichlet_alpha == 0.6
    assert config.root_noise_fraction == 0.25
    assert config.fpu_reduction == 0.2
    assert config.forced_playouts
    assert config.policy_target_pruning
    assert config.forced_playout_weight == 2.0


def test_full_search_position_is_recorded_with_final_mover_value():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    config = SelfPlayConfig(
        full_simulations=4,
        fast_simulations=1,
        full_search_probability=1.0,
        raw_policy_injection_probability=0.0,
    )

    record = play_self_play_game(
        game,
        ZeroEvaluator(),
        config,
        rng=np.random.default_rng(4),
        initial_state=state,
        game_id="forced-win",
        run_id="test-run",
        config_hash="abc123",
    )

    assert record.winner == 0
    assert record.terminal_status is TerminalStatus.MOVER_LOST
    assert record.full_searches == 1
    assert record.fast_searches == 0
    assert record.injected_ply is None
    assert record.actions == (0,)
    assert len(record.samples) == 1
    assert record.samples[0].value == 1.0
    assert record.samples[0].current_player == 0
    assert record.samples[0].policy[0] == 1.0
    assert record.samples[0].game_id == "forced-win"
    assert record.samples[0].run_id == "test-run"
    assert record.samples[0].config_hash == "abc123"
    assert record.samples[0].scoring_scheme == "terminal-win-loss-cap-zero"


def test_fast_search_moves_continue_game_without_entering_replay():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    config = SelfPlayConfig(
        full_simulations=4,
        fast_simulations=1,
        full_search_probability=0.0,
        raw_policy_injection_probability=0.0,
    )

    record = play_self_play_game(
        game,
        ZeroEvaluator(),
        config,
        rng=np.random.default_rng(2),
        initial_state=state,
    )

    assert record.winner == 0
    assert record.full_searches == 0
    assert record.fast_searches == 1
    assert record.samples == ()


@dataclass(frozen=True)
class LineState:
    ply: int = 0
    current_player: int = 0


class LineGame:
    action_count = 2
    board_size = 1

    def initial_state(self):
        return LineState()

    def legal_actions(self, state):
        if self.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            return np.zeros(2, dtype=np.bool_)
        return np.asarray([True, False])

    def next_state(self, state, action):
        assert action == 0
        return LineState(ply=state.ply + 1, current_player=1 - state.current_player)

    def is_terminal(self, state):
        return TerminalStatus.MOVER_LOST if state.ply == 3 else TerminalStatus.NOT_TERMINAL

    def canonical_observation(self, state):
        return np.full((1, 1, 1), state.current_player, dtype=np.float32)

    def state_key(self, state):
        return bytes((state.ply, state.current_player))


class ScriptedRng:
    def __init__(self):
        # Move 0: no injection, full search. Move 1: injection. Move 2: full search.
        self._random_values = iter((0.5, 0.0, 0.0, 0.0))

    def random(self):
        return next(self._random_values)

    def choice(self, count, p):
        return int(np.flatnonzero(np.asarray(p) > 0)[0])


class ScriptedSearch:
    def run(self, game, state):
        policy = np.asarray([1.0, 0.0], dtype=np.float32)
        return MCTSResult(action=0, policy=policy, root_value=0.25, visits=policy.copy())


def test_raw_policy_injection_discards_pre_injection_value_targets():
    config = SelfPlayConfig(
        full_simulations=4,
        fast_simulations=1,
        full_search_probability=1.0,
        raw_policy_injection_probability=0.04,
        diversification_plies=3,
    )
    factory_calls = []

    def search_factory(search_config, evaluator, rng):
        factory_calls.append(search_config)
        return ScriptedSearch()

    record = play_self_play_game(
        LineGame(),
        ZeroEvaluator(),
        config,
        rng=ScriptedRng(),
        search_factory=search_factory,
    )

    assert record.injected_ply == 1
    assert record.winner == 0
    assert record.actions == (0, 0, 0)
    assert record.full_searches == 2
    assert len(record.samples) == 1
    assert record.samples[0].ply == 2
    assert record.samples[0].value == 1.0
    assert [call.simulations for call in factory_calls] == [4, 4]


def test_full_and_fast_searches_use_the_required_exploration_settings():
    game = LineGame()
    calls = []

    class AlternatingRng:
        def __init__(self):
            # Move 0 full; move 1 fast; move 2 full. Injection is disabled.
            self.values = iter((0.0, 1.0, 0.0))

        def random(self):
            return next(self.values)

        def choice(self, count, p):
            return 0

    def search_factory(search_config, evaluator, rng):
        calls.append(search_config)
        return ScriptedSearch()

    config = SelfPlayConfig(
        full_simulations=20,
        fast_simulations=5,
        full_search_probability=0.5,
        raw_policy_injection_probability=0.0,
        root_dirichlet_alpha=0.6,
        root_noise_fraction=0.25,
        forced_playouts=True,
        policy_target_pruning=True,
    )
    play_self_play_game(
        game,
        ZeroEvaluator(),
        config,
        rng=AlternatingRng(),
        search_factory=search_factory,
    )

    assert [call.simulations for call in calls] == [20, 5, 20]
    assert calls[0].root_noise_fraction == 0.25
    assert calls[0].forced_playouts
    assert calls[0].policy_target_pruning
    assert calls[1].root_noise_fraction == 0.0
    assert not calls[1].forced_playouts
    assert not calls[1].policy_target_pruning
    assert calls[2].root_noise_fraction == 0.25


def test_generate_self_play_games_appends_samples_to_replay():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    replay = AlphaZeroReplayBuffer(
        capacity=8,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )
    config = SelfPlayConfig(
        full_simulations=2,
        fast_simulations=1,
        full_search_probability=1.0,
        raw_policy_injection_probability=0.0,
    )

    records = generate_self_play_games(
        game,
        ZeroEvaluator(),
        config,
        games=2,
        replay_buffer=replay,
        rng=np.random.default_rng(3),
        initial_state_factory=lambda _: state,
        run_id="test-run",
        config_hash="abc123",
        git_commit="deadbeef",
    )

    assert len(records) == 2
    assert replay.size == 2
    assert replay.samples[0].run_id == "test-run"
    assert replay.samples[0].git_commit == "deadbeef"
    assert {sample.value for record in records for sample in record.samples} == {1.0}


def test_following_ply_search_policy_becomes_opponent_policy_target():
    config = SelfPlayConfig(
        full_simulations=4,
        fast_simulations=1,
        full_search_probability=1.0,
        raw_policy_injection_probability=0.0,
    )

    record = play_self_play_game(
        LineGame(),
        ZeroEvaluator(),
        config,
        rng=np.random.default_rng(8),
        search_factory=lambda search_config, evaluator, rng: ScriptedSearch(),
    )

    assert len(record.samples) == 3
    assert record.samples[0].has_opponent_policy_target
    np.testing.assert_array_equal(record.samples[0].opponent_policy, np.asarray([1.0, 0.0]))
    assert record.samples[1].has_opponent_policy_target
    assert not record.samples[2].has_opponent_policy_target
