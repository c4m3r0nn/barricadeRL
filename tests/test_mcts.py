import numpy as np

from barricade_rl.mcts import MCTS, MCTSConfig, UniformEvaluator
from barricade_rl.small_board import SmallGame, SmallState


class IllegalLovingEvaluator:
    def evaluate(self, game, state):
        logits = np.zeros(game.action_count, dtype=np.float32)
        logits[1] = 100.0  # illegal in the initial position
        logits[-1] = 50.0
        return logits, 0.0


class ExplodingEvaluator:
    def evaluate(self, game, state):
        raise AssertionError("forced wins must not consult the evaluator")


def test_mcts_masks_illegal_network_logits_at_expansion():
    game = SmallGame()
    state = game.initial_state()
    search = MCTS(MCTSConfig(simulations=8), IllegalLovingEvaluator())
    result = search.run(game, state)

    assert result.policy.shape == (game.action_count,)
    assert np.isclose(result.policy.sum(), 1.0)
    assert result.policy[1] == 0.0
    assert np.all(result.policy[~game.legal_actions(state)] == 0.0)
    assert game.legal_actions(state)[result.action]


def test_mcts_immediate_win_bypasses_network_evaluator():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    result = MCTS(MCTSConfig(simulations=16), ExplodingEvaluator()).run(game, state)

    assert result.action == 0
    assert result.root_value == 1.0
    assert result.policy[0] == 1.0
    assert result.policy.sum() == 1.0


def test_mcts_finds_immediate_winning_pawn_move_with_uniform_priors():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    search = MCTS(MCTSConfig(simulations=16), UniformEvaluator())
    result = search.run(game, state)

    assert result.action == 0
    assert result.root_value > 0.5
    assert result.policy[0] == result.policy.max()


def test_mcts_value_sign_makes_forced_reply_win_bad_for_mover():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((0, 0), (1, 2)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    search = MCTS(MCTSConfig(simulations=16), UniformEvaluator())
    result = search.run(game, state)

    assert game.legal_actions(state)[result.action]
    assert result.root_value < 0.0


def test_self_play_root_noise_changes_only_legal_priors_deterministically():
    game = SmallGame()
    state = game.initial_state()
    plain = MCTS(MCTSConfig(simulations=1), UniformEvaluator(), rng=np.random.default_rng(7)).run(game, state)
    noisy = MCTS(
        MCTSConfig(
            simulations=1,
            root_dirichlet_alpha=0.6,
            root_noise_fraction=0.25,
        ),
        UniformEvaluator(),
        rng=np.random.default_rng(7),
    ).run(game, state)

    legal = game.legal_actions(state)
    assert plain.root_priors is not None
    assert noisy.root_priors is not None
    assert np.isclose(noisy.root_priors.sum(), 1.0)
    assert np.all(noisy.root_priors[~legal] == 0.0)
    assert not np.allclose(noisy.root_priors[legal], plain.root_priors[legal])


def test_forced_playout_visits_are_pruned_from_policy_target():
    game = SmallGame()
    result = MCTS(
        MCTSConfig(
            simulations=100,
            forced_playouts=True,
            policy_target_pruning=True,
        ),
        UniformEvaluator(),
        rng=np.random.default_rng(9),
    ).run(game, game.initial_state())

    assert result.forced_visits is not None
    assert result.policy_target_visits is not None
    assert result.forced_visits.sum() > 0
    assert np.all(result.policy_target_visits <= result.visits)
    assert np.any(result.policy_target_visits < result.visits)
    assert np.all(result.policy[~game.legal_actions(game.initial_state())] == 0.0)
    assert np.isclose(result.policy.sum(), 1.0)
