import numpy as np

from barricade_rl.az_replay import AlphaZeroReplayBuffer, make_replay_sample
from barricade_rl.small_board import SmallGame


def test_alpha_zero_replay_sample_validates_policy_mask_and_round_trips_npz(tmp_path):
    game = SmallGame()
    state = game.initial_state()
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32)
    policy /= float(policy.sum())

    sample = make_replay_sample(
        game,
        state,
        policy=policy,
        value=0.5,
        source="unit",
        config_hash="abc123",
        root_value=0.25,
        observation_version=1,
        scoring_scheme="terminal-win-loss-cap-zero",
        game_id="game-7",
        run_id="run-3",
        git_commit="deadbeef",
        opponent_policy=policy,
        opponent_action_mask=mask,
    )

    assert sample.observation.shape == (6, 5, 5)
    assert sample.policy.shape == (game.action_count,)
    assert sample.action_mask.dtype == np.bool_
    assert sample.state_key == game.state_key(state)
    assert sample.board_size == 5
    assert sample.observation_version == 1
    assert sample.auxiliary_distances.shape == (2,)
    assert np.isfinite(sample.auxiliary_distances).all()
    assert sample.has_auxiliary_target
    assert sample.has_opponent_policy_target

    buffer = AlphaZeroReplayBuffer(
        capacity=2,
        observation_shape=sample.observation.shape,
        action_count=game.action_count,
    )
    buffer.add(sample)
    buffer.add(sample)
    buffer.add(sample)

    batch = buffer.sample(2, rng=np.random.default_rng(7))
    assert buffer.size == 2
    assert np.isclose(buffer.samples_per_position_ratio, 2 / 3)
    assert batch.observations.shape == (2, 6, 5, 5)
    assert batch.policies.shape == (2, game.action_count)
    assert batch.values.shape == (2,)
    assert batch.action_masks.shape == (2, game.action_count)
    assert batch.auxiliary_distances.shape == (2, 2)
    assert batch.opponent_policies.shape == (2, game.action_count)

    path = tmp_path / "replay.npz"
    buffer.save_npz(path)
    loaded = AlphaZeroReplayBuffer.load_npz(path)

    assert loaded.size == buffer.size
    assert loaded.capacity == buffer.capacity
    assert loaded.action_count == game.action_count
    assert loaded.samples[0].game_id == "game-7"
    assert loaded.samples[0].run_id == "run-3"
    assert loaded.samples[0].git_commit == "deadbeef"
    assert loaded.samples[0].config_hash == "abc123"
    assert loaded.samples[0].scoring_scheme == "terminal-win-loss-cap-zero"
    assert loaded.samples[0].observation_version == 1
    assert loaded.samples[0].board_size == 5
    np.testing.assert_allclose(loaded.samples[0].auxiliary_distances, sample.auxiliary_distances)
    np.testing.assert_allclose(loaded.samples[0].opponent_policy, policy)
    loaded_batch = loaded.sample(1, rng=np.random.default_rng(1))
    assert loaded_batch.observations.shape == (1, 6, 5, 5)


def test_alpha_zero_replay_rejects_illegal_policy_mass():
    game = SmallGame()
    state = game.initial_state()
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32)
    policy /= float(policy.sum())
    illegal_action = int(np.flatnonzero(~mask)[0])
    policy[illegal_action] = 0.01
    policy /= float(policy.sum())

    try:
        make_replay_sample(game, state, policy=policy, value=0.0)
    except ValueError as exc:
        assert "illegal action" in str(exc)
    else:
        raise AssertionError("policy mass on illegal actions must be rejected")


def test_observation_batches_are_deterministic_and_do_not_consume_replay():
    game = SmallGame()
    state = game.initial_state()
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32) / mask.sum()
    buffer = AlphaZeroReplayBuffer(
        capacity=8,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )
    for action in np.flatnonzero(mask)[:3]:
        next_state = game.next_state(state, int(action))
        next_mask = game.legal_actions(next_state)
        next_policy = next_mask.astype(np.float32) / next_mask.sum()
        buffer.add(make_replay_sample(game, next_state, policy=next_policy, value=0.0))
    counters_before = (buffer.total_positions_added, buffer.gradient_samples_consumed)

    first = tuple(buffer.observation_batches(batch_size=2))
    second = tuple(buffer.observation_batches(batch_size=2))

    assert [batch.shape[0] for batch in first] == [2, 1]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(
        np.concatenate(first),
        np.stack([sample.observation for sample in buffer.samples]),
    )
    assert (buffer.total_positions_added, buffer.gradient_samples_consumed) == counters_before
