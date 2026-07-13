from barricade_rl.verify import differential_verification, random_game_verification


def test_random_game_invariants_and_deterministic_replay():
    metrics = random_game_verification(games=100, seed=20260705)
    assert metrics["games"] == 100
    assert metrics["plies"] > 0


def test_native_engine_matches_python_oracle():
    assert differential_verification(states=1_000, seed=20260705) == 1_000

