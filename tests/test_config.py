from pathlib import Path

from barricade_rl.config import config_hash, load_config, small_game_from_config


def test_m2_config_loads_and_matches_small_board_contract():
    config = load_config(Path("configs/m2_5x5.json"))
    game = small_game_from_config(config)

    assert config["milestone"] == "M2"
    assert config["acceptance"]["known_initial_result"] == "second-player-win"
    assert game.board_size == 5
    assert game.spec.walls_per_player == 3
    assert game.action_count == config["action_space"]["action_count"] == 44


def test_config_hash_is_stable_for_canonical_json():
    config = load_config(Path("configs/m2_5x5.json"))
    assert config_hash(config) == config_hash(load_config(Path("configs/m2_5x5.json")))
    assert len(config_hash(config)) == 64
