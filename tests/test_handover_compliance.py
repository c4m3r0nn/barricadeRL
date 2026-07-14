from pathlib import Path

import tomllib

from barricade_rl.config import load_config
from barricade_rl.dashboard import REQUIRED_DASHBOARD_METRICS


ROOT = Path(__file__).resolve().parents[1]


def test_m2_config_matches_handover_non_negotiables():
    config = load_config(ROOT / "configs/m2_5x5.json")

    assert config["board"] == {"size": 5, "walls_per_player": 3, "max_plies": 200}
    assert config["action_space"]["action_count"] == 44
    assert config["action_space"]["policy_head"] == "flat-board-size-dependent"
    assert config["action_space"]["size_agnostic_policy_head"] is False
    assert config["observation"]["version"] == 1

    assert config["network"]["trunk"] == "resnet-global-pooling"
    assert config["network"]["blocks"] == 8
    assert config["network"]["filters"] == 96
    assert config["network"]["global_pool_blocks"] == [2, 5]
    assert config["network"]["auxiliary_targets"] == ["own_shortest_path", "opponent_shortest_path"]
    assert config["network"]["ema_decay"] == 0.999
    assert config["network"]["batch_norm_epsilon"] == 1e-5
    assert config["network"]["batch_norm_momentum"] == 0.1
    assert config["network"]["medium_variant"]["blocks"] == 12
    assert config["network"]["medium_variant"]["filters"] == 128

    assert config["mcts"]["mode"] == "puct"
    assert config["mcts"]["self_play_simulations"] == 200
    assert config["mcts"]["evaluation_simulations"] == 800
    assert config["mcts"]["cpuct_init"] == 1.6
    assert config["mcts"]["root_dirichlet_alpha"] == 0.6
    assert config["mcts"]["root_noise_fraction"] == 0.25
    assert config["mcts"]["fpu_reduction"] == 0.2
    assert config["mcts"]["forced_playout_weight"] == 2.0

    assert config["self_play"]["gamma"] == 1.0
    assert config["self_play"]["reward"] == "terminal-win-loss-cap-zero"
    assert config["self_play"]["resignation"] is False
    assert config["self_play"]["full_search_probability"] == 0.25
    assert config["self_play"]["fast_search_fraction"] == 0.25
    assert config["self_play"]["cap_adjudication"] == {
        "fraction_threshold": 0.05,
        "consecutive_cycles": 3,
        "scoring_scheme": "terminal-win-loss-cap-shortest-path-adjudicated",
    }
    assert config["self_play"]["weak_start_state_diversification"]["enabled"] is True
    assert config["self_play"]["weak_start_state_diversification"]["diversification_plies"] == 8

    assert config["training"]["optimizer"] == "sgd"
    assert config["training"]["momentum"] == 0.9
    assert config["training"]["batch_size"] == 512
    assert config["training"]["mirror_augmentation"] is True
    assert config["training"]["learning_rate_drop_steps"] == [100000, 200000]

    assert config["gating"] == {"enabled": True, "games": 200, "promotion_threshold": 0.55}
    assert config["acceptance"]["known_initial_result"] == "second-player-win"
    assert config["acceptance"]["solver_labelled_positions"] == 5000
    assert config["acceptance"]["value_sign_accuracy_min"] == 0.99
    assert config["acceptance"]["mcts_optimal_move_accuracy_min"] == 0.99


def test_project_uses_gymnasium_not_deprecated_gym_package():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("gymnasium") for dep in dependencies)
    assert not any(dep == "gym" or dep.startswith("gym<") or dep.startswith("gym>") for dep in dependencies)
    assert "import gymnasium as gym" in (ROOT / "barricade_rl/env.py").read_text()


def test_softmax_occurrences_are_masked_by_construction():
    occurrences = []
    for path in (ROOT / "barricade_rl").glob("*.py"):
        text = path.read_text()
        if "softmax" in text:
            occurrences.append(path.name)

    assert occurrences == ["mcts.py"]
    assert "def _masked_softmax" in (ROOT / "barricade_rl/mcts.py").read_text()


def test_dashboard_schema_reserves_required_handover_metrics():
    required = {
        "policy_loss",
        "value_loss",
        "auxiliary_loss",
        "root_policy_entropy",
        "value_calibration",
        "avg_game_length",
        "cap_fraction",
        "mean_walls_placed",
        "samples_per_position",
        "games_per_hour",
        "gpu_utilization",
        "ladder_elo",
    }
    assert required <= set(REQUIRED_DASHBOARD_METRICS)
