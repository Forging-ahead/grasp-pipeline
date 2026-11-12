import json

import pytest

from graspgen_libero_bridge.adapters.robocasa.config import load_config


@pytest.mark.parametrize("heights", [[0.], [float("nan")], [True], [.1, .1]])
def test_invalid_transfer_heights_stop_before_simulation(tmp_path, heights):
    policy = load_config("PickPlaceCounterToStove")["policy"]
    policy["transfer_clearance_candidates_m"] = heights
    file = tmp_path / "policy.json"; file.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Transfer clearances"):
        load_config("PickPlaceCounterToStove", policy_file=file)
