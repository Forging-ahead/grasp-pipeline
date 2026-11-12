"""A bounded failed search must retain its real structured failure result."""
import json

import numpy as np
import pytest

from graspgen_libero_bridge.adapters.robocasa import native
from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.runtime.robocasa_driver import main


@pytest.mark.parametrize("code", ["SEARCH_BUDGET_EXHAUSTED", "COLLISION_REJECTED"])
def test_driver_records_nested_native_matrices_and_closes_the_environment(monkeypatch, tmp_path, code):
    closed = []
    class FailingEnvironment:
        def __init__(self, *args, **kwargs):
            self.env = None
        def reset(self):
            raise EpisodeFailure(code, "bounded native search failed", details={"regions": [
                {"frame": np.eye(4), "half_size": np.ones(3), "count": np.int64(4), "passed": np.bool_(False)}]})
        def close(self):
            closed.append(True)
    monkeypatch.setattr(native, "RoboCasaEnvironment", FailingEnvironment)
    config = tmp_path / "CONFIG.json"
    config.write_text(json.dumps(load_config("PickPlaceCounterToStove")))
    assert main(["probe", "--config", str(config), "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert result["status"] == "failed" and result["failure_code"] == code
    assert result["details"]["regions"][0]["frame"] == np.eye(4).tolist()
    assert result["details"]["regions"][0]["count"] == 4
    assert result["source_unchanged_during_run"] and closed == [True]
    assert not (tmp_path / "RESULT.json.tmp").exists()
