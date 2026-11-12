import json
import pytest

from graspgen_libero_bridge.adapters.robocasa.config import load_config


def test_native_budget_is_default_and_teacher_requires_explicit_selection(tmp_path):
    native = load_config("PickPlaceCounterToStove")["policy"]
    assert native["budget_profile"] == "native" and "teacher_step_limit" not in native
    teacher = load_config("PickPlaceCounterToStove", budget_profile="libero-v10-teacher")["policy"]
    assert teacher["teacher_step_limit"] == 5000
    file = tmp_path / "policy.json"; file.write_text(json.dumps(teacher))
    restored = load_config("PickPlaceCounterToStove", policy_file=file)["policy"]
    assert restored["budget_profile"] == "native" and "teacher_step_limit" not in restored


def test_unknown_budget_profile_is_rejected_before_simulation():
    with pytest.raises(ValueError, match="budget profile"):
        load_config("PickPlaceCounterToStove", budget_profile="silently-unlimited")
