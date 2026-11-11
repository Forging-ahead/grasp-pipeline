"""JSON-compatible port definitions; field structures live in pipeline-v1.schema.json.

These are integration contracts, not automatic wrappers around existing code.
The existing LIBERO driver does not call these ports in this phase.
"""
from typing import Any, Mapping, Optional, Protocol, Sequence, TypedDict

Payload = Mapping[str, Any]


class StepResult(TypedDict):
    observation: Payload
    reward: float
    terminated: bool
    truncated: bool
    native_success: Optional[bool]


class EnvironmentPort(Protocol):
    def reset(self, task_id: str, *, seed: int, initial_state_id: Any, episode_id: str) -> Payload: ...
    def observe(self) -> Payload: ...
    def step(self, action: Sequence[float]) -> StepResult: ...
    def close(self) -> None: ...


class TaskPort(Protocol):
    def describe_task(self, task_id: str) -> Payload: ...
    def check_native_success(self) -> Optional[bool]: ...
    def check_subgoal(self, goal_id: str) -> Optional[bool]: ...


class RobotPort(Protocol):
    def describe_robot(self) -> Payload: ...
    def robot_state(self) -> Payload: ...


class ScenePort(Protocol):
    def scene_snapshot(self, target_entity_id: str) -> Payload: ...


class PlanningPort(Protocol):
    def plan(self, request: Payload, robot: Payload, observation: Payload, scene: Payload) -> Payload: ...


class PreviewPort(Protocol):
    def validate_and_restore(self, plan: Payload, contact_policy: Payload) -> Payload: ...


class ExecutionPort(Protocol):
    def execute(self, request: Payload, plan: Payload) -> Payload: ...


class RecordingPort(Protocol):
    def record_step(self, before: Payload, action: Sequence[float], after: Payload, *, phase: str,
                    decision_record: Optional[Payload] = None) -> None: ...
    def finalize(self, outcome: Payload) -> Payload: ...


class CapabilityPort(Protocol):
    def capabilities(self) -> Payload: ...
    def check_requirements(self, task: Payload, robot: Payload, observation: Payload, scene: Payload) -> Payload: ...
