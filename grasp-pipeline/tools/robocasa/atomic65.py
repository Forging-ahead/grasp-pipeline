"""Official 65-task inventory and semantic-only bindings for the frozen r10 actor."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


# These are native object / fixture names checked against the fixed task source.
# No positions, orientations, robot trajectories or task-specific tolerances.
PICK_PLACE_BINDINGS = {
    "PickPlaceCabinetToCounter": ("obj", "counter", "fixture"),
    "PickPlaceCounterToBlender": ("obj", "blender", "fixture"),
    "PickPlaceCounterToCabinet": ("obj", "cab", "fixture"),
    "PickPlaceCounterToDrawer": ("obj", "drawer", "fixture"),
    "PickPlaceCounterToMicrowave": ("obj", "container", "object"),
    "PickPlaceCounterToOven": ("obj", "oven_tray", "object"),
    "PickPlaceCounterToSink": ("obj", "sink", "fixture"),
    "PickPlaceCounterToStandMixer": ("obj", "stand_mixer", "fixture"),
    "PickPlaceCounterToStove": ("obj", "container", "object"),
    "PickPlaceCounterToToasterOven": ("obj", "toaster_oven", "fixture"),
    "PickPlaceDrawerToCounter": ("obj", "counter", "fixture"),
    "PickPlaceFridgeDrawerToShelf": ("obj", "fridge", "fixture"),
    "PickPlaceFridgeShelfToDrawer": ("obj", "fridge", "fixture"),
    "PickPlaceMicrowaveToCounter": ("obj", "container", "object"),
    "PickPlaceSinkToCounter": ("obj", "container", "object"),
    "PickPlaceStoveToCounter": ("obj", "container", "object"),
    "PickPlaceToasterOvenToCounter": ("obj", "container", "object"),
    "PickPlaceToasterToCounter": ("obj", "plate", "object"),
    "CheesyBread": ("cheese", "bread", "object"),
    "MakeIcedCoffee": ("ice_cube1", "cup", "object"),
    "PackDessert": ("dessert", "cooked_food_container", "object"),
    "CoffeeServeMug": ("obj", "counter", "fixture"),
}

CATEGORY_TASKS = {
    "door": {"OpenCabinet", "CloseCabinet", "OpenDishwasher", "CloseDishwasher", "OpenFridge", "CloseFridge",
             "OpenMicrowave", "CloseMicrowave", "OpenOven", "CloseOven", "OpenToasterOvenDoor", "CloseToasterOvenDoor"},
    "drawer": {"OpenDrawer", "CloseDrawer", "OpenFridgeDrawer", "CloseFridgeDrawer"},
    "lid_head": {"OpenBlenderLid", "CloseBlenderLid", "OpenElectricKettleLid", "CloseElectricKettleLid",
                 "OpenStandMixerHead", "CloseStandMixerHead"},
    "slide_rack": {"SlideDishwasherRack", "SlideOvenRack", "SlideToasterOvenRack"},
    "knob": {"AdjustToasterOvenTemperature", "PreheatOven", "TurnOnStove", "TurnOffStove", "LowerHeat", "TurnOnToasterOven"},
    "lever_spout": {"TurnOnSinkFaucet", "TurnOffSinkFaucet", "AdjustWaterTemperature", "TurnSinkSpout"},
    "button": {"StartCoffeeMachine", "TurnOnMicrowave", "TurnOffMicrowave", "TurnOnBlender", "TurnOnElectricKettle", "TurnOnToaster"},
    "navigation": {"NavigateKitchen"},
    "insertion": {"CoffeeServeMug", "CoffeeSetupMug"},
}
CATEGORY_LABELS = {"pick_place": "抓取放置", "door": "开关门", "drawer": "开关抽屉", "lid_head": "开合盖子或机头",
                   "slide_rack": "滑动托架", "knob": "旋钮调节", "lever_spout": "水龙头与水嘴", "button": "按钮或启动操作",
                   "navigation": "移动导航", "insertion": "咖啡机取放与对位"}


def inventory(repo):
    repo = Path(repo)
    registry_file = repo / "robocasa/utils/dataset_registry.py"
    tree = ast.parse(registry_file.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ATOMIC_TASK_DATASETS" for t in node.targets))
    entries = []
    for task in node.value.keywords:
        fields = {k.arg: k.value for k in task.value.keywords}
        entries.append({"task_id": task.arg, "reference_horizon_steps": ast.literal_eval(fields["horizon"]),
                        "available_splits": [s for s in ("pretrain", "target") if s in fields]})
    if len(entries) != 65 or len({e["task_id"] for e in entries}) != 65:
        raise ValueError("The fixed official registry is not exactly 65 unique tasks")
    classes = {}
    for path in sorted((repo / "robocasa/environments/kitchen/atomic").glob("*.py")):
        source = path.read_text()
        for cls in ast.parse(source).body:
            if isinstance(cls, ast.ClassDef):
                classes[cls.name] = {"path": str(path.relative_to(repo)), "line": cls.lineno,
                                     "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "class_sha256": hashlib.sha256(ast.get_source_segment(source, cls).encode()).hexdigest()}
    names = {e["task_id"] for e in entries}
    if not set(PICK_PLACE_BINDINGS) <= names:
        raise ValueError("A declared semantic binding does not occur in the native registry")
    covered = set(PICK_PLACE_BINDINGS)
    covered.update(name for group in CATEGORY_TASKS.values() for name in group)
    if covered != names:
        raise ValueError("Explicit task classification does not cover exactly the native registry")
    for entry in entries:
        name = entry["task_id"]
        entry.update(category=next((k for k, tasks in CATEGORY_TASKS.items() if name in tasks), "pick_place"),
                     role="atomic65_coverage", native_source=classes[name])
        if "pretrain" not in entry["available_splits"]:
            raise ValueError("This sweep declares pretrain for every task")
        if name in PICK_PLACE_BINDINGS:
            target, destination, kind = PICK_PLACE_BINDINGS[name]
            entry.update(execution_mode="run", stages=[{"skill": "pick_place", "target": target,
                         "destination": destination, "destination_kind": kind}],
                         binding_status="shared_r10_pick_place_attempt", unsupported_reason=None)
        else:
            reason = ("当前 r10 未实现咖啡机出水口的专用对位终点规则" if name == "CoffeeSetupMug" else
                      "当前 r10 的底盘导航技能未开放" if name == "NavigateKitchen" else
                      "当前 r10 的机构、旋钮或按钮技能尚未通过验收，保持原关闭配置")
            entry.update(execution_mode="probe", stages=[], binding_status="unsupported_no_task_action", unsupported_reason=reason)
        if name == "MakeIcedCoffee":
            entry["binding_note"] = "原生目标允许两块冰中任一块；固定绑定 ice_cube1，不依据 seed 或运行结果挑选"
        if name == "PickPlaceSinkToCounter":
            entry["binding_note"] = "按原生成功条件放入台面上的 container；旧 12 任务绑定错误地指向 counter。本轮只修正语义绑定，不改 r10 策略"
        if name in {"PickPlaceFridgeDrawerToShelf", "PickPlaceFridgeShelfToDrawer", "PickPlaceCounterToToasterOven"}:
            entry["binding_limitation"] = "共享 r10 会搜索该 fixture 的原生内部区域；尚无专用托架/层级选择接口，最终必须通过未修改的原生目标"
    return {"schema_version": "robocasa-atomic65-bindings-1", "native_commit": "3443cdf3dd20f9ae8ed41adba442059b506891d0",
            "registry_sha256": hashlib.sha256(registry_file.read_bytes()).hexdigest(), "tasks": entries,
            "category_labels": CATEGORY_LABELS, "policy_actor_changed": False,
            "scope": "semantic task bindings only; run 22 shared pick-place attempts and probe 43 unsupported scenes; never count probe_passed as task success"}
