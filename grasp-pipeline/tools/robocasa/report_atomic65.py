#!/usr/bin/env python3
"""Retain all 65 cells, separating unsupported scene probes from policy outcomes."""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.runtime.robocasa_provenance import digest_json
from atomic65 import CATEGORY_LABELS
from build_report import explain_failure, native_budget_success


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # Live CONFIG/RESULT files can be observed between write and close.
        # A finalized missing/malformed result is classified as PROCESS_ERROR.
        return None


def summarize(root):
    root = Path(root)
    matrix = read(root / "MATRIX.json")
    bindings = read(root / "TASK_BINDINGS.json")
    if matrix["planned_episodes"] != 65 or len(matrix["cells"]) != 65:
        raise ValueError("The denominator must be the complete 65-task registry")
    if len({c["task_id"] for c in matrix["cells"]}) != 65 or digest_json(bindings) != matrix["task_bindings_digest"]:
        raise ValueError("Duplicate tasks or changed semantic bindings")
    task_lookup = {t["task_id"]: t for t in bindings["tasks"]}
    rows = []
    for cell in matrix["cells"]:
        name = cell["task_id"] + "__seed" + str(cell["seed"]).zfill(6)
        case = root / "cases" / name
        result = read(case / "RESULT.json") or {}
        launcher = read(case / "LAUNCHER.json")
        supervisor = read(case / "SUPERVISOR.json") or {}
        config = read(case / "CONFIG.json") or {}
        lock = read(case / "ENVIRONMENT_LOCK.json") or {}
        completed = bool(launcher)
        identity_ok = bool(result and all(result.get(k) == cell[k] for k in ("task_id", "seed", "split"))
                           and result.get("mode") == cell["execution_mode"]
                           and result.get("source_digest") == matrix["source_digest"]
                           and result.get("source_unchanged_during_run")
                           and result.get("policy_digest") == matrix["policy_digest"]
                           and config.get("policy") and digest_json(config["policy"]) == matrix["policy_digest"]
                           and config.get("task") == task_lookup[cell["task_id"]]
                           and not result.get("paired_diagnostic")
                           and result.get("budget_profile") == matrix["budget_profile"])
        policy = cell["execution_mode"] == "run"
        interrupted = bool(supervisor.get("timed_out") or supervisor.get("interrupted_signal"))
        full_success = bool(completed and identity_ok and not interrupted and policy and result.get("status") == "success"
                            and result.get("pipeline_success") and result.get("native_success_at_end")
                            and not result.get("native_success_at_reset") and result.get("physical_steps", 0) > 0)
        if not completed:
            status, failure = ("running", "RUNNING") if config else ("pending", "NOT_RUN")
        elif interrupted or result.get("failure_code") == "WALL_TIMEOUT":
            status, failure = "technical_failure", "WALL_TIMEOUT"
        elif not result:
            status, failure = "technical_failure", "PROCESS_ERROR"
        elif not identity_ok:
            status, failure = "technical_failure", "VERSION_OR_IDENTITY_MISMATCH"
        elif full_success:
            status, failure = "success", None
        elif not policy and result.get("status") == "probe_passed":
            status, failure = "unsupported", "UNSUPPORTED_SKILL"
        else:
            failure = result.get("failure_code", "NATIVE_GOAL_FALSE")
            technical = {"ENVIRONMENT_ERROR", "ASSET_MISSING", "ROBOT_MODEL_MISMATCH", "ENVIRONMENT_VERSION_MISMATCH",
                         "STORAGE_ERROR", "MODEL_ERROR", "PLANNER_ERROR", "SOURCE_CHANGED_DURING_RUN"}
            status = "technical_failure" if failure in technical else "policy_failure"
        messages = {"TARGET_NOT_VISIBLE": "相机中目标的有效深度点不足，未使用完整几何替代抓取输入",
                    "OBSERVATION_INVALID": "当前相机观测、掩码或深度未通过检查",
                    "IK_NOT_CONVERGED": "局部逆运动学没有收敛到满足要求的姿态",
                    "cartesian_ik_collision_blocked": "原版局部接触路径搜索被真实几何接触阻挡",
                    "cartesian_start_collision": "局部接触规划起点存在禁止接触",
                    "ASSET_MISSING": "原生环境所需文件或资产缺失",
                    "ENVIRONMENT_ERROR": "原生环境创建、渲染或运行发生技术错误"}
        reason = cell["unsupported_reason"] if status == "unsupported" else (messages.get(failure, explain_failure(failure, result)) if failure else "完整成功")
        if status == "unsupported" and result.get("physical_steps", 0) != 0:
            raise ValueError("An unsupported scene probe unexpectedly executed actions")
        recording = read(case / "RECORDING.json") or {}
        hz = recording.get("fps", config.get("policy", {}).get("control_freq", 20))
        close_records = [read(p) for p in case.glob("stages/*/grasp_*/CLOSE_EVIDENCE.json")]
        lift_records = [read(p) for p in case.glob("stages/*/grasp_*/LIFT_EVIDENCE.json")]
        lift_limit = config.get("policy", {}).get("lift_clearance_m", .15) * .5
        passed_lift = any(p and p.get("native_bilateral_grasp") and p.get("object_lift_m", 0.) >= lift_limit for p in lift_records)
        decisions = []
        decision_file = case / "DECISIONS.jsonl"
        if decision_file.is_file():
            for line in decision_file.read_text().splitlines():
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        last_phase = decisions[-1]["phase"] if decisions else result.get("failure_stage", "initialization")
        rows.append(dict(cell, case=name, completed=completed, status=status, failure_code=failure, reason=reason,
                         identity_valid=identity_ok, pipeline_success=full_success,
                         native_budget_success=bool(full_success and native_budget_success(result)),
                         native_success_at_reset=result.get("native_success_at_reset"),
                         physical_steps=result.get("physical_steps", 0), wall_s=result.get("wall_s", supervisor.get("wall_s")),
                         simulation_action_s=result.get("physical_steps", 0) / hz,
                         initialized=bool(lock), video_available=(case / "VIDEO.mp4").is_file(),
                         real_closure_completed=bool(close_records), ever_passed_lift=bool(passed_lift), last_recorded_phase=last_phase,
                         grasp_evidence={"close": close_records, "lift": lift_records},
                         alignment_valid=recording.get("alignment_valid", False),
                         instruction=lock.get("episode_meta", {}).get("lang", ""),
                         task_binding=task_lookup[cell["task_id"]], result=result, supervisor=supervisor))
    groups = defaultdict(lambda: dict(planned=0, completed=0, policy_planned=0, policy_completed=0, success=0, native_budget_success=0,
                                     unsupported=0, policy_failure=0, technical_failure=0, videos=0, real_closure=0, passed_lift=0))
    for row in rows:
        for group in ("all", row["category"]):
            value = groups[group]
            value["planned"] += 1; value["completed"] += int(row["completed"])
            value["policy_planned"] += int(row["execution_mode"] == "run")
            value["policy_completed"] += int(row["completed"] and row["execution_mode"] == "run")
            value["success"] += int(row["pipeline_success"])
            value["native_budget_success"] += int(row["native_budget_success"])
            value["videos"] += int(row["video_available"])
            value["real_closure"] += int(row["real_closure_completed"])
            value["passed_lift"] += int(row["ever_passed_lift"])
            for field in ("unsupported", "policy_failure", "technical_failure"):
                value[field] += int(row["status"] == field)
    complete = groups["all"]["completed"] == 65
    return {"schema_version": "robocasa-atomic65-summary-1", "source_digest": matrix["source_digest"],
            "policy_digest": matrix["policy_digest"], "task_bindings_digest": matrix["task_bindings_digest"],
            "complete": complete, "seed": matrix["cells"][0]["seed"], "split": matrix["cells"][0]["split"],
            "budget_profile": matrix["budget_profile"], "work_deadline_s": matrix["work_deadline_s"],
            "groups": dict(groups), "rows": rows,
            "failure_counts": dict(Counter(r["failure_code"] for r in rows if r["completed"] and r["failure_code"])),
            "overall_success_rate": groups["all"]["success"] / 65 if complete else None,
            "shared_policy_success_rate": groups["all"]["success"] / groups["all"]["policy_planned"] if complete else None,
            "scope": "single-state coverage sweep; 22 shared-policy attempts and 43 unsupported native scene probes; no historical best-of or policy tuning during sweep"}


def first_frame(case, cameras):
    import h5py
    import numpy as np
    from PIL import Image
    path = case / "TRAJECTORY.h5"
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as f:
            views = [f["observations/" + camera + "_image"][0] for camera in cameras]
        buffer = io.BytesIO()
        Image.fromarray(np.concatenate(views, axis=1)).save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except (OSError, KeyError, IndexError):
        return None


def write_report(root, output):
    root, output = Path(root), Path(output)
    summary = summarize(root)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing report")
    policy = read(root / "POLICY.json")
    cards = []
    for row in summary["rows"]:
        case = root / "cases" / row["case"]
        poster = first_frame(case, policy["camera_names"]) if row["completed"] else None
        video = case / "VIDEO.mp4"
        if video.is_file() and row["completed"]:
            src = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
            media = '<video controls preload="none"' + (' poster="' + poster + '"' if poster else '') + ' src="' + src + '"></video>'
        elif poster:
            media = '<img alt="真实初始场景" src="' + poster + '">'
        else:
            media = '<p>无可用视频，保留初始化或进程记录。</p>'
        status_label = {"success": "完整成功", "unsupported": "未支持：只检查场景", "policy_failure": "未完成",
                        "technical_failure": "技术或预算退出", "running": "运行中", "pending": "排队中"}[row["status"]]
        details = {"task_binding": row["task_binding"], "result": row["result"], "supervisor": row["supervisor"]}
        note = '<p>本任务仅创建了真实场景，没有执行操作；短视频不代表抓取失败。</p>' if row["execution_mode"] == "probe" else ''
        wall = "—" if row["wall_s"] is None else "{:.1f} 秒".format(row["wall_s"])
        yes_no = lambda value: '是' if value else '否'
        cards.append('<article data-status="' + row["status"] + '"><h3>' + html.escape(row["task_id"]) + '</h3><p>' +
                     html.escape(row["instruction"]) + '</p><p><b>' + status_label + '</b> · ' + html.escape(row["reason"] or '') +
                     '</p><p>闭合完成：' + yes_no(row["real_closure_completed"]) + '；曾通过真实抬升保持：' + yes_no(row["ever_passed_lift"]) +
                     '；最后记录阶段：' + html.escape(row["last_recorded_phase"]) + '</p><p>真实动作 ' + str(row["physical_steps"]) +
                     ' 步；仿真动作 {:.2f} 秒；实际耗时 '.format(row["simulation_action_s"]) + wall + '；原生预算内成功：' +
                     yes_no(row["native_budget_success"]) + '</p>' + note + media + '<details><summary>原始结果与任务绑定</summary><pre>' +
                     html.escape(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details></article>')
    total = summary["groups"]["all"]
    category_rows = ''.join('<tr><td>' + CATEGORY_LABELS.get(key, key) + '</td><td>' + str(value["planned"]) + '</td><td>' +
        str(value["policy_planned"]) + '</td><td>' + str(value["success"]) + '</td><td>' + str(value["native_budget_success"]) +
        '</td><td>' + str(value["unsupported"]) + '</td><td>' + str(value["policy_failure"]) + '</td><td>' + str(value["technical_failure"]) +
        '</td></tr>' for key, value in summary["groups"].items() if key != "all")
    rate = ('全 65 项覆盖口径成功 {:.1%}；实际绑定共享策略的 22 项成功 {:.1%}。'.format(summary["overall_success_rate"], summary["shared_policy_success_rate"])
            if summary["complete"] else '尚未完成全部计划任务，暂不报告最终成功率。')
    reason_lookup = {row["failure_code"]: row["reason"] for row in summary["rows"] if row["failure_code"]}
    reason_lookup["UNSUPPORTED_SKILL"] = "尚未支持该类操作，仅检查真实初始场景"
    reasons = '<ul>' + ''.join('<li>' + html.escape(str(key)) + '：' + str(value) + ' 项；' + html.escape(reason_lookup[key] or '') + '</li>' for key, value in summary["failure_counts"].items()) + '</ul>'
    successes = '<ul>' + ''.join('<li>' + html.escape(row["task_id"]) + '：' + str(row["physical_steps"]) + ' 步；原生预算内完成：' + ('是' if row["native_budget_success"] else '否') + '</li>' for row in summary["rows"] if row["pipeline_success"]) + '</ul>'
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>RoboCasa 65 基础任务 · r10</title><style>body{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 18px;color:#18334a;line-height:1.6}'
        'article{border:1px solid #ccd7df;border-radius:12px;margin:20px 0;padding:20px}video,img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}button{padding:8px 14px;margin:4px;cursor:pointer}'
        '</style><h1>RoboCasa 65 个基础任务：当前 r10 规则覆盖验证</h1>'
        '<p>固定官方 commit 3443cdf3dd20f9ae8ed41adba442059b506891d0；每任务 pretrain、seed ' + str(summary["seed"]) + '，一次外层运行；全程冻结同一策略代码和参数。</p>'
        '<p><b>22 个共享抓放尝试，43 个未支持技能的真实场景检查。</b>后者不执行任务动作，不把 probe_passed 当作任务成功。完整 65 项保留在总体分母中，并另报 22 项条件结果。</p>'
        '<p>本轮使用显式 5000 步采集预算，另列原生步数内成功；每回合总上限 900 秒，其中最后 10 秒预留关闭与保存。全部抓放重新调用 GraspGen，无历史候选或成功结果复用。四张卡、每卡两个独立会话。</p>'
        '<p>r10 曾在 CounterToStove 和 SinkToCounter 上调试，其他任务也有源码/绑定接触，不能将本轮称为全部未见任务泛化。新增的是语义任务绑定，策略未改；SinkToCounter 的目的地按官方定义修正为台面上的 container。层架专用选择、底盘自动换位等能力仍有限制。</p>'
        '<p>已处理 ' + str(total["completed"]) + '/65；当前预算完整成功 ' + str(total["success"]) + '；原生预算内成功 ' + str(total["native_budget_success"]) +
        '；未支持 ' + str(total["unsupported"]) + '；策略失败 ' + str(total["policy_failure"]) + '；技术/预算退出 ' + str(total["technical_failure"]) + '。</p><p>' + rate + '</p>'
        '<p>这是每个任务一个固定初态的覆盖测试，比例表示本轮得到成功轨迹的任务占比，不能视为多初态平均成功率。未支持的 43 项没有进行任务操作，也没有被标成模型抓取失败。</p>'
        '<h2>本轮完整成功任务</h2>' + successes +
        '<p>真实完成闭合的回合：' + str(total["real_closure"]) + '；至少一次通过抬升保持检查的回合：' + str(total["passed_lift"]) + '。中间阶段通过不等于完整任务成功。</p>'
        '<p>任务分类为本报告按动作类型整理，最终成功一律使用官方原始判定。三个视频画面依次是左机位、右机位、腕部相机；无操作的回合仅保存真实初始画面。</p>'
        '<p>视频按真实环境动作记录。规划等待时仿真不前进，不增加动作帧；因此搜索耗时很长的回合，也可能只有初态或张开夹爪的短录像。短录像不会被解释成已经执行抓取。</p>'
        '<table><thead><tr><th>类型</th><th>任务</th><th>计划动作尝试</th><th>完整成功</th><th>原生预算内</th><th>未支持</th><th>策略失败</th><th>技术/预算</th></tr></thead><tbody>' + category_rows + '</tbody></table>'
        '<h2>失败分类</h2>' + reasons + '<p>源码指纹：' + summary["source_digest"] + '</p><p>任务绑定指纹：' + summary["task_bindings_digest"] + '</p>'
        '<h2>逐任务视频</h2><div><button onclick="filterRows(\'all\')">全部</button><button onclick="filterRows(\'success\')">成功</button>'
        '<button onclick="filterRows(\'policy_failure\')">策略失败</button><button onclick="filterRows(\'technical_failure\')">技术/预算</button><button onclick="filterRows(\'unsupported\')">未支持</button></div>' +
        ''.join(cards) + '<script>function filterRows(s){document.querySelectorAll("article").forEach(e=>e.hidden=s!=="all"&&e.dataset.status!==s)}</script></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    summary["report_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["report_sha256"] = hashlib.sha256(document.encode()).hexdigest()
    summary["embedded_videos"] = document.count("data:video/mp4;base64,")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = write_report(args.root, args.output) if args.output else summarize(args.root)
    print(json.dumps({"complete": summary["complete"], **summary["groups"]["all"], "failures": summary["failure_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
