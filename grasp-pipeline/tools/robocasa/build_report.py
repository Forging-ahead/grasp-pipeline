#!/usr/bin/env python3
"""Summarize one frozen matrix, retaining every missing or failed case and video."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import html
import hashlib
import json
import math
from pathlib import Path


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def wilson_interval(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def native_budget_success(result):
    if not result:
        return False
    limit = result.get("native_horizon")
    return bool(type(limit) in (int, float) and math.isfinite(limit) and limit > 0
                and result.get("pipeline_success") and result.get("native_success_at_end")
                and 0 < result.get("physical_steps", 0) <= limit)


def explain_failure(code, result):
    error = (result or {}).get("error", "")
    stage = (result or {}).get("failure_stage", "")
    if code == "SEARCH_BUDGET_EXHAUSTED":
        return "抓取前未找到可执行路径，尚未闭合夹爪" if "No executable grasp candidate" in error else "搬运或放置终点未找到可执行路径"
    horizon = (result or {}).get("runner_step_limit", (result or {}).get("native_horizon"))
    if code == "EPISODE_STOPPED" and isinstance(horizon, (int, float)) and (result or {}).get("physical_steps", 0) >= horizon:
        return "用完了显式采集配置的动作步数预算" if (result or {}).get("budget_profile") == "libero-v10-teacher" else "用完了原生任务的动作步数预算"
    if code == "invalid_start_state":
        return "规划器认为当前机器人姿态与障碍物冲突，拒绝继续；需核对球模型与真实接触"
    if code == "curobo_plan_failed":
        return "放下后的撤离路径规划失败" if "return_home" in error else "cuRobo 在本次预算内未找到通过检查的路径"
    if code == "COLLISION_REJECTED" and ("approach" in error or "approach" in stage):
        return "到达接近点后，最后靠近物体的路径未通过真实几何检查"
    return {"NOT_RUN": "尚未运行", "WALL_TIMEOUT": "用完了整回合时间预算", "PROCESS_ERROR": "运行进程异常退出", "PROCESS_INTERRUPTED": "回合被外部信号中断",
            "selection_budget_exhausted": "原版多轮候选搜索预算耗尽，尚未执行闭合",
            "checked_candidates_rejected": "原版已检查候选均未通过完整路径或双侧接触预检",
            "container_release_search_failed": "原版两段放置预检未找到可执行的释放方案",
            "path_preflight_rejected": "真实状态下的路径预检拒绝执行",
            "PATH_PREFLIGHT_REJECTED": "真实状态下的路径预检拒绝执行",
            "JOINT_BRANCH_DEVIATION": "实际手臂关节姿态偏离所规划的分支",
            "PAIRED_INPUT_MISMATCH": "配对初态或输入不一致，未使用旧候选继续运行",
            "waypoint_timeout": "原版反馈控制用完当前路点步数预算，仍未达到位姿容差",
            "motion_plan_failed": "原版阶段规划或实测状态下的接续规划失败",
            "VERSION_OR_IDENTITY_MISMATCH": "版本或任务身份不符，未计成功", "EXECUTION_TIMEOUT": "控制未在规定时间内到位",
            "JOINT_LIMIT": "关节限位或恢复路径未通过", "COLLISION_REJECTED": "真实几何检查拒绝了路径或动作",
            "MUJOCO_PREVIEW_REJECTED": "MuJoCo 真实几何预检未通过", "GRASP_CONTACT_FAILED": "闭合后未形成双侧抓取接触",
            "HOLD_FAILED": "抬升后未持续抓稳", "UNSUPPORTED_SKILL": "所需技能尚未开放",
            "PLACEMENT_UNSTABLE": "释放后未满足原生放置条件", "NATIVE_GOAL_FALSE": "最终原生任务目标尚未满足"}.get(code, code)


def summarize(root):
    root = Path(root)
    matrix = json.loads((root / "MATRIX.json").read_text())
    protocol = read_optional(root / "PROTOCOL.json") or {}
    policy = read_optional(root / "POLICY.json") or {}
    budget_profile = matrix.get("budget_profile", policy.get("budget_profile", "native"))
    if len(matrix["cells"]) != matrix["planned_episodes"]:
        raise ValueError("Matrix denominator differs from its declared cells")
    identifiers = [(c["task_id"], c["seed"]) for c in matrix["cells"]]
    if len(set(identifiers)) != len(identifiers) or not identifiers:
        raise ValueError("Matrix has duplicate or missing case identifiers")
    rows = []
    for cell in matrix["cells"]:
        name = cell["task_id"] + f"__seed{cell['seed']:06d}"
        if Path(name).name != name:
            raise ValueError("Invalid case identifier")
        directory = root / "cases" / name
        result = read_optional(directory / "RESULT.json")
        supervisor = read_optional(directory / "SUPERVISOR.json") or {}
        version_valid = bool(result and result.get("source_digest") == matrix["source_digest"] and
                         result.get("policy_digest") == matrix["policy_digest"] and
                         result.get("mode") == "run" and result.get("source_unchanged_during_run"))
        input_mode_valid = bool(result and bool(result.get("paired_diagnostic")) == bool(matrix.get("paired_diagnostic")))
        if result and result.get("budget_profile", "native") != budget_profile:
            input_mode_valid = False
        identity_valid = bool(result and all(result.get(k) == cell[k] for k in ("task_id", "seed", "split")))
        usable = version_valid and identity_valid and input_mode_valid and not supervisor.get("timed_out") and not supervisor.get("interrupted_signal")
        if result is None:
            failure = "NOT_RUN" if not supervisor else ("WALL_TIMEOUT" if supervisor.get("timed_out") else "PROCESS_ERROR")
        elif not version_valid or not identity_valid or not input_mode_valid:
            failure = "VERSION_OR_IDENTITY_MISMATCH"
        elif supervisor.get("timed_out"):
            failure = "WALL_TIMEOUT"
        elif supervisor.get("interrupted_signal"):
            failure = "PROCESS_INTERRUPTED"
        else:
            failure = result.get("failure_code") or (None if result.get("pipeline_success") else result.get("status", "UNKNOWN_FAILURE"))
        baseline_result = None
        if matrix.get("paired_diagnostic"):
            baseline_record = read_optional(Path(matrix["baseline_root"]) / "cases" / name / "RESULT.json")
            if baseline_record:
                baseline_result = {k: baseline_record.get(k) for k in ("source_digest", "pipeline_success", "native_success_at_end", "physical_steps", "failure_code", "error")}
        rows.append(dict(cell, case=name, version_valid=version_valid, identity_valid=identity_valid, input_mode_valid=input_mode_valid,
                    baseline_result=baseline_result,
                    attempted=result is not None or bool(supervisor),
                    native_success=bool(usable and result.get("native_success_at_end")),
                    pipeline_success=bool(usable and result.get("pipeline_success")),
                    native_budget_success=bool(usable and native_budget_success(result)),
                    training_candidate=bool(usable and result.get("training_candidate")),
                    failure=failure, video_available=(directory / "VIDEO.mp4").is_file(),
                    failure_explanation=explain_failure(failure, result) if failure else None,
                    physical_steps=0 if result is None else result.get("physical_steps", 0),
                    result=result, supervisor=supervisor))
    groups = defaultdict(lambda: {"planned": 0, "attempted": 0, "pending": 0, "invalid_version": 0,
                                  "pipeline_success": 0, "native_success": 0, "native_budget_success": 0, "training_candidate": 0})
    for row in rows:
        for key in ("all", "category:" + row["category"], "role:" + row["role"], "task:" + row["task_id"]):
            groups[key]["planned"] += 1
            groups[key]["attempted"] += int(row["attempted"])
            groups[key]["pending"] += int(row["failure"] == "NOT_RUN")
            groups[key]["invalid_version"] += int(row["failure"] == "VERSION_OR_IDENTITY_MISMATCH")
            for metric in ("pipeline_success", "native_success", "native_budget_success", "training_candidate"):
                groups[key][metric] += int(row[metric])
    for value in groups.values():
        value["complete"] = not value["pending"] and not value["invalid_version"]
        value["pipeline_success_per_planned"] = value["pipeline_success"] / value["planned"]
        value["pipeline_success_rate"] = value["pipeline_success_per_planned"] if value["complete"] else None
        value["wilson_95_percent"] = wilson_interval(value["pipeline_success"], value["planned"]) if value["complete"] else None
    task_groups = {key[5:]: value for key, value in groups.items() if key.startswith("task:")}
    summary = {"schema_version": "robocasa-summary-1", "source_digest": matrix["source_digest"],
               "paired_diagnostic": bool(matrix.get("paired_diagnostic")),
               "budget_profile": budget_profile,
               "experiment": matrix.get("experiment", "unspecified"), "task_exposure": protocol.get("actual_exposure", {}),
               "report_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "policy_digest": matrix["policy_digest"], "groups": dict(groups),
               "task_coverage": {"planned_tasks": len(task_groups),
                                 "tasks_with_pipeline_success": sum(v["pipeline_success"] > 0 for v in task_groups.values())},
               "failures": dict(Counter(row["failure"] for row in rows if row["failure"])),
               "failure_explanations": dict(Counter(row["failure_explanation"] for row in rows if row["failure_explanation"])),
               "rows": rows, "scope": "one frozen privileged-teacher matrix; no historical best-of aggregation"}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linked-videos", action="store_true", help="Use local relative video paths; ship the whole directory together")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a report")
    summary = summarize(args.root)
    cards = []
    for row in summary["rows"]:
        directory = args.root / "cases" / row["case"]
        video = directory / "VIDEO.mp4"
        if video.exists():
            if args.linked_videos:
                import os
                from urllib.parse import quote
                source = quote(os.path.relpath(video.resolve(), args.output.parent.resolve()))
            else:
                source = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
            media = '<video controls preload="none" src="' + html.escape(source, quote=True) + '"></video>'
        else:
            media = '<p>无有效视频；查看初始化、未运行或进程错误记录。</p>'
        label = "完整成功" if row["pipeline_success"] else row["failure_explanation"] or "未完成"
        details = html.escape(json.dumps({"result": row["result"], "supervisor": row["supervisor"], "paired_baseline": row["baseline_result"]}, ensure_ascii=False, indent=2))
        cards.append('<article><h2>' + html.escape(row["task_id"]) + ' · seed ' + str(row["seed"]) + '</h2><p>' +
                     html.escape(label) + '；真实动作 ' + str(row["physical_steps"]) + ' 步</p>' + media +
                     '<details><summary>完整结果</summary><pre>' + details + '</pre></details></article>')
    total = summary["groups"]["all"]
    boundary = ("本报告为开发任务验证；任务参与过代码调试，不能作为未见任务泛化的证据。" if summary["experiment"].startswith("E1_")
                else "本报告仅覆盖当前矩阵，不能与其他版本的历史最好结果合并。")
    if summary["paired_diagnostic"]:
        boundary = "本报告为已知失败开发初态的配对诊断。场景、初始 qpos、目标点云逐项核对，复用 r7 的原始 GraspGen 候选池；模型未重新推理，耗时不能当作新模型推理端到端耗时，也不能代表未见任务泛化。"
    budget = ("原生动作步数预算" if summary["budget_profile"] == "native" else
              "LIBERO v10 的显式 5000 步采集预算；超过 RoboCasa 原生 horizon 的成功不能算原生预算内成功，也不能与原生预算的成功率直接比较")
    progress = '已尝试 ' + str(total['attempted']) + '/' + str(total['planned']) + '，待运行 ' + str(total['pending'])
    if total["complete"]:
        lo, hi = total["wilson_95_percent"]
        progress += '；成功率 {:.1%}，95% Wilson 区间 {:.1%}–{:.1%}'.format(total["pipeline_success_rate"], lo, hi)
    else:
        progress += '；本矩阵尚未完整完成或存在版本不符，暂不报告最终成功率。'
    reasons = '<ul>' + ''.join('<li>' + html.escape(k) + '：' + str(v) + '</li>' for k, v in summary["failure_explanations"].items()) + '</ul>'
    task_table = '<table><thead><tr><th>任务</th><th>完整成功 / 计划回合</th><th>已尝试</th></tr></thead><tbody>' + ''.join(
        '<tr><td>' + html.escape(key[5:]) + '</td><td>' + str(value['pipeline_success']) + ' / ' + str(value['planned']) +
        '</td><td>' + str(value['attempted']) + '</td></tr>' for key, value in summary['groups'].items() if key.startswith('task:')) + '</tbody></table>'
    exposure = html.escape(json.dumps(summary["task_exposure"], ensure_ascii=False, indent=2))
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RoboCasa 接入验证报告</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;color:#18334a}'
        'article{border:1px solid #cdd7df;border-radius:12px;padding:20px;margin:20px 0}video{width:100%}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:20px}table{border-collapse:collapse;width:100%}'
        'td,th{padding:10px;border:1px solid #cdd7df;text-align:left}</style>'
        '<h1>RoboCasa 接入验证报告</h1><p>仿真真值 teacher 条件；单一冻结版本；保留全部计划回合。</p>'
        '<p>' + boundary + '</p><p>实验：' + html.escape(summary['experiment']) + '</p><p>' + progress + '</p>'
        '<p>动作预算：' + budget + '</p><p>当前预算下完整成功 ' + str(total['pipeline_success']) +
        '；原生最终目标成立 ' + str(total['native_success']) + '；原生步数内完整成功 ' + str(total['native_budget_success']) + '</p>' +
        task_table + '<p>额外诊断演示不计入本矩阵；成功轨迹仍需审核目标可见性、工具标签和位姿，不能直接视为已通过 SFT 数据审核。</p>'
        '<p>视频从左至右为同一机器人的左机位、右机位、腕部相机。没有执行抓取的回合可能仅有初始或张开夹爪的画面。</p>'
        '<p>源码指纹：' + html.escape(summary['source_digest']) + '</p><h2>失败原因</h2>' + reasons +
        '<details><summary>任务接触与迭代范围</summary><pre>' + exposure + '</pre></details>' + ''.join(cards) + '</html>')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "planned": total["planned"], "pipeline_success": total["pipeline_success"]}))


if __name__ == "__main__":
    main()
