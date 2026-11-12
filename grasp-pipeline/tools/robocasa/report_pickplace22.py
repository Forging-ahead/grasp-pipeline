#!/usr/bin/env python3
"""Audit recordings and build an offline paired-video r10/r13 report."""
import argparse
import base64
import datetime
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

from pickplace22_records import read, summarize


def audit_recordings(root, summary):
    import h5py
    import imageio_ffmpeg
    import numpy as np
    policy = read(root / "POLICY.json")
    checks = []
    for row in summary["rows"]:
        if not row["completed"]:
            continue
        case = root / "cases" / row["case"]
        video = case / "VIDEO.mp4"
        digest = hashlib.sha256(video.read_bytes()).hexdigest() if video.is_file() else None
        cache = read(case / "VIDEO_AUDIT.json")
        if cache and cache.get("sha256") == digest:
            media = cache
        else:
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                capture_output=True, text=True, timeout=120) if video.is_file() else None
            media = {"sha256": digest, "decoded": bool(result is not None and result.returncode == 0),
                "bytes": video.stat().st_size if video.is_file() else 0,
                "error": result.stderr[-1000:] if result is not None and result.returncode else None}
            (case / "VIDEO_AUDIT.json").write_text(json.dumps(media, indent=2) + "\n")
        errors = []
        try:
            with h5py.File(case / "TRAJECTORY.h5", "r") as trajectory:
                n = row["physical_steps"]
                if (len(trajectory["actions"]) if "actions" in trajectory else 0) != n:
                    errors.append("action_count")
                for camera in policy["camera_names"]:
                    for suffix in ("_image", "_depth", "_segmentation_instance"):
                        key = "observations/" + camera + suffix
                        if key not in trajectory or len(trajectory[key]) != n + 1:
                            errors.append(key)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        generators = [read(p) or {} for p in case.glob("stages/*/grasp_*/GENERATOR.json")]
        cached = any(g.get("backend") == "recorded_identical_graspgen_pool" for g in generators)
        input_match = pool_match = None
        clouds = sorted(case.glob("stages/*/grasp_00/CLOUD.npz"))
        if clouds:
            old_cloud = Path(row["baseline_case"]) / clouds[0].relative_to(case)
            if old_cloud.is_file():
                with np.load(clouds[0], allow_pickle=False) as a, np.load(old_cloud, allow_pickle=False) as b:
                    input_match = bool(np.array_equal(a["points_world"], b["points_world"]))
                current_pool, old_pool = clouds[0].with_name("CANDIDATES.npz"), old_cloud.with_name("CANDIDATES.npz")
                if current_pool.is_file() and old_pool.is_file():
                    with np.load(current_pool, allow_pickle=False) as a, np.load(old_pool, allow_pickle=False) as b:
                        pool_match = bool(np.array_equal(a["poses_world_grasp"], b["poses_world_grasp"]) and np.array_equal(a["scores"], b["scores"]))
        baseline_video = Path(row["baseline_case"]) / "VIDEO.mp4"
        baseline_audit = read(baseline_video.with_name("VIDEO_AUDIT.json")) or {}
        baseline_media_valid = bool(baseline_video.is_file() and baseline_audit.get("decoded") and
            hashlib.sha256(baseline_video.read_bytes()).hexdigest() == baseline_audit.get("sha256"))
        checks.append({"task_id": row["task_id"], "identity_valid": row["identity_valid"],
            "initial_state_verified": row["initial_state_verified"], "video": media,
            "alignment_valid": row["alignment_valid"] and not errors, "recording_errors": errors,
            "used_cached_candidates": cached, "fresh_inference_calls": sum(g.get("generation_calls", 0) for g in generators),
            "first_target_cloud_identical_to_baseline": input_match, "first_fresh_pool_identical_to_baseline": pool_match,
            "baseline_video_still_matches_original_decoding_audit": baseline_media_valid})
    return {"checked_cases": len(checks), "decoded_videos": sum(c["video"]["decoded"] for c in checks),
        "aligned_recordings": sum(c["alignment_valid"] for c in checks),
        "all_identities_valid": all(c["identity_valid"] for c in checks),
        "all_initial_states_verified": all(c["initial_state_verified"] for c in checks),
        "used_cached_candidates": any(c["used_cached_candidates"] for c in checks),
        "fresh_inference_calls": sum(c["fresh_inference_calls"] for c in checks),
        "identical_first_clouds": sum(c["first_target_cloud_identical_to_baseline"] is True for c in checks),
        "different_first_clouds": sum(c["first_target_cloud_identical_to_baseline"] is False for c in checks),
        "identical_fresh_pools": sum(c["first_fresh_pool_identical_to_baseline"] is True for c in checks),
        "baseline_videos_valid": sum(c["baseline_video_still_matches_original_decoding_audit"] for c in checks), "cases": checks}


def make_html(root, summary):
    from report_atomic65 import first_frame
    policy = read(root / "POLICY.json")
    labels = {"success": "完整成功", "policy_failure": "策略未完成", "technical_failure": "技术/预算退出",
              "pending": "待运行", "running": "运行中"}
    def media(case):
        poster = first_frame(case, policy["camera_names"])
        path = case / "VIDEO.mp4"
        if not path.is_file():
            return '<p>没有有效视频，保留原始技术错误记录。</p>'
        source = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
        return '<video controls preload="none"' + (' poster="' + poster + '"' if poster else '') + ' src="' + source + '"></video>'
    cards, table = [], []
    for row in summary["rows"]:
        old = row["baseline_result"]
        table.append('<tr><td>' + html.escape(row["task_id"]) + '</td><td>' + ('成功' if row["baseline_success"] else html.escape(old.get("failure_code", "未完成"))) +
            '</td><td>' + labels[row["status"]] + '</td><td>' + str(row["physical_steps"]) + '</td><td>' + ('是' if row["native_budget_success"] else '否') + '</td></tr>')
        raw = {"new": row["result"], "baseline": old, "initial_state_verified": row["initial_state_verified"],
               "reused_start_rejections": row["reused_start_rejections"], "uncached_start_rejections": row["uncached_start_rejections"]}
        wall = '—' if row["wall_s"] is None else '{:.1f}'.format(row["wall_s"])
        cards.append('<article data-status="' + row["status"] + '"><h3>' + html.escape(row["task_id"]) + '</h3><p>' + html.escape(row["instruction"]) +
            '</p><p><b>' + labels[row["status"]] + '</b>：' + html.escape(row["reason"] or '') + '</p><p>新版本：' + str(row["physical_steps"]) +
            ' 个动作，仿真动作 {:.2f} 秒，实际耗时 '.format(row["simulation_action_s"]) + wall + ' 秒；原生预算内成功：' + ('是' if row["native_budget_success"] else '否') +
            '；完整初态核对：' + ('一致' if row["initial_state_verified"] else '未通过') + '</p><p>最后记录阶段：' + html.escape(row["last_recorded_phase"]) +
            '；起点拒绝复用 ' + str(row["reused_start_rejections"]) + ' 次，实际查询的起点拒绝 ' + str(row["uncached_start_rejections"]) + ' 次。</p>' +
            '<div class="pair"><div><p>旧 r10 · 900 秒</p>' + media(Path(row["baseline_case"])) + '</div><div><p>新 r13 · 1200 秒</p>' + media(root / "cases" / row["case"]) +
            '</div></div><details><summary>原始结果与核对信息</summary><pre>' + html.escape(json.dumps(raw, ensure_ascii=False, indent=2)) + '</pre></details></article>')
    rate = '尚未完成' if summary["success_rate"] is None else '{:.1%}'.format(summary["success_rate"])
    failures = '<ul>' + ''.join('<li>' + html.escape(k) + '：' + str(v) + '</li>' for k, v in summary["failure_counts"].items()) + '</ul>'
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>RoboCasa 22 项抓放：r10 与 r13 修复复测</title><style>body{font:16px system-ui;line-height:1.6;max-width:1280px;margin:32px auto;padding:0 18px;color:#17364b}'
        'article{border:1px solid #ccd6de;border-radius:12px;padding:18px;margin:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6de;padding:7px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'button{margin:4px;padding:8px 12px}@media(max-width:760px){.pair{grid-template-columns:1fr}}</style>'
        '<h1>RoboCasa 22 项抓放：修复后的同版复测</h1><p><b>新版本完整成功 ' + str(summary["success"]) + '/22（' + rate + '），原生步数内成功 ' + str(summary["native_budget_success"]) +
        '；旧版本完整成功 ' + str(summary["baseline_success"]) + '/22，原生步数内成功 ' + str(summary["baseline_native_budget_success"]) + '。</b></p>'
        '<p>完成 ' + str(summary["completed"]) + '/22；初态核对通过 ' + str(summary["initial_states_verified"]) + '/22；实际闭合 ' + str(summary["real_closures"]) +
        ' 项，至少一次通过抬升保持 ' + str(summary["passed_lift"]) + ' 项。中间阶段不计完整任务成功。</p>'
        '<p>本轮只重跑上轮实际执行过的同一 22 个任务，pretrain、seed 0，各一次外层运行。运行前比较原生场景、成功函数、动作布局及所有保存的初始物理数组；不一致时不执行任务动作。全部候选重新调用 GraspGen，不用旧成功替换新失败。</p>'
        '<p>开发阶段 r11 因持物刷新接口错误、r12 因失败结果中的矩阵无法保存而中止，两批已单独保留并排除；本报告的新结果全部来自最终冻结 r13 的完整 22 项。</p>'
        '<p>两版均使用 5000 步采集预算；r13 将总墙钟上限由 900 恢复到 1200 秒，1190 秒停止工作并预留 10 秒保存。原生 horizon 与成功判定不变，超出原生步数的成功单列。每任务一个固定初态，比例不是多初态 benchmark 平均成功率。</p>'
        '<h2>本轮修正</h2><ol><li>抬升前向 cuRobo 登记持物附件，抬升保持后显式 detach/attach 刷新规划几何，物理夹爪持续闭合；恢复原版初始与当前支撑接触的交接，支撑许可只用于全局抬升回退，原生几何预检仍检查机器人碰撞。</li>'
        '<li>恢复原版默认最多 4 个放置物体朝向，继续使用原搬运高度、偏航和放置搜索。</li><li>完全相同的场景、关节/手指状态、附件与接触配置下，只查询一次无效起点；每个目标仍执行原生局部回退，状态或场景改变后重新查询。</li>'
        '<li>恢复 1200 秒总预算，并在动作前核对完整初态，统一冻结源码运行全部 22 项。</li><li>失败结果复用原生矩阵/标量转换与原子 JSON 保存，保留真正的失败原因，避免结果记录错误被误计为策略结果。</li></ol>'
        '<p>这是组合修复的工程对照：GraspGen 候选仍有随机性，墙钟预算也发生变化，不能把每项改善归因于某一个改动。分割、位姿和规划几何仍来自仿真 teacher 信息；底盘及未验证技能没有开放。</p>'
        '<p>新增成功任务：' + html.escape('、'.join(summary["newly_successful"]) or '无') + '。原成功回合本轮未成功：' + html.escape('、'.join(summary["regressed"]) or '无') + '。</p>'
        '<table><thead><tr><th>任务</th><th>旧 r10</th><th>新 r13</th><th>新动作步数</th><th>原生预算内成功</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table>'
        '<h2>新版本失败出口</h2>' + failures + '<p>r13 源码指纹：' + summary["source_digest"] + '</p><p>r10 源码指纹：' + summary["baseline_source_digest"] + '</p>'
        '<h2>逐项视频对照</h2><p>每段依次为左机位、右机位、腕部相机。规划等待时物理仿真不前进，因此动作录像可能很短；短录像不代表已经执行抓取。可单独转发本 HTML，全部视频已内嵌。</p>'
        '<button onclick="filterRows(\'all\')">全部</button><button onclick="filterRows(\'success\')">新成功</button><button onclick="filterRows(\'policy_failure\')">策略未完成</button><button onclick="filterRows(\'technical_failure\')">技术/预算退出</button>' +
        ''.join(cards) + '<script>function filterRows(s){document.querySelectorAll("article").forEach(e=>e.hidden=s!=="all"&&e.dataset.status!==s)}</script></html>')


def compact(summary):
    return {k: summary[k] for k in ("complete", "completed", "success", "native_budget_success", "initial_states_verified", "status_counts", "failure_counts", "reused_start_rejections")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        summary = summarize(args.root)
        print(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **compact(summary)}, ensure_ascii=False), flush=True)
        if not args.watch or summary["complete"]:
            break
        time.sleep(45)
    if args.output or args.audit:
        if not summary["complete"]:
            parser.error("Wait for all 22 cases before final audit/report")
        if args.audit:
            if args.audit.exists(): parser.error("Audit already exists")
            audit = audit_recordings(args.root, summary)
            args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
            valid = audit["checked_cases"] == audit["decoded_videos"] == audit["aligned_recordings"] == 22
            if not valid or not audit["all_identities_valid"] or not audit["all_initial_states_verified"] or audit["used_cached_candidates"] or audit["baseline_videos_valid"] != 22:
                raise RuntimeError("Final media/identity/initial-state audit did not pass")
        if args.output:
            if args.output.exists(): parser.error("Report already exists")
            document = make_html(args.root, summary)
            args.output.write_text(document)
            summary.update(report_sha256=hashlib.sha256(document.encode()).hexdigest(), embedded_videos=document.count("data:video/mp4;base64,"))
            args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
