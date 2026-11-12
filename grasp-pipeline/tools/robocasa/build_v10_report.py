#!/usr/bin/env python3
"""Offline video report for final same-source migration checks, by budget."""
import argparse
import base64
import html
import json
from pathlib import Path

from build_report import summarize, explain_failure


def read(path):
    return json.loads(path.read_text())


def video(path):
    if not path.is_file():
        return '<p>没有有效视频文件。</p>'
    return '<video controls preload="none" src="data:video/mp4;base64,' + base64.b64encode(path.read_bytes()).decode() + '"></video>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--teacher-pair-root", type=Path, required=True)
    parser.add_argument("--fresh-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    native, paired = summarize(args.native_root), summarize(args.teacher_pair_root)
    fresh = read(args.fresh_case / "RESULT.json")
    source = native["source_digest"]
    if source != paired["source_digest"] or source != fresh["source_digest"]:
        parser.error("All sections must use the same final actor source")
    if (native["budget_profile"] != "native" or paired["budget_profile"] != "libero-v10-teacher"
            or fresh.get("paired_diagnostic") or fresh.get("candidate_source") != "fresh_graspgen_inference"
            or not fresh.get("source_unchanged_during_run")):
        parser.error("Unexpected budget or candidate-source conditions")
    if not native["groups"]["all"]["complete"] or not paired["groups"]["all"]["complete"]:
        parser.error("Do not publish final counts before all planned cells complete")
    groups = [("原生预算：已知失败初态回归", native, args.native_root),
              ("5000 步采集预算：相同候选池诊断", paired, args.teacher_pair_root)]
    sections = []
    for title, summary, root in groups:
        total = summary["groups"]["all"]
        content = ['<section><h2>' + title + '</h2><p>当前预算完整成功 ' + str(total["pipeline_success"]) + '/' +
                   str(total["planned"]) + '；原生步数内完整成功 ' + str(total["native_budget_success"]) + '/' + str(total["planned"]) + '。</p>']
        for row in summary["rows"]:
            case = root / "cases" / row["case"]
            result = row["result"]
            lock = read(case / "ENVIRONMENT_LOCK.json")
            label = "完整成功" if row["pipeline_success"] else row["failure_explanation"]
            content += ['<article><h3>' + html.escape(row["task_id"]) + ' · seed ' + str(row["seed"]) + '</h3>',
                        '<p>' + html.escape(lock["episode_meta"]["lang"]) + '</p>',
                        '<p>' + html.escape(label) + '；动作 ' + str(row["physical_steps"]) + ' 步；耗时 {:.1f} 秒。</p>'.format(result.get("wall_s", 0)),
                        video(case / "VIDEO.mp4"), '<details><summary>原始结果与基线</summary><pre>' +
                        html.escape(json.dumps({"result": result, "baseline": row["baseline_result"]}, ensure_ascii=False, indent=2)) + '</pre></details></article>']
        content.append('</section>'); sections.append(''.join(content))
    lock = read(args.fresh_case / "ENVIRONMENT_LOCK.json")
    sections.append('<section><h2>5000 步采集预算：重新运行 GraspGen 的额外演示</h2><p>这是额外开发验证，不计入上面的原生预算回归或配对分母。</p><article><h3>' +
                    html.escape(fresh["task_id"]) + ' · seed ' + str(fresh["seed"]) + '</h3><p>' + html.escape(lock["episode_meta"]["lang"]) +
                    '</p><p>完整成功：' + str(bool(fresh.get("pipeline_success"))) + '；原生最终目标：' + str(bool(fresh.get("native_success_at_end"))) +
                    '；原生步数内成功：' + str(bool(fresh.get("native_budget_success"))) + '；动作 ' + str(fresh["physical_steps"]) +
                    ' 步；耗时 {:.1f} 秒。</p>'.format(fresh["wall_s"]) + video(args.fresh_case / "VIDEO.mp4") +
                    '<details><summary>原始结果</summary><pre>' + html.escape(json.dumps(fresh, ensure_ascii=False, indent=2)) + '</pre></details></article></section>')
    doc = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>LIBERO v10 → RoboCasa 迁移验证</title>'
           '<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#18334a;line-height:1.6}article{border:1px solid #ccd7df;border-radius:12px;padding:20px;margin:20px 0}video{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:40px}</style>'
           '<h1>基于成熟 LIBERO v10 规则的 RoboCasa 验证</h1><p>2026-11-12 · 最终代码 r10 · 仿真真值 teacher 条件。</p>'
           '<p>实际复用原版候选筛选、局部接触规划、双侧接触检查、搬运和两段放置，以及原版反馈控制与积分补偿；原 LIBERO 代码未修改。目标区域来自 RoboCasa 原生定义，不使用逐任务/seed 世界坐标。</p>'
           '<p><strong>三种条件分别报告，不合并成功率。</strong>原生回归只有三个已知失败开发初态；5000 步是显式的数据采集条件，超过原生 600 步的成功不能当作原生预算内成功。未重新测完整十回合，也未测未见任务泛化。</p>'
           '<p>配对回合核对场景、初始 qpos 和目标点云后，复用原始候选池；额外演示重新运行 GraspGen。配对仍含规划器运行波动，不是严格单组件消融。所有视频内嵌，可单独转发此 HTML。</p>'
           '<p>三个画面依次为左机位、右机位、腕部相机。关键点来自 code policy，训练数据审核尚未完成。</p><p>最终源码指纹：' + source + '</p>' + ''.join(sections) + '</html>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(doc)
    args.output.with_suffix(".summary.json").write_text(json.dumps({"source_digest": source,
        "native_budget_regression": native, "teacher_paired_diagnostic": paired, "extra_fresh_teacher": fresh,
        "no_combined_success_rate": True, "ready_for_sft": False}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "embedded_videos": doc.count("data:video/mp4;base64,")}))


if __name__ == "__main__":
    main()
