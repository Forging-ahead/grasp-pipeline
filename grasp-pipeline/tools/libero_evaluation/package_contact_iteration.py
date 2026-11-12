#!/usr/bin/env python3
"""Audit contact-policy episodes and build an offline, embedded-video report.

Only init 0 is paired against the immediately preceding iteration. Additional
initial states are reported separately, with no historical success substitution.
"""
import argparse
import base64
import collections
import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from run_all import read, dump, identity, valid_result
from package_subset import failure, sha, LABELS
LABELS.update(surface_push_stalled='推动后机构不再进展',surface_push_ik_failed='推动机构的后续预检未通过',
    surface_push_goal_not_reached='推动机构未达到原始目标',regrasp_no_progress='重夹后关节进度不足',
    regrasp_budget_exhausted='重夹预算内未完成开门/通行条件',contact_global_start_invalid='cuRobo代理模型把当前起点判为不可规划',
    access_aperture_not_retained='松手后未保持取放所需通行开度')


def render(rows, summary, out):
    h=html.escape
    cards=[]
    for row in rows:
        videos=[]
        for key,title in [('before_video','上一版，同一初态'),('after_video','本轮冻结版')]:
            if not row.get(key):continue
            data=base64.b64encode((out/row[key]).read_bytes()).decode()
            videos.append(f'<div><p>{title}</p><video controls playsinline preload="none" src="data:video/mp4;base64,{data}"></video></div>')
        record=json.dumps({k:row[k] for k in ['queue','failure','result','audits']},ensure_ascii=False,indent=2)
        cards.append(f'''<details class="case" data-search="{h((row['key']+' '+row['language']+' '+row['category_name']).lower(),quote=True)}" data-status="{row['change']}">
<summary><b>{h(row['language'])}</b> · 初态{row['initial_state_id']} · {h(row['change_label'])}</summary>
<section><p>{h(row['category_name'])} · {h(row['key'])}</p><div class="videos">{''.join(videos)}</div>
<p>{h(row['failure']['label'])}；{row['result']['steps']}步，仿真动作时间{row['result']['steps']/20:.2f}秒，实际耗时{row['queue']['wall_seconds']/60:.2f}分钟。</p>
<details><summary>原始结果与接触诊断</summary><pre>{h(record)}</pre></details></section></details>''')
    table=[]
    for group in summary['categories']:
        table.append(f"<tr><td>{h(group['name'])}</td><td>{group['initial_state_id']}</td><td>{group['tasks']}</td><td>{group['before_success'] if group['before_success'] is not None else '—'}</td><td>{group['after_success']}</td></tr>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO机构接触规则迭代</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#20374a;font-family:system-ui,-apple-system,"PingFang SC",sans-serif;line-height:1.7}main{max-width:1250px;margin:auto;padding:30px 20px}h1{line-height:1.3}.notice{background:#fff5df;border-left:4px solid #d29b31;padding:15px}table{width:100%;background:white;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce3e9}.case{background:white;border:1px solid #dce3e9;border-radius:9px;margin:15px 0;overflow:hidden}summary{padding:15px;cursor:pointer}section{padding:0 15px 20px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:15px}video{width:100%;background:#111;border-radius:6px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:600px;overflow:auto;background:#edf2f6;padding:12px}input,select{font:inherit;padding:9px;border:1px solid #bdcbd5;border-radius:5px}.filters{display:flex;gap:12px;margin:20px 0}input{flex:1}.hidden{display:none}small{color:#637989}@media(max-width:750px){.videos{grid-template-columns:1fr}.filters{flex-wrap:wrap}h1{font-size:26px}}
</style><main><h1>LIBERO：机构接触与反馈控制迭代</h1>'''
    paired=summary['paired_init0']
    page+=f'''<p><b>相同初态0的{paired['tasks']}项对照：上一版{paired['before_success']}项通过，本轮{paired['after_success']}项通过；新增{paired['improved']}项，退步{paired['regressed']}项。</b></p>
<p class="notice">这是机构操作及相关复合任务的定向复验，不能视为新版130项成功率。当前所有新回合使用同一冻结源码；开发阶段的诊断、部分完成和历史成功均未补入本轮结果。其他初态单独列出，不与初态0混算。每段视频已嵌入本文件，可以单文件转发。</p>
<h2>修改内容与适用范围</h2><ul><li>把手闭爪后的运动预检使用闭合手指；执行阶段的短程预检也同步机构运动，避免把门固定在旧角度后误报碰撞。</li>
<li>独立开门以官方阈值内的目标为准，需要取放通行时保留更大开度。整段不可行时，有限次尝试分段松手、重夹，并检查真实关节进度。</li>
<li>原夹把手方案不可行的关闭任务，补充从活动面板推闭的候选。推点由实时面板几何重算，末段控制容差随剩余位移收紧；官方目标达成后退让并复查。</li>
<li>松手后补充经检查的退让方向和腕部旋转自由度。保存重置后的模型、精确模型参数及完整积分输入，以便重现失败。</li></ul>
<p>以上仍是依赖仿真分割、位姿、碰撞几何及任务定义的手写code policy。没有新增按任务名指定的世界坐标，没有修改GraspGen权重或评分顺序，没有删除障碍物或放宽原碰撞容差。规划预检会在快照内改变机器人和机构状态并恢复；真实执行通过控制动作和物理接触完成。</p>
<p>失败退出可能来自接近、接触保持、后续机构运动、退让，或复合任务中的刚体抓放。规划拒绝不代表物理空间无解。包含GraspGen的复合任务会重新推理，故这不是固定抓取候选的单组件消融。</p>
<h2>各类结果</h2><table><tr><th>类别</th><th>初态</th><th>任务数</th><th>上一版</th><th>本轮</th></tr>{''.join(table)}</table>
<h2>逐项视频</h2><div class="filters"><select id="status"><option value="">全部</option><option value="improved">新增通过</option><option value="retained">保持通过</option><option value="failed">仍失败</option><option value="regressed">退步</option><option value="additional">其他初态</option></select><input id="search" placeholder="搜索任务、物体、类别"></div><p id="count"></p>{''.join(cards)}
<h2>验证说明</h2><p>{summary['cpu_tests']}项回归通过。任务源码、初态资产、原始目标、初态0起始状态和原数字配置均已核对；连续轨迹、模型快照与视频解码检查通过。视频末尾增加2秒静帧，没有加入未执行动作。成功回合仍需按指令过程、接触质量与学生观测条件审核后，才可作为训练数据。</p>
<details><summary>完整汇总</summary><pre>{h(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>
<small>源码 {h(summary['source_revision'])}；教师策略继续使用仿真真值，尚未验证真实感知或真机迁移。</small></main>'''
    page+='''<script>const cards=[...document.querySelectorAll('.case')],status=document.querySelector('#status'),search=document.querySelector('#search');function filter(){let n=0;for(const c of cards){const show=(!status.value||c.dataset.status===status.value)&&c.dataset.search.includes(search.value.trim().toLowerCase());c.classList.toggle('hidden',!show);if(show)n++;else for(const v of c.querySelectorAll('video'))v.pause();}document.querySelector('#count').textContent=`显示 ${n}/${cards.length} 回合`;}status.oninput=search.oninput=filter;for(const v of document.querySelectorAll('video'))v.onplay=()=>{for(const other of document.querySelectorAll('video'))if(other!==v)other.pause();};filter();</script></html>'''
    path=out/'LIBERO_机构接触规则迭代_视频报告.html';path.write_text(page)
    return path


def main():
    import numpy as np
    import imageio_ffmpeg
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations',type=Path,nargs='+',required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--external-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve();before={x['key']:x for x in read(baseline/'RESULTS.json')}
    suites=ET.parse(args.tests).getroot();assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
    tests=sum(int(x.attrib['tests']) for x in suites)
    jobs=[];protocols=[];manifests=[]
    for evaluation in args.evaluations:
        protocol=read(evaluation/'PROTOCOL.json');manifest=read(evaluation/'SOURCE_IDENTITY.json')
        assert identity(Path(protocol['source']))==manifest
        queue=read(evaluation/'SUBSET_COMPLETE.json');catalog=read(evaluation/'ALL_TASKS.json')
        assert queue['source_unchanged'] and queue['total']==len(catalog)==len(queue['results'])
        protocols.append(protocol);manifests.append(manifest)
        for task in catalog:jobs.append((evaluation,task,protocol,manifest))
    assert len({x['source_digest'] for x in protocols})==len({x['revision'] for x in protocols})==1
    assert len({(x[1]['key'],x[2]['initial_state_id']) for x in jobs})==len(jobs)
    with tarfile.open(args.source_archive) as archive:
        for name,expected in manifests[0]['files'].items():
            assert hashlib.sha256(archive.extractfile('grasp-pipeline/'+name).read()).hexdigest()==expected,name
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    def process(item):
        evaluation,task,protocol,manifest=item;key=task['key'];init=protocol['initial_state_id']
        src=evaluation/'runs/subset'/key;dst=out/'cases'/('init_'+str(init))/key;dst.mkdir(parents=True)
        result,queue=read(src/'RESULT.json'),read(src/'QUEUE_RECORD.json');previous=before[key]
        assert valid_result(result,task,manifest,initial_state_id=init) and queue['result_identity_valid'] and queue['source_unchanged'],key
        assert queue['source_digest']==manifest['source_digest']==protocol['source_digest']
        env,cfg=read(src/'environment/ENVIRONMENT.json'),read(src/'environment/config.json')
        assert env['init_state_id']==cfg['init_state_id']==init and cfg['seed']==0 and cfg['max_steps']==5000 and cfg['settle_steps']==10
        assert result['original_goals']==previous['result']['original_goals'] and result['original_goals_unchanged']
        assert sha(Path(env['bddl_file']))==previous['original_bddl_sha256']
        assert sha(Path(env['init_state_file']))==previous['initial_state_file_sha256']
        states=read(src/'states.json');old_cfg=read(baseline/'cases'/key/'environment/config.json')
        dump(dst/'INITIAL_CONFIG_BEFORE.json',old_cfg)
        expected=dict(read(evaluation/'FROZEN_CONFIGS.json')[key])
        for name in ['simulation_gpu','planner_gpu','model_gpu']:expected[name]=queue['gpu']
        assert all(cfg.get(k)==v for k,v in expected.items())
        deployment={'simulation_gpu','planner_gpu','model_gpu','init_state_id'}
        assert {k:v for k,v in cfg.items() if k not in deployment}=={k:v for k,v in old_cfg.items() if k not in deployment}
        if init==0:
            old_state=read(baseline/'cases'/key/'states.json')[0]
            for name in ['q_arm','q_fingers','qvel_arm','qvel_fingers','world_T_eef','world_T_base']:
                np.testing.assert_allclose(states[0][name],old_state[name],atol=1e-10,rtol=0)
            for name in states[0]['object_poses']:
                np.testing.assert_allclose(states[0]['object_poses'][name],old_state['object_poses'][name],atol=1e-10,rtol=0)
            dump(dst/'INITIAL_STATE_BEFORE.json',old_state)
        with np.load(src/'trajectory.npz',allow_pickle=False) as trace:
            n=len(trace['actions']);assert n==result['steps'] and len(states)==n+1
            for name,shape in [('actions',(n,7)),('q_arm',(n+1,7)),('q_fingers',(n+1,2)),('world_T_eef',(n+1,4,4))]:
                assert trace[name].shape==shape and np.isfinite(trace[name]).all()
        np.testing.assert_allclose(np.diff([x['sim_time'] for x in states]),.05,atol=1e-9,rtol=0)
        for name in ['RESULT.json','QUEUE_RECORD.json','states.json','trajectory.npz','FINAL_PHYSICS_STATE.npz','SKILL_PLAN.json','PACKING_PLAN.json','preview.png']:
            if (src/name).is_file():shutil.copy2(src/name,dst/name)
        shutil.copytree(src/'environment',dst/'environment',ignore=shutil.ignore_patterns('unused_datasets','libero_config','*.log'))
        model_record=read(dst/'environment/MODEL_SNAPSHOT.json')
        assert model_record['phase']=='after_official_reset_and_settle' and not model_record['live_physics_modified']
        for name,record in model_record['files'].items():assert sha(dst/'environment'/name)==record['sha256']
        shutil.copy2(env['bddl_file'],dst/'task.bddl');shutil.copy2(evaluation/'logs'/(key+'.log'),dst/'driver.log')
        for stage in src.glob('stage_*'):
            for path in stage.rglob('*.json'):
                relative=path.relative_to(src);target=dst/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        after=dst/'after.mp4'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','2','-i',str(src/'rollout.mp4'),'-an','-vf','tpad=stop_mode=clone:stop_duration=2','-r','20','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(after)],check=True,capture_output=True)
        decoded=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','2','-i',str(after),'-f','framehash','-'],check=True,capture_output=True,text=True)
        frames=[x for x in decoded.stdout.splitlines() if x and not x.startswith('#')];assert len(frames)==n+41
        old_success=bool(previous['after_success']) if init==0 else None
        passed=bool(not queue['watchdog_reached'] and result.get('pipeline_success') and result.get('official_goal_at_end') and n>0)
        assert passed==queue['pipeline_success']
        change=('retained' if passed and old_success else 'improved' if passed else 'regressed' if old_success else 'failed') if init==0 else 'additional'
        before_video=None
        if init==0:
            video=baseline/previous['after_video'];assert sha(video)==previous['after_video_sha256']
            shutil.copy2(video,dst/'before.mp4');before_video=str((dst/'before.mp4').relative_to(out))
        audits=[dict(path=str(f.relative_to(src)),data=read(f)) for stage in src.glob('stage_*')
            for f in stage.rglob('*.json') if f.name in ['ARTICULATION.json','REGRASP_LEGS.json','SURFACE_PUSH_PROGRESS.json','PUSH_SEGMENTS.json']]
        return dict(key=key,initial_state_id=init,language=previous['language'],category=previous['category'],category_name=previous['category_name'],
            before_success=old_success,after_success=passed,change=change,
            change_label={'retained':'保持通过','improved':'新增通过','regressed':'本轮未通过','failed':'仍失败','additional':'补充初态：'+('通过' if passed else '未完成')}[change],
            result=result,queue=queue,failure=failure(result,queue),audits=audits,before_video=before_video,after_video=str(after.relative_to(out)),
            after_video_sha256=sha(after),before_video_sha256=sha(out/before_video) if before_video else None,
            decoded_frames=len(frames),initial_state_matched=init==0,original_bddl_sha256=sha(dst/'task.bddl'),
            initial_state_file_sha256=sha(Path(env['init_state_file'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(process,jobs))
    external=read(args.external_manifest)
    assert all(sha(Path(path))==expected for path,expected in external.items())
    paired=[x for x in rows if x['initial_state_id']==0]
    groups=collections.defaultdict(list)
    for row in rows:groups[row['initial_state_id'],row['category_name']].append(row)
    summary=dict(episodes=len(rows),source_revision=protocols[0]['revision'],source_digest=protocols[0]['source_digest'],cpu_tests=tests,external_files_unchanged=len(external),
        paired_init0=dict(tasks=len(paired),before_success=sum(x['before_success'] for x in paired),after_success=sum(x['after_success'] for x in paired),
            improved=sum(x['change']=='improved' for x in paired),regressed=sum(x['change']=='regressed' for x in paired)),
        categories=[dict(initial_state_id=init,name=name,tasks=len(items),before_success=sum(x['before_success'] for x in items) if init==0 else None,
            after_success=sum(x['after_success'] for x in items)) for (init,name),items in sorted(groups.items())],
        failure_counts=dict(collections.Counter(x['failure']['code'] for x in rows if not x['after_success'])),
        historical_successes_substituted=False,development_runs_counted=False,scope='Targeted contact-policy iteration, not a full-130 score')
    for name,value in [('PROTOCOLS.json',protocols),('SOURCE_IDENTITY.json',manifests[0]),('BASELINE_RESULTS.json',[before[x['key']] for x in paired]),('SUMMARY.json',summary),('RESULTS.json',rows)]:dump(out/name,value)
    parents={e.resolve().parent for e in args.evaluations}
    if len(parents)==1 and (next(iter(parents))/'MATRIX_PROTOCOL.json').exists():
        parent=next(iter(parents));matrix=read(parent/'MATRIX_COMPLETE.json')
        assert matrix['episodes']==len(rows) and matrix['source_digest']==summary['source_digest'] and matrix['source_unchanged']
        for name in ['MATRIX_PROTOCOL.json','MATRIX_COMPLETE.json','COHORTS.json']:shutil.copy2(parent/name,out/name)
    shutil.copy2(args.source_archive,out/'frozen-source.tar.gz');shutil.copy2(args.tests,out/'REGRESSION.xml')
    dump(out/'FROZEN_CONFIGS.json',{str(read(e/'PROTOCOL.json')['initial_state_id']):read(e/'FROZEN_CONFIGS.json') for e in args.evaluations})
    shutil.copy2(args.external_manifest,out/'EXTERNAL_SOURCE_BEFORE.json')
    for name in ['package_contact_iteration.py','validate_contact_iteration.py','validate_subset.py','run_all.py','package_subset.py','run_matrix.py','run_subset.py']:
        shutil.copy2(Path(__file__).parent/name,out/name)
    report=render(rows,summary,out)
    dump(out/'FILES_SHA256.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps(dict(summary=summary,report=str(report)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
