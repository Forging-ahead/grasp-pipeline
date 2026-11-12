#!/usr/bin/env python3
"""Audit a frozen subset and build an offline before/after video report."""
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
import zipfile

from run_all import dump, identity, read, valid_result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


LABELS = {
    'grasp_environment_collision':'所有候选的夹爪终点被环境或目标掌部接触挡住',
    'selection_budget_exhausted':'抓取候选搜索预算耗尽',
    'checked_candidates_rejected':'检查过的抓取候选均未通过规划或预检',
    'contact_approach_exhausted':'把手或推移接触候选未通过预检',
    'motion_plan_failed':'阶段路径规划未通过',
    'forbidden_contact':'实际执行中碰到不允许接触的部件',
    'waypoint_timeout':'控制跟踪未在预算内到达目标',
    'lift_verification_failed':'实际夹取后物体未通过抬升保持检查',
    'handle_engagement_failed':'把手未形成持续双指夹持',
    'mechanism_stalled':'接触后机构停止运动',
    'mechanism_goal_not_reached':'机构未达到目标关节状态',
    'contact_ik_failed':'机构运动的局部IK未通过',
    'mechanism_feedback_ik_failed':'机构反馈修正的局部IK未通过',
    'contact_retreat_failed':'操作后退手路径未通过',
}


def failure(result, queue):
    if queue['pipeline_success']:
        return dict(code='completed',label='完整流程与官方最终条件通过',evidence='')
    if queue['watchdog_reached']:
        return dict(code='wall_budget_exhausted',label='实际运行时间预算耗尽',evidence=result.get('reason',''))
    item = result.get('failure')
    if not item:
        item = next((x.get('failure') for x in reversed(result.get('stages',[])) if x.get('failure')),None)
    item = dict(item or dict(code=result.get('status','unknown')))
    code = item['code']
    return dict(item,label=LABELS.get(code,'最终目标或后续阶段未完整完成'),evidence=result.get('reason',''))


def render(rows, summary, protocol, out):
    h = html.escape
    cards = []
    for index,row in enumerate(rows,1):
        videos=[]
        for name,title in [('before_video','修改前'),('after_video','本轮冻结新版')]:
            encoded=base64.b64encode((out/row[name]).read_bytes()).decode()
            videos.append(f'<div><p><b>{title}</b></p><video controls preload="none" playsinline src="data:video/mp4;base64,{encoded}"></video></div>')
        screen = row['grasp_environment_screens']
        screened=sum(x['input_candidates'] for x in screen)
        rejected=sum(x['rejected'] for x in screen)
        timing=sum(x['wall_s'] for x in screen)
        screen_note=(f'GraspGen候选环境筛选：{screened}个输入，剔除{rejected}个，累计{timing*1000:.1f}毫秒。'
            if screen else '本任务未产生刚体抓取环境筛选记录；机构操作使用几何候选。')
        result_view=json.dumps(dict(result=row['result'],failure=row['failure'],
            grasp_environment_screens=screen,handle_screens=row['handle_environment_screens']),ensure_ascii=False,indent=2)
        cards.append(f'''<details class="case" data-status="{h(row['change'])}" data-search="{h((row['key']+' '+row['language']+' '+row['failure']['label']).lower(),quote=True)}">
<summary><span class="badge {h(row['change'])}">{h(row['change_label'])}</span><b>{index:02d} · {h(row['language'])}</b></summary>
<div class="body"><p class="meta">{h(row['category_name'])} · {h(row['key'])}</p><div class="videos">{''.join(videos)}</div>
<p><b>本轮：</b>{h(row['failure']['label'])}；{row['result']['steps']}步，实际耗时{row['queue']['wall_seconds']/60:.2f}分钟。</p>
<p>{h(screen_note)}</p><p class="meta">{h(str(row['failure'].get('evidence',''))[:1800])}</p>
<details><summary>阶段、候选筛选与原始结果</summary><pre>{h(result_view)}</pre></details></div></details>''')
    table=[]
    groups=collections.defaultdict(list)
    for row in rows:groups[row['category']].append(row)
    for items in groups.values():
        table.append(f'<tr><td>{h(items[0]["category_name"])}</td><td>{len(items)}</td><td>{sum(x["before_success"] for x in items)}</td><td>{sum(x["after_success"] for x in items)}</td></tr>')
    errors=''.join(f'<li>{h(name)}：{count}项</li>' for name,count in summary['failure_counts'].items())
    scope=('包含全部26项机构及相关操作、候选筛选失败/超时任务及6项原成功抓放对照。'
        if len(rows)==39 else '具体任务以本页逐项清单为准。')
    comparison_note=('<p class="meta">补充核对：三个新增成功抓放样例的点云输入一致，但GraspGen重复推理的候选并非逐值一致。'
        '本页是工程前后对照，不把全部新增成功归因于单个筛选模块；尚未开展固定候选的组件消融。完整核对数据随结果包保留。</p>'
        if (out/'CANDIDATE_PAIR_AUDIT.json').exists() else '')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO环境筛选与机构位姿 · 定向复验</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#193047;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65}main{max-width:1280px;margin:auto;padding:32px 22px 70px}h1{font-size:31px;line-height:1.3}h2{margin-top:34px}.meta{color:#61758a;font-size:14px}.notice{padding:16px 20px;border-left:4px solid #d29225;background:#fff7e6}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,.case{background:white;border:1px solid #dce4ec;border-radius:12px}.metric{padding:18px}.metric b{display:block;font-size:29px;color:#17657c}.metric span{font-size:14px;color:#61758a}.case{margin:14px 0;overflow:hidden}.case>summary{display:flex;gap:14px;align-items:center;cursor:pointer;padding:18px}.body{padding:0 18px 22px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#101318;display:block;border-radius:7px}.badge{font-size:12px;white-space:nowrap;padding:4px 9px;border-radius:5px;background:#e6edf3}.improved,.retained{color:#16624b;background:#e4f5ed}.regressed{color:#a62f36;background:#fce9eb}.failed{color:#8d640f;background:#fff3d3}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:11px;border-bottom:1px solid #dce4ec}th{background:#e8f0f6}pre{white-space:pre-wrap;word-break:break-word;background:#f0f4f7;padding:14px;font-size:12px;max-height:650px;overflow:auto}.filters{display:flex;gap:12px;margin:18px 0}input,select{padding:10px;border:1px solid #c7d3dd;border-radius:6px;font:inherit}input{flex:1;min-width:80px}.hidden{display:none}@media(max-width:750px){.metrics{grid-template-columns:1fr 1fr}.videos{grid-template-columns:1fr}.case>summary{align-items:flex-start}h1{font-size:25px}.filters{flex-wrap:wrap}}
</style></head><body><main>'''
    page+=f'''<p class="meta">LIBERO · 初态0 · 单次运行 · 冻结源码 {h(protocol['revision'][:7])}</p><h1>环境碰撞筛选与机构位姿修复：{len(rows)}项定向复验</h1>
<p>优先处理抓取候选与环境不匹配，以及抽屉、门等几何操作候选不足的问题。前后视频均已嵌入本文件，可单文件转发、离线观看。</p>
<div class="metrics"><div class="metric"><b>{summary['before_success']}/{len(rows)}</b><span>同一批任务修改前通过</span></div><div class="metric"><b>{summary['after_success']}/{len(rows)}</b><span>本轮冻结新版通过</span></div><div class="metric"><b>{summary['improved']}</b><span>原失败、本轮通过</span></div><div class="metric"><b>{summary['regressed']}</b><span>原通过、本轮未完成</span></div></div>
<p class="notice">这是按失败类型选出的{len(rows)}项复验，{scope}修改前来自上一轮完整130项的对应记录；修改后只使用本轮同一冻结版本的新输出，没有合并开发阶段的成功。该比例不能作为新版全130项成功率。</p>
<h2>本轮改了什么</h2><p>GraspGen仍接收目标物体点云，生成器、判别器权重及评分顺序不变。先用实际张开的夹爪几何筛掉与桌面、邻物、容器或目标掌部接触相撞的终点，再调用原规划器。未加入GraspGen候选去重/多样性重排，也未增加闭爪后抬升预检。</p>
<p>机构操作保留原候选在前，补齐把手倾角和中段位置，以真实几何过滤额外候选；修正圆柱把手的轴向自由度、门角度预检推算，以及后续预检失败时真正调用全局接近分支。仍使用仿真真值、手写code policy、cuRobo及MuJoCo局部IK/反馈控制。</p>
{comparison_note}
<h2>各类任务对照</h2><table><thead><tr><th>本次覆盖类型</th><th>任务数</th><th>修改前通过</th><th>本轮通过</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<h2>本轮未完成的退出原因</h2><ul>{errors}</ul><p class="meta">这些是实际退出分类，规划拒绝不等于物理空间无解；端点通过也不保证全臂可达、接近可行或实际抓取保持成功。仍失败的任务照常保留视频与记录。</p>
<h2>逐任务视频</h2><div class="filters"><select id="status"><option value="">全部任务</option><option value="improved">原失败、本轮通过</option><option value="regressed">原通过、本轮未完成</option><option value="failed">两轮均未完成</option><option value="retained">两轮均通过</option></select><input id="search" type="search" placeholder="搜索任务、物体或退出原因"></div><p class="meta" id="count"></p>{''.join(cards)}
<h2>验证与数据</h2><p>{len(rows)}项源码和原始目标匹配；初态资产、起始状态及原数字参数核对；连续动作/状态与视频完整解码检查通过。146项CPU回归通过。候选筛选耗时单独记录，任务实际耗时包含模型启动、规划、仿真和渲染；并发不同不用于声称组件加速倍数。</p>
<details><summary>运行协议与汇总</summary><pre>{h(json.dumps(dict(protocol=protocol,summary=summary),ensure_ascii=False,indent=2))}</pre></details><p class="meta">每段视频结尾增加2秒静帧；没有拼接未执行的动作。轨迹仍需按指令过程、稳定性及学生观测条件审核，不能仅凭本页通过数直接认定为SFT成品数据。</p></main>
'''
    page+='''<script>const cases=[...document.querySelectorAll('.case')],status=document.querySelector('#status'),search=document.querySelector('#search');function filter(){let count=0;for(const c of cases){const show=(!status.value||c.dataset.status===status.value)&&c.dataset.search.includes(search.value.trim().toLowerCase());c.classList.toggle('hidden',!show);if(show)count++;else for(const v of c.querySelectorAll('video'))v.pause();}document.querySelector('#count').textContent=`显示${count}/${cases.length}项`;}status.addEventListener('input',filter);search.addEventListener('input',filter);for(const video of document.querySelectorAll('video'))video.addEventListener('play',()=>{for(const other of document.querySelectorAll('video'))if(other!==video)other.pause();});filter();</script></body></html>'''
    path=out/f'LIBERO_环境筛选与机构修复_{len(rows)}项对照视频.html'
    path.write_text(page,encoding='utf-8')
    return path


def main():
    import numpy as np
    import imageio_ffmpeg
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--source-archive',type=Path,required=True)
    args=parser.parse_args()
    r,baseline=args.evaluation.resolve(),args.baseline.resolve()
    protocol=read(r/'PROTOCOL.json');manifest=read(r/'SOURCE_IDENTITY.json')
    source=Path(protocol['source']);assert identity(source)==manifest
    catalog=read(r/'ALL_TASKS.json');queue=read(r/'SUBSET_COMPLETE.json')
    assert len(catalog)==queue['total'] and queue['source_unchanged']
    before={row['key']:row for row in read(baseline/'RESULTS.json')}
    out=r/'delivery';out.mkdir(exist_ok=False)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    configs=read(r/'FROZEN_CONFIGS.json')
    def process(row):
        key=row['key'];previous=before[key]
        src=r/'runs/subset'/key;dst=out/'cases'/key;dst.mkdir(parents=True)
        result,record=read(src/'RESULT.json'),read(src/'QUEUE_RECORD.json')
        assert valid_result(result,row,manifest) and record['result_identity_valid'] and record['source_unchanged'],key
        assert record['source_digest']==protocol['source_digest']==manifest['source_digest'],key
        assert result['original_goals']==previous['result']['original_goals'] and result['original_goals_unchanged'],key
        env=read(src/'environment/ENVIRONMENT.json');cfg=read(src/'environment/config.json')
        assert env['init_state_id']==0 and env['task_name']==row['task_name'] and env['task_suite']==row['suite'],key
        assert cfg['seed']==0 and cfg['max_steps']==5000 and cfg['settle_steps']==10,key
        old_cfg=read(baseline/'cases'/key/'environment/config.json')
        deployment={'simulation_gpu','model_gpu','planner_gpu'}
        assert {k:v for k,v in cfg.items() if k not in deployment}=={k:v for k,v in old_cfg.items() if k not in deployment},key
        expected=dict(configs[key])
        for name in deployment:expected[name]=record['gpu']
        assert all(cfg.get(k)==v for k,v in expected.items()),key
        assert sha(Path(env['bddl_file']))==previous['bddl_sha256'],key
        assert sha(Path(env['init_state_file']))==previous['initial_state_file_sha256'],key
        states=read(src/'states.json');old_states=read(baseline/'cases'/key/'states.json')
        for name in ['q_arm','q_fingers','qvel_arm','qvel_fingers','world_T_eef','world_T_base']:
            np.testing.assert_allclose(states[0][name],old_states[0][name],atol=1e-10,rtol=0,err_msg=key+name)
        assert states[0]['object_poses'].keys()==old_states[0]['object_poses'].keys(),key
        for name in states[0]['object_poses']:
            np.testing.assert_allclose(states[0]['object_poses'][name],old_states[0]['object_poses'][name],atol=1e-10,rtol=0,err_msg=key+name)
        dump(dst/'INITIAL_STATE_BEFORE.json',old_states[0])
        dump(dst/'INITIAL_CONFIG_BEFORE.json',old_cfg)
        with np.load(src/'trajectory.npz',allow_pickle=False) as trace:
            n=len(trace['actions']);assert n==result['steps'] and len(states)==n+1,key
            for name,shape in [('q_arm',(n+1,7)),('q_fingers',(n+1,2)),('world_T_eef',(n+1,4,4)),('actions',(n,7))]:
                assert trace[name].shape==shape and np.isfinite(trace[name]).all(),(key,name)
            assert trace['phases'].shape==(n,),key
        if len(states)>1:
            np.testing.assert_allclose(np.diff([x['sim_time'] for x in states]),.05,atol=1e-9,rtol=0,err_msg=key)
        for name in ['RESULT.json','QUEUE_RECORD.json','states.json','trajectory.npz','FINAL_PHYSICS_STATE.npz','SKILL_PLAN.json','PACKING_PLAN.json','preview.png']:
            if (src/name).exists():shutil.copy2(src/name,dst/name)
        (dst/'environment').mkdir()
        for name in ['ENVIRONMENT.json','config.json','merged_model.xml']:shutil.copy2(src/'environment'/name,dst/'environment'/name)
        shutil.copy2(env['bddl_file'],dst/'task.bddl')
        shutil.copy2(r/'logs'/(key+'.log'),dst/'driver.log')
        screens,handles=[],[]
        keep_names={'RESULT.json','config.json','perception.json','GRASP_ENVIRONMENT_SCREEN.json','environment_candidates.json',
            'grasp_candidates.npz','grasp_candidates.npz.json','input_points.npz','selected.json','selection_rejections.json',
            'HANDLE_ENVIRONMENT_SCREEN.json','HANDLE_CANDIDATES.json','HANDLE_CANDIDATES_ALL.json','ARTICULATION.json',
            'CONTACT_STATES.json','PLANS.json','EXECUTION_CHECKS.json','PUSH_SEGMENTS.json'}
        for stage in sorted(src.glob('stage_*')):
            target=dst/stage.name;target.mkdir()
            for path in stage.iterdir():
                if path.is_file() and path.name in keep_names:shutil.copy2(path,target/path.name)
            if (stage/'GRASP_ENVIRONMENT_SCREEN.json').exists():screens.append(read(stage/'GRASP_ENVIRONMENT_SCREEN.json'))
            if (stage/'HANDLE_ENVIRONMENT_SCREEN.json').exists():handles.append(read(stage/'HANDLE_ENVIRONMENT_SCREEN.json'))
        video=dst/'after.mp4'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','2','-i',str(src/'rollout.mp4'),
            '-an','-vf','tpad=stop_mode=clone:stop_duration=2','-r','20','-c:v','libx264','-preset','fast','-crf','22',
            '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(video)],check=True,capture_output=True)
        decode=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','2','-i',str(video),'-an','-f','framehash','-'],
            check=True,capture_output=True,text=True)
        frames=[line for line in decode.stdout.splitlines() if line and not line.startswith('#')]
        assert len(frames)==n+41,(key,len(frames),n)
        old_video=baseline/previous['video_path'];assert sha(old_video)==previous['video_sha256'],key
        shutil.copy2(old_video,dst/'before.mp4')
        passed=bool(valid_result(result,row,manifest) and not record['watchdog_reached'] and
            result.get('pipeline_success') and result.get('official_goal_at_end') and n>0)
        assert passed==record['pipeline_success'],key
        old=bool(previous['full_success'])
        change='retained' if passed and old else 'improved' if passed else 'regressed' if old else 'failed'
        return dict(key=key,language=previous['language'],category=previous['category'],category_name=previous['category_name'],
            before_success=old,after_success=passed,change=change,
            change_label={'retained':'保持通过','improved':'新增通过','regressed':'本轮未通过','failed':'仍未完成'}[change],
            result=result,queue=record,failure=failure(result,record),grasp_environment_screens=screens,
            handle_environment_screens=handles,before_video=str((dst/'before.mp4').relative_to(out)),
            after_video=str(video.relative_to(out)),before_video_sha256=sha(dst/'before.mp4'),after_video_sha256=sha(video),
            decoded_frames=len(frames),continuous_states=True,identical_initial_state=True,identical_numeric_config=True,
            original_bddl_sha256=sha(dst/'task.bddl'),initial_state_file_sha256=sha(Path(env['init_state_file'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(process,catalog))
    assert identity(source)==manifest
    external=read(r.parent/'EXTERNAL_SOURCE_BEFORE.json')
    assert all(sha(Path(path))==expected for path,expected in external.items())
    summary=dict(tasks=len(rows),source_commit=protocol['revision'],source_digest=manifest['source_digest'],
        before_success=sum(x['before_success'] for x in rows),after_success=sum(x['after_success'] for x in rows),
        improved=sum(x['change']=='improved' for x in rows),regressed=sum(x['change']=='regressed' for x in rows),
        retained=sum(x['change']=='retained' for x in rows),failed=sum(not x['after_success'] for x in rows),
        failure_counts=dict(collections.Counter(x['failure']['label'] for x in rows if not x['after_success'])),
        trial_per_task=1,init_state_id=0,cpu_tests=146,external_files_unchanged=len(external),
        source_unchanged=True,development_runs_counted=False,scope='Targeted subset only; not full-130 success rate')
    for name in ['PROTOCOL.json','SOURCE_IDENTITY.json','FROZEN_CONFIGS.json','ALL_TASKS.json','SUBSET_COMPLETE.json']:
        shutil.copy2(r/name,out/name)
    for name in ['REGRESSION_FROZEN.xml','FROZEN_SOURCE.json','EXTERNAL_PRECHECK.json','FINAL_LAUNCH.json']:
        shutil.copy2(r.parent/name,out/name)
    shutil.copy2(args.source_archive,out/'frozen-source.tar.gz')
    dump(out/'SOURCE_AUDIT.json',dict(passed=True,actor_files=len(manifest['files']),external_files=len(external),source_unchanged=True))
    dump(out/'BASELINE_RESULTS.json',[before[row['key']] for row in catalog])
    shutil.copy2(baseline/'SUMMARY.json',out/'BASELINE_FULL130_SUMMARY.json')
    if (r.parent/'CANDIDATE_PAIR_AUDIT.json').exists():
        shutil.copy2(r.parent/'CANDIDATE_PAIR_AUDIT.json',out/'CANDIDATE_PAIR_AUDIT.json')
    (out/'report-tools').mkdir()
    for name in ['package_subset.py','validate_subset.py','finalize_subset.py']:
        path=Path(__file__).parent/name
        if path.exists():shutil.copy2(path,out/'report-tools'/name)
    dump(out/'SUMMARY.json',summary);dump(out/'RESULTS.json',rows)
    report=render(rows,summary,protocol,out)
    dump(out/'FILES_SHA256.json',{str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    print(json.dumps(dict(summary=summary,report=str(report)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
