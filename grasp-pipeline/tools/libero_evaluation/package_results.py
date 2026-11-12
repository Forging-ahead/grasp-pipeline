"""Build a portable report exclusively from this frozen 130-task evaluation."""
import base64, collections, concurrent.futures, datetime, hashlib, html, json, shutil, subprocess, zipfile, sys
from pathlib import Path
import numpy as np
import imageio_ffmpeg

r=Path(sys.argv[1]).resolve()
protocol=json.loads((r/'PROTOCOL.json').read_text())
version=Path(protocol['source'])
run_root=r/'runs/full130'
progress=json.loads((r/'PROGRESS.json').read_text())
assert progress['complete'] and progress['finished']==progress['total']==130
catalog=json.loads((r/'ALL_TASKS.json').read_text())
taxonomy=json.loads((r/'TASK_TAXONOMY.json').read_text())
assert len(catalog)==len(taxonomy)==130
assert {x['key'] for x in catalog}=={x['key'] for x in taxonomy}=={x['key'] for x in progress['results']}
manifest=json.loads((r/'SOURCE_IDENTITY.json').read_text())['files']
assert all(hashlib.sha256((version/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
from run_all import identity,valid_result
assert identity(version)['source_digest']==protocol['source_digest']
assert protocol['source_commit']=='6d8cd7d7e6a34878bb71a0f5e61007e3f81ac405'
out=r/'delivery';out.mkdir(exist_ok=False)
ff=imageio_ffmpeg.get_ffmpeg_exe()
class_order=['single_surface','single_container','single_compartment','single_cabinet','single_extract','single_rack','single_stack',
    'multi_place','stack_transport','drawer','microwave','stove','drawer_place','microwave_place','stove_place','push']
tax={x['key']:x for x in taxonomy}
cat={x['key']:x for x in catalog}

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def successful(d):
    return bool(d.get('pipeline_success') and d.get('official_goal_at_end') and d.get('steps',0)>0)

def diagnose(d,q,parts):
    if successful(d) and q.get('pipeline_success'):return dict(code='completed',label='完整执行且最终条件通过',phase='completed',evidence='pipeline_success、最终官方条件和实际动作均通过。')
    if d.get('steps',0)==0 and d.get('official_goal_at_end'):
        return dict(code='no_action_initial_success',label='初态目标已满足，未执行任务动作',phase='initial',evidence='最终谓词为真，但没有真实动作，未计完整成功。')
    stage=next((x for x in reversed(d.get('stages',[])) if x.get('status')!='already_satisfied'),{})
    f=d.get('failure') or stage.get('failure') or next((x.get('failure') for x in reversed(parts) if x.get('failure')), {})
    code=f.get('code','');phase=f.get('phase') or stage.get('stage',{}).get('skill','unknown')
    message=str(d.get('reason') or stage.get('reason') or '')
    if q.get('watchdog_reached'):label='现实运行预算耗尽';code='wall_budget_exhausted'
    elif 'Fewer than 16 valid visible pixels' in message or 'No finite points' in message:label='当前相机目标可见点云不足'
    elif code=='episode_step_limit':label='仿真动作步数预算耗尽'
    elif code=='initial_contact':label='初始机器人场景触发碰撞检查'
    elif code=='unsupported_task_policy':label='当前策略未实现所需技能'
    elif code=='planning_state_restore_failed':label='规划预览状态恢复检查失败'
    elif code in ('selection_budget_exhausted','checked_candidates_rejected'):label='抓取候选规划或预检未通过'
    elif code=='contact_approach_exhausted':label='机构或接触技能候选未通过预检'
    elif code in ('container_release_search_failed','region_placement_exhausted','surface_release_search_failed','opening_fit_failed'):label='放置位姿候选未通过几何或路径检查'
    elif code=='motion_plan_failed' and phase=='lift':label='闭爪后抬升路径未找到'
    elif code=='motion_plan_failed' and phase in ('transfer','region_reposition'):label='搬运路径未找到'
    elif code=='handle_engagement_failed':label='实际把手接触未建立'
    elif code in ('mechanism_stalled','mechanism_goal_not_reached'):label='机构实际运动停滞或未达到目标'
    elif code=='lift_verification_failed':label='实际夹取后未通过抬升保持检查'
    elif code=='forbidden_contact':label='执行中触发碰撞限制'
    elif code=='waypoint_timeout':label='控制跟踪未在步数内收敛'
    elif code.startswith('joint_'):label='关节限位或轨迹跟踪失败'
    elif f.get('category')=='planning' or code=='path_preflight_rejected':label='阶段路径规划或预检未通过'
    elif f.get('category')=='joint_limit':label='关节限位或状态衔接失败'
    elif f.get('category') in ('control','contact'):label='接触或控制检查未通过'
    elif d.get('status')=='driver_exit_without_result' or code=='unhandled_exception' or d.get('traceback'):label='程序或运行环境异常'
    else:label='释放后或复合目标未全部满足'
    return dict(code=code or d.get('status','unknown'),label=label,phase=phase,details=f.get('details',{}),
        evidence=d.get('reason') or stage.get('reason') or '参见完整结果与阶段记录。',
        root_cause_status='此处是实测退出原因；未经对照实验不将规划拒绝等同于物理不可行或模型能力不足。')

def process(t):
    key=t['key'];src=run_root/key;dst=out/'cases'/key;dst.mkdir(parents=True)
    d=json.loads((src/'RESULT.json').read_text());q=json.loads((src/'QUEUE_RECORD.json').read_text())
    assert d['key']==q['key']==key and q['evaluation_id']==protocol['evaluation_id']
    assert d['initial_state_id']==q['initial_state_id']==0 and q['seed']==0
    assert q['source_verified_before'] and q['source_verified_after']
    assert d.get('driver_sha256')==manifest['apps/libero/run_task.py'],key
    assert valid_result(d,cat[key],{'files':manifest}) and q['result_identity_valid'],key
    assert q['source_digest']==protocol['source_digest'],key
    assert d['original_goals_unchanged'] and d['original_goals']==t['goals'],key
    env=json.loads((src/'environment/ENVIRONMENT.json').read_text())
    assert env['init_state_id']==0 and env['task_name']==t['task_name'] and env['task_suite']==t['suite']
    bddl=Path(env['bddl_file'])
    assert hashlib.sha256(bddl.read_bytes()).hexdigest()==t['bddl_sha256'],key
    for name in ['RESULT.json','QUEUE_RECORD.json','SKILL_PLAN.json','PACKING_PLAN.json','FINAL_PHYSICS_STATE.npz','states.json','trajectory.npz','preview.png']:
        if (src/name).exists():shutil.copy2(src/name,dst/name)
    (dst/'environment').mkdir()
    for name in ['ENVIRONMENT.json','config.json','merged_model.xml']:
        if (src/'environment'/name).exists():shutil.copy2(src/'environment'/name,dst/'environment'/name)
    shutil.copy2(bddl,dst/'task.bddl')
    shutil.copy2(r/'queue_logs'/(key+'.log'),dst/'driver.log')
    shutil.copy2(r/'queue_logs'/(key+'.pid.json'),dst/'process.json')
    parts=[]
    for folder in sorted(src.glob('stage_*')):
        partfile=folder/'RESULT.json'
        if partfile.exists():parts.append(json.loads(partfile.read_text()))
        for name in ['RESULT.json','ARTICULATION.json','CONTACT_STATES.json','PLANS.json','PUSH_SEGMENTS.json','EXECUTION_CHECKS.json','config.json',
                     'input_points.npz','grasp_candidates.npz','grasp_candidates.npz.json','initial_scene.json','perception.json','selected.json']:
            if (folder/name).exists():
                target=dst/folder.name/name;target.parent.mkdir(exist_ok=True);shutil.copy2(folder/name,target)
    states=json.loads((src/'states.json').read_text());n=int(d['steps'])
    assert len(states)==n+1 and [x['step_count'] for x in states]==list(range(n+1)),key
    assert all(abs(x['sim_time']-states[0]['sim_time']-i/20)<1e-7 for i,x in enumerate(states)),key
    with np.load(src/'trajectory.npz',allow_pickle=False) as z:
        phases=z['phases'].tolist()
        assert len(phases)==n and z['actions'].shape==(n,7) and z['world_T_eef'].shape==(n+1,4,4),key
        ids=sorted({0,n-1}|{i for i in range(1,n) if phases[i]!=phases[i-1]}|{i-1 for i in range(1,n) if phases[i]!=phases[i-1]}) if n else []
        points=[dict(action_index=i,phase=phases[i],world_T_eef=z['world_T_eef'][i+1].tolist(),
                     q_arm=z['q_arm'][i+1].tolist(),gripper_command=float(z['actions'][i,-1]),
                     label_source='handwritten code policy; not a VLM decision',
                     pose_source='measured simulator state after the indexed action') for i in ids]
    dump(dst/'KEYPOINTS.json',points)
    actual_phases={'approach','descend','engage','articulate','articulate_feedback','push','close_gripper','close_on_handle'}
    physical=any(str(p).split('/')[-1] in actual_phases for p in phases)
    phase_counts=dict(collections.Counter(phases))
    order_issue=False;order_note=''
    if t['category']=='stack_transport':
        executed=[s['stage'] for s in d.get('stages',[]) if s.get('stage',{}).get('skill')=='pick_place' and s.get('status')!='already_satisfied']
        order_issue=bool(executed and executed[0]['predicate'][0].lower()=='in')
        order_note='策略先将底碗放入托盘，再处理上碗；不符合语言要求的先堆叠再整体搬运。' if order_issue else '需要逐段检查是否先堆叠、再保持整堆搬运；最终谓词不能证明此顺序。'
    tracking=[dict(stage=f.parent.name,**e) for f in src.glob('stage_*/RESULT.json')
              for e in json.loads(f.read_text()).get('execution_checks',[]) if e.get('event','').startswith('near_goal_tracking_extension')]
    raw=src/'rollout.mp4';video=dst/'video.mp4'
    assert raw.is_file(), 'Missing original rollout: '+key
    subprocess.run([ff,'-v','error','-y','-i',str(raw),'-vf','tpad=stop_mode=clone:stop_duration=2',
        '-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    decoded=subprocess.run([ff,'-v','error','-i',str(video),'-f','framemd5','-'],capture_output=True,text=True,check=True).stdout
    frames=sum(bool(x) and not x.startswith('#') for x in decoded.splitlines())
    assert frames==n+41,(key,frames,n)
    row=dict(key=key,category=t['category'],category_name=t['category_name'],language=t['language'],suite=t['suite'],
        result=d,queue_record=q,full_success=bool(q['pipeline_success']),physical_skill_attempt=physical,
        failure=diagnose(d,q,parts),last_phase=phases[-1] if phases else 'initial',phase_action_counts=phase_counts,
        order_issue=order_issue,order_note=order_note,tracking_extension_events=tracking,
        initial_state_file=env['init_state_file'],initial_state_file_sha256=hashlib.sha256(Path(env['init_state_file']).read_bytes()).hexdigest(),
        bddl_sha256=t['bddl_sha256'],source_verified=True,states_continuous=True,initial_state_id=0,
        video_path=str(video.relative_to(out)),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        frames=frames,video_fully_decoded=True,sim_action_seconds=n/20,wall_seconds=q['wall_seconds'])
    dump(dst/'ANALYSIS.json',{k:v for k,v in row.items() if k not in ('result','queue_record')})
    return row

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    rows=list(pool.map(process,sorted(taxonomy,key=lambda t:(class_order.index(t['category']),t['key']))))
assert len(rows)==len({x['key'] for x in rows})==130
summary=dict(tasks=130,trials=130,source_version='organized-libero-6d8cd7d',source_commit=protocol['source_commit'],source_digest=protocol['source_digest'],init_state_id=0,seed=0,
    full_success=sum(x['full_success'] for x in rows),official_goal_at_end=sum(bool(x['result'].get('official_goal_at_end')) for x in rows),
    success_with_order_issue=sum(x['full_success'] and x['order_issue'] for x in rows),
    failed=sum(not x['full_success'] for x in rows),videos=len(rows),
    failure_only_preflight=sum(not x['full_success'] and not x['physical_skill_attempt'] for x in rows),
    failure_after_physical_attempt=sum(not x['full_success'] and x['physical_skill_attempt'] for x in rows),
    wall_elapsed_seconds=progress['elapsed_seconds'],sum_task_wall_seconds=sum(x['wall_seconds'] for x in rows),
    wall_task_min_seconds=min(x['wall_seconds'] for x in rows),wall_task_max_seconds=max(x['wall_seconds'] for x in rows),
    no_historical_outcomes=True,one_frozen_source=True)
summary['success_rate']=summary['full_success']/130
summary['success_excluding_known_order_violations']=summary['full_success']-summary['success_with_order_issue']
groups=[]
for category in class_order:
    selected=[x for x in rows if x['category']==category]
    groups.append(dict(category=category,name=selected[0]['category_name'],tasks=len(selected),
        full_success=sum(x['full_success'] for x in selected),failed=sum(not x['full_success'] for x in selected),
        success_rate=sum(x['full_success'] for x in selected)/len(selected),
        success_with_order_issue=sum(x['full_success'] and x['order_issue'] for x in selected),
        failure_reasons=dict(collections.Counter(x['failure']['label'] for x in selected if not x['full_success'])),
        failed_tasks=[x['key'] for x in selected if not x['full_success']] ))
failures=dict(collections.Counter(x['failure']['label'] for x in rows if not x['full_success']))
dump(out/'RESULTS.json',rows);dump(out/'SUMMARY.json',summary);dump(out/'CATEGORY_SUMMARY.json',groups);dump(out/'FAILURE_SUMMARY.json',failures)
(out/'README.txt').write_text('''LIBERO 130任务：libero分支6d8cd7d同版本复测（2026-11-13）

直接打开“LIBERO130_libero分支复测_20261113_含全部视频.html”，建议使用Chrome。
全部130段视频已嵌入HTML；只转发该HTML也可以离线观看，不依赖实验服务器。
cases/保存逐任务单独视频、结果、阶段记录、动作、状态和关键点。

口径：每任务初态0、种子0，只运行一次；没有合并旧版本的成功记录。
完整成功要求有实际动作、pipeline_success及原始官方最终条件均为真。
堆叠再整体搬运的已知顺序问题单独标注，不能仅以最终谓词认证语言过程。
轨迹通过最终条件也仍需视频、动作顺序及持物稳定性复核，才能筛选为SFT样本。

视频为20fps、两相机视角，结尾增加2秒末帧停留。
计算等待时间不等于仿真动作时间；无抓取动作的短视频不应解释为实际抓取失败。
KEYPOINTS.json记录阶段边界的实际测量末端位姿和夹爪命令，标签来自手写code policy。
这些不是VLM作出的工具选择；本数据包也不是完整无损RGB-D观测序列导出。

research-source.zip提供libero分支冻结源码与运行/统计工具供审计。
以SOURCE_IDENTITY.json及PROTOCOL.json记录的提交与指纹为依据。
外部依赖、模型权重及LIBERO场景资产没有打包；运行配置含实验服务器绝对路径。
SCHEDULING_HISTORY.json记录本轮实际并发变化。
并发影响墙钟耗时和预算内搜索深度，时间预算失败不直接证明模型能力不足。
''')
for name in ['PROTOCOL.json','PROGRESS.json','ALL_TASKS.json','TASK_TAXONOMY.json','SOURCE_IDENTITY.json','FROZEN_CONFIGS.json','GPU_SLOTS.json','SCHEDULING_HISTORY.json','SWEEP_COMPLETE.json','FIRST_BATCH_AUDIT.json']:
    if (r/name).exists():shutil.copy2(r/name,out/name)
external=json.loads((r.parent/'EXTERNAL_SOURCE_BEFORE.json').read_text())
for name in ['EXTERNAL_SOURCE_BEFORE.json','DEPLOYMENT.json','REGRESSION.xml','LAUNCH.json','PRELAUNCH_VALIDATION.json']:
    if (r.parent/name).exists():shutil.copy2(r.parent/name,out/name);external_changes=[]
for name,sha in external.items():
    h=hashlib.sha256()
    with Path(name).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=sha:external_changes.append(name)
assert not external_changes,external_changes
dump(out/'SOURCE_AUDIT.json',dict(frozen_files=len(manifest),all_actor_hashes_match=True,external_files=len(external),external_changes=[],
    bddl_goals_match_all130=True,initial_state_id0_all130=True,videos_fully_decoded=130,continuous_episodes=130,
    historical_outcomes_imported=False,original_server_partial_results_excluded=True))
with zipfile.ZipFile(out/'research-source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['run_all.py','package_results.py','render_report.py','watch.py','validate_delivery.py','finalize.py']:
        p=Path(__file__).parent/name
        if p.is_file():z.write(p,'harness/'+name)
    for name in ['ALL_TASKS.json','TASK_TAXONOMY.json','PROTOCOL.json','SOURCE_IDENTITY.json','FROZEN_CONFIGS.json']:
        z.write(r/name,name)
    for p in sorted(version.parent.rglob('*')):
        if p.is_file() and not any(x in p.relative_to(version.parent).parts for x in ('runs','__pycache__','.pytest_cache')) and p.suffix!='.pyc':
            z.write(p,'source/'+str(p.relative_to(version.parent)))
subprocess.run([sys.executable,str(Path(__file__).parent/'render_report.py'),str(r)],check=True)
dump(out/'PACKAGE_QA.json',dict(passed=True,summary=summary,source_audit=json.loads((out/'SOURCE_AUDIT.json').read_text())))
print(json.dumps(summary,ensure_ascii=False),flush=True)
