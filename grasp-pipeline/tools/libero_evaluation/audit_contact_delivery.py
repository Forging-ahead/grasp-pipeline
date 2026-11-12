#!/usr/bin/env python3
"""Post-run model replay and contact-quality review; never edits task outcomes."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    import numpy as np
    import mujoco
    import imageio_ffmpeg
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('delivery',type=Path)
    p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    r=a.delivery.resolve();rows=read(r/'RESULTS.json');summary=read(r/'SUMMARY.json')
    assert not (r/'CONTACT_QUALITY_AUDIT.json').exists(),'Already post-audited'
    models=[];contacts=[];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for row in rows:
        key,init=row['key'],row['initial_state_id'];case=r/'cases'/('init_'+str(init))/key
        states=read(case/'states.json');last=states[-1]
        model=mujoco.MjModel.from_binary_path(str(case/'environment/reset_model.mjb'))
        data=mujoco.MjData(model)
        with np.load(case/'FINAL_PHYSICS_STATE.npz',allow_pickle=False) as saved:
            for name in saved.files:
                if name=='time':data.time=float(saved[name])
                elif hasattr(data,name):getattr(data,name)[:]=saved[name]
        with np.load(case/'environment/MODEL_PARAMETERS.npz',allow_pickle=False) as params:
            for name in params.files:np.testing.assert_array_equal(getattr(model,name),params[name])
        arm=[int(model.jnt_qposadr[model.joint(name).id]) for name in last['joint_names']]
        fingers=[int(model.jnt_qposadr[model.joint(name).id]) for name in last['finger_joint_names']]
        np.testing.assert_array_equal(data.qpos[arm],last['q_arm'])
        np.testing.assert_array_equal(data.qpos[fingers],last['q_fingers'])
        mujoco.mj_forward(model,data)
        site=model.site(last['eef_site']).id
        eef_error=float(np.linalg.norm(data.site_xpos[site]-np.asarray(last['world_T_eef'])[:3,3]))
        body_names={model.body(i).name:i for i in range(model.nbody)}
        fixed_errors=[];unresolved=[]
        for name,pose in last['object_poses'].items():
            body=body_names.get(name+'_main',body_names.get(name))
            if body is None:
                candidates=[i for n,i in body_names.items() if n.startswith(name+'_') and not model.body(int(model.body_parentid[i])).name.startswith(name+'_')]
                if len(candidates)==1:body=candidates[0]
            if body is None:unresolved.append(name);continue
            ancestor=body;dynamic=False
            while ancestor:
                if model.body_jntnum[ancestor] or model.body_mocapid[ancestor]>=0:dynamic=True;break
                ancestor=int(model.body_parentid[ancestor])
            if not dynamic:
                position=float(np.max(np.abs(data.xpos[body]-np.asarray(pose)[:3,3])))
                rotation=float(np.max(np.abs(data.xmat[body].reshape(3,3)-np.asarray(pose)[:3,:3])))
                assert max(position,rotation)<1e-9,(key,init,name,position,rotation)
                fixed_errors.append(max(position,rotation))
        models.append(dict(key=key,initial_state_id=init,model_parameters_exact=True,robot_joint_state_exact=True,
            fixed_object_pose_error_max=max(fixed_errors,default=0.),unresolved_object_names=unresolved,
            recorded_cache_vs_fresh_fk_eef_error_m=eef_error))
        if not row['after_success']:continue
        samples=[dict(x,trace=str(path.relative_to(case))) for path in case.rglob('CONTACT_STATES.json') for x in read(path)]
        peak=max(samples,key=lambda x:x.get('nonfinger_mechanism_force_N',0.),default={})
        force=float(peak.get('nonfinger_mechanism_force_N',0.));flag=force>.1
        entry=dict(key=key,initial_state_id=init,language=row['language'],task_completed=True,
            contact_trace_samples=len(samples),observed_nonfinger_mechanism_force_peak_N=force,
            samples_above_point1N=sum(x.get('nonfinger_mechanism_force_N',0.)>.1 for x in samples),
            requires_manual_contact_review=flag,peak_step=peak.get('step'),peak_phase=peak.get('phase'),
            trace=peak.get('trace'),quality_certified=False)
        if flag:
            frame=r/'quality_frames'/('init_'+str(init)+'__'+key+'.png');frame.parent.mkdir(exist_ok=True)
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(r/row['after_video']),'-ss',str(peak['step']/20),'-frames:v','1',str(frame)],check=True,capture_output=True)
            entry.update(frame=str(frame.relative_to(r)),frame_sha256=sha(frame))
        contacts.append(entry)
    record=dict(actor_source_commit=summary['source_revision'],postprocessor_sha256=sha(__file__),
        task_outcomes_unchanged=True,sampling_hz=20,review_threshold_N=.1,geometry_monitor_tolerance_m=.001,
        scope='Recorded robot non-finger normal forces against the target mechanism only; not a substep/all-scene contact or SFT-quality certificate',
        completed_episodes=len(contacts),contact_review_required=sum(x['requires_manual_contact_review'] for x in contacts),episodes=contacts)
    write(r/'CONTACT_QUALITY_AUDIT.json',record)
    write(r/'MODEL_REPLAY_QA.json',dict(passed=True,episodes=len(models),mujoco_version=mujoco.__version__,
        limitation='Binary model plus final integration inputs; Python controller internal state is not restored. Fresh FK can differ from last cached observation within an integration substep.',results=models))
    shutil.copy2(__file__,r/'audit_contact_delivery.py')
    recovery_path=r.parent/'matrix/INFRASTRUCTURE_RECOVERY.json'
    recovery_notice=''
    if recovery_path.exists():
        recovery=read(recovery_path)
        assert recovery['complete'] and recovery['frozen_actor_unchanged'] and recovery['original_files_retained']
        assert recovery['actor_revision']==summary['source_revision'] and recovery['source_digest']==summary['source_digest']
        retained={(x['key'],x['initial_state_id']) for x in recovery['retained']}
        interrupted={(x['key'],x['initial_state_id']) for x in recovery['infrastructure_reruns']}
        all_episodes={(x['key'],x['initial_state_id']) for x in rows}
        assert not (retained&interrupted) and retained|interrupted<=all_episodes
        assert len(retained)+len(interrupted)+recovery['previously_unstarted']==len(rows)
        for row in rows:
            episode=(row['key'],row['initial_state_id'])
            if episode not in retained:continue
            original=next(x for x in recovery['retained'] if (x['key'],x['initial_state_id'])==episode)
            case=Path(original['directory'])/'runs/subset'/row['key']
            assert read(case/'RESULT.json')==row['result'] and read(case/'QUEUE_RECORD.json')==row['queue']
            delivered=r/'cases'/('init_'+str(row['initial_state_id']))/row['key']
            for name in ['RESULT.json','QUEUE_RECORD.json','states.json','trajectory.npz','FINAL_PHYSICS_STATE.npz']:
                assert sha(case/name)==sha(delivered/name)
        runner=r.parent/'matrix/recover_matrix.py';assert sha(runner)==recovery['recovery_runner_sha256']
        shutil.copy2(recovery_path,r/'INFRASTRUCTURE_RECOVERY.json');shutil.copy2(runner,r/'recover_matrix.py')
        old=Path(recovery['original_matrix'])
        shutil.copy2(old.parent/'MATRIX_RUN.log',r/'STORAGE_INTERRUPTION.log')
        for item in recovery['infrastructure_reruns']:
            episode='init_'+str(item['initial_state_id'])+'__'+item['key'];src=Path(item['original_directory'])
            dst=r/'infrastructure_interrupted'/episode;dst.mkdir(parents=True)
            for name in ['RESULT.json','QUEUE_RECORD.json']:
                path=src/'runs/subset'/item['key']/name
                if path.exists():shutil.copy2(path,dst/name)
            log=src/'logs'/(item['key']+'.log')
            if log.exists():shutil.copy2(log,dst/'original_driver.log')
        recovery_notice=f'<h2>存储中断与恢复记录</h2><p>公共数据卷配额耗尽导致队列中断。{len(retained)}条原始完整回合（含失败）经轨迹、视频、模型哈希检查后保留；{len(interrupted)}条受存储错误影响的回合使用同一冻结源码补跑；另有{recovery["previously_unstarted"]}条此前尚未开始。原尝试全部保留，没有重试已完整记录的策略失败。原协议不做外部重试，本次仅对基础设施中断作了明确例外；恢复脚本和中断日志随完整包提供。</p>'
    flagged=[x for x in contacts if x['requires_manual_contact_review']]
    h=html.escape;details=[]
    for row in flagged:
        encoded=base64.b64encode((r/row['frame']).read_bytes()).decode()
        details.append(f"<details><summary>{h(row['language'])} · 初态{row['initial_state_id']} · {row['observed_nonfinger_mechanism_force_peak_N']:.2f} N</summary><p>第{row['peak_step']}步；{h(row['peak_phase'] or '')}；共{row['samples_above_point1N']}个20 Hz采样点高于0.1 N。</p><img alt=\"接触复核画面\" style=\"max-width:100%\" src=\"data:image/png;base64,{encoded}\"></details>")
    section=f'''{recovery_notice}<h2>完成轨迹的接触质量复核</h2><p class="notice">{len(contacts)}条完整完成轨迹中，{len(flagged)}条记录到高于0.1 N的非手指机构接触，已标记人工复核。任务完成统计保持不变；这些轨迹暂不直接作为高质量SFT样本。</p><p>原几何检查容差仍为1 mm，小于容差的接触也可能产生真实力。本项只检查20 Hz记录到的目标机构接触；未触发此项不等于已通过完整数据质量审核。</p>{''.join(details)}'''
    report=next(r.glob('*.html'));page=report.read_text();assert '<h2>验证说明</h2>' in page
    report.write_text(page.replace('<h2>验证说明</h2>',section+'<h2>验证说明</h2>',1))
    assert read(r/'RESULTS.json')==rows and read(r/'SUMMARY.json')==summary
    files={str(f.relative_to(r)):sha(f) for f in sorted(r.rglob('*')) if f.is_file() and f.name not in ['FILES_SHA256.json','DELIVERY_QA.json']}
    write(r/'FILES_SHA256.json',files)
    subprocess.run([__import__('sys').executable,'-B',str(r/'validate_contact_iteration.py'),str(r)],check=True)
    temporary=a.archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for f in sorted(r.rglob('*')):
            if f.is_file():archive.write(f,str(Path('delivery')/f.relative_to(r)))
    temporary.replace(a.archive)
    complete=dict(complete=True,post_audit_complete=True,zip=str(a.archive),zip_bytes=a.archive.stat().st_size,zip_sha256=sha(a.archive),
        report=str(report),report_bytes=report.stat().st_size,report_sha256=sha(report),manual_contact_review_episodes=len(flagged),
        source_commit=summary['source_revision'])
    write(r.parent/'DELIVERY_COMPLETE.json',complete)
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
