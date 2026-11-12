import argparse, os, sys, json, time, traceback, hashlib
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('key');p.add_argument('--gpu',type=int,default=0);p.add_argument('--tag',default='dev1')
p.add_argument('--init',type=int,default=0);p.add_argument('--inspect',action='store_true')
p.add_argument('--paths',type=Path)
p.add_argument('--output-root',type=Path)
args=p.parse_args()
r=Path(__file__).resolve().parents[2]
os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),MUJOCO_EGL_DEVICE_ID=str(args.gpu),MUJOCO_GL='egl',
    OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(r/'src'))
import numpy as np
from graspgen_libero_bridge.policy import evaluate
from graspgen_libero_bridge.policy.evaluate import PickPlace, write_json, progress
from graspgen_libero_bridge.control.controller import Recorder
from graspgen_libero_bridge.policy.task_policy import schedule
from graspgen_libero_bridge.policy.skills import ContactSkill, serializable

from graspgen_libero_bridge.runtime.config import load_task
row,cfg=load_task(args.key,gpu=args.gpu,init=args.init,paths_file=args.paths)
# Some unsupported-task placeholders named a nonexistent destination. Bootstrap
# against the real target, then resolve each destination from the parsed goals.
cfg['destination']=cfg['target_object']
out=(args.output_root or r/'runs')/args.tag/args.key
out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();mainrec=None;shared=None;close_env=None;original_goals=None
result=dict(key=args.key,suite=row['suite'],task_name=row['task_name'],initial_state_id=args.init,
    status='running',pipeline_success=False,official_goal_at_end=False,stages=[],
    release='20261113-contact-feedback-'+args.tag,uses_simulator_truth=True,
    stage_inputs='BDDL goals, object/region geometry, joint axes and measured contacts',
    source_sha256={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted((r/'src/graspgen_libero_bridge').rglob('*.py'))},
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class StageRecorder:
    def __init__(self, prefix):self.prefix=prefix;self.phase='initial'
    @property
    def phase(self):return mainrec.phase
    @phase.setter
    def phase(self,value):mainrec.phase=self.prefix+'/'+value
    def capture(self,action=None):mainrec.capture(action)
    def save(self):pass


def switch_target(stage, directory):
    scfg=dict(cfg,target_object=stage['target'],destination=stage.get('destination',stage['target']),
        goal_destination=stage['region'],goal_relation=stage['relation'])
    if 'packed_surface_xy_world_m' in stage:
        scfg['packed_surface_xy_world_m']=stage['packed_surface_xy_world_m']
        scfg['transfer_yaw_degrees']=[0]
    shared.config=scfg;shared.target_object=scfg['target_object'];shared.destination=scfg['destination']
    shared.run_dir=directory
    return scfg


def contact(stage, label):
    directory=out/label;directory.mkdir(exist_ok=False)
    scfg=switch_target(stage,directory)
    job.config=scfg;job.adapter=shared
    recorder=StageRecorder(label)
    skill=ContactSkill(job,recorder,stage,directory)
    try:
        data=skill.push() if stage['skill']=='push' else skill.articulate()
        return dict(status='completed',**data)
    finally:skill.close()


try:
    job=PickPlace(cfg,out/'environment');shared=job.adapter;shared.reset()
    close_env=shared.close;shared.close=lambda:None;shared.reset=lambda:shared.obs
    original_goals=json.loads(json.dumps(shared.task_env.parsed_problem['goal_state']))
    names=set(shared.task_env.obj_body_id)|set(shared.task_env.parsed_problem.get('fixtures',{}))
    names.update(shared.task_env.parsed_problem.get('regions',{}).get(k,{}).get('target')
                 for k in shared.task_env.parsed_problem.get('regions',{}))
    names.discard(None)
    plan=schedule(original_goals,names,shared.task_env.parsed_problem.get('regions',{}),shared.provenance['language'])
    from graspgen_libero_bridge.policy.surface_packing import surface_slots
    slots,packing=surface_slots(shared,plan,out/'packing_geometry')
    for index,xy in slots.items():plan[index]['packed_surface_xy_world_m']=xy
    write_json(out/'PACKING_PLAN.json',packing)
    result.update(original_goals=original_goals,skill_plan=plan,
        instruction_order_review_required='stack' in shared.provenance['language'].lower())
    write_json(out/'SKILL_PLAN.json',result)
    mainrec=Recorder(shared,out)
    evaluate.Recorder=lambda adapter,directory:StageRecorder(str(Path(directory).relative_to(out)))
    if args.inspect:
        m,d=shared.env.sim.model,shared.env.sim.data
        info=dict(goals=original_goals,plan=plan,state=shared.state(),objects=shared.task_env.obj_body_id,
            parsed_problem=shared.task_env.parsed_problem,joints=[],sites=[],geoms=[])
        for j in range(m.njnt):
            name=m.joint_id2name(j) or ''
            if name.startswith('robot') or name.startswith('gripper'):continue
            info['joints'].append(dict(name=name,type=int(m.jnt_type[j]),body=m.body_id2name(int(m.jnt_bodyid[j])),
                axis=d.xaxis[j].tolist(),anchor=d.xanchor[j].tolist(),range=m.jnt_range[j].tolist(),q=float(d.qpos[int(m.jnt_qposadr[j])])) )
        for s in range(m.nsite):
            name=m.site_id2name(s) or ''
            if name.startswith(('robot','gripper')):continue
            info['sites'].append(dict(name=name,xyz=d.site_xpos[s].tolist(),size=m.site_size[s].tolist(),rotation=d.site_xmat[s].reshape(3,3).tolist()))
        for g in range(m.ngeom):
            name=m.geom_id2name(g) or '';body=m.body_id2name(int(m.geom_bodyid[g])) or ''
            if body.startswith(('robot','gripper')):continue
            info['geoms'].append(dict(name=name,body=body,type=int(m.geom_type[g]),xyz=d.geom_xpos[g].tolist(),
                size=m.geom_size[g].tolist(),rotation=d.geom_xmat[g].reshape(3,3).tolist(),contact=bool(m.geom_contype[g] or m.geom_conaffinity[g])))
        write_json(out/'INSPECT.json',serializable(info));result['status']='inspected'
    else:
        for i,stage in enumerate(plan):
            label='stage_%02d'%i
            if shared.task_env._eval_predicate(stage['predicate']):
                result['stages'].append(dict(stage=stage,status='already_satisfied',physical_execution=False))
                continue
            # An In goal on a closed articulated region needs physical opening
            # before grasping the object. This is a precondition, not a rewrite
            # of the official goals or an initial-state edit.
            if stage['skill']=='pick_place' and stage['relation']=='in':
                target=stage['destination']
                obj=shared.task_env.fixtures_dict.get(target)
                if obj is not None and obj.object_properties.get('articulation',{}).get('default_open_ranges'):
                    opening=stage['region']
                    m=shared.env.sim.model
                    site=int(m.site_name2id(opening));body=int(m.site_bodyid[site])
                    if int(m.body_jntnum[body])==0:opening=target
                    predicate=['open',opening]
                    if not shared.task_env._eval_predicate(predicate):
                        prerequisite=dict(skill='articulate',target=target,region=opening,relation='open',predicate=predicate,requires_access=True)
                        data=contact(prerequisite,label+'_precondition_open')
                        result['stages'].append(dict(stage=prerequisite,precondition=True,**data))
                        if not data['official_subgoal']:raise RuntimeError('Opening precondition did not remain satisfied')
            before=shared.step_count
            if stage['skill']=='pick_place':
                directory=out/label
                scfg=switch_target(stage,directory)
                part=PickPlace(scfg,directory);part.adapter=shared
                partial=part.run()
                met=bool(shared.task_env._eval_predicate(stage['predicate']))
                complete=partial.get('status')=='completed' and bool(partial.get('release_completed'))
                data=dict(stage=stage,status=partial.get('status'),official_subgoal=met,
                    complete_execution=complete,reason=partial.get('reason'),failure=partial.get('failure'),
                    steps=shared.step_count-before)
                result['stages'].append(data)
                if not met or not complete:
                    result.update(status='stage_failed',reason=partial.get('reason') or 'Placement predicate false at stage end')
                    break
            else:
                data=contact(stage,label)
                result['stages'].append(dict(stage=stage,steps=shared.step_count-before,**data))
                if not data['official_subgoal']:
                    result.update(status='stage_failed',reason='Contact skill predicate false after release/settling')
                    break
            assert shared.task_env.parsed_problem['goal_state']==original_goals
            write_json(out/'PROGRESS.json',result)
        else:result['status']='completed'
        result['official_goal_at_end']=bool(shared.check_success())
        result['pipeline_success']=result['status']=='completed' and result['official_goal_at_end']
except (Exception,KeyboardInterrupt) as e:
    result.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',reason=str(e),traceback=traceback.format_exc())
    if hasattr(e,'record'):result['failure']=e.record()
finally:
    if shared is not None and shared.obs is not None:
        result['official_goal_at_end']=bool(shared.check_success());result['steps']=shared.step_count
        result['original_goals_unchanged']=original_goals is not None and shared.task_env.parsed_problem['goal_state']==original_goals
        sim=shared.env.sim
        native=getattr(sim.data,'_data',sim.data)
        fields=('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied',
                'mocap_pos','mocap_quat','userdata','eq_active','plugin_state')
        np.savez_compressed(out/'FINAL_PHYSICS_STATE.npz',time=float(native.time),
            **{name:np.asarray(getattr(native,name)).copy() for name in fields if hasattr(native,name)})
    if mainrec is not None:mainrec.save()
    if close_env is not None:close_env()
    result['wall_seconds']=time.monotonic()-start
    write_json(out/'RESULT.json',result)
print(json.dumps({k:result.get(k) for k in ['key','status','pipeline_success','reason','steps','wall_seconds']}),flush=True)
