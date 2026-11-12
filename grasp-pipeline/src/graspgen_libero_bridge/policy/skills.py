"""Contact skills using true geometry; no object state writes during execution."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.libero.articulation import discover, grip_candidates, handle_motion_pose, joint_goal, HandleMonitor, ArticulationPreviewMonitor
from graspgen_libero_bridge.planning.cartesian import plan_cartesian, _snapshot, _restore
from graspgen_libero_bridge.control.controller import Controller
from graspgen_libero_bridge.common.errors import ExecutionError
from graspgen_libero_bridge.policy.evaluate import write_json, progress
from graspgen_libero_bridge.common.rpc import WorkerClient
from graspgen_libero_bridge.adapters.libero.scene import _geometry
from graspgen_libero_bridge.planning.grasp_environment import screen_grasp_environment
from graspgen_libero_bridge.planning.grasp_contact import preview_handle_contacts


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class ContactSkill:
    def __init__(self, job, recorder, stage, out):
        self.job, self.a, self.stage, self.out = job, job.adapter, stage, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.c = Controller(self.a, recorder, stage['target'])
        self.planner = None
        self.audit, self.states = [], []
        self.info = None
        original_step = self.c.step

        def step(*args, **kwargs):
            value = original_step(*args, **kwargs)
            state = dict(step=self.a.step_count, phase=recorder.phase, eef=self.a.state()['world_T_eef'])
            if self.info is not None:
                state['joint_q'] = self.joint_q()
                state['finger_forces'] = self.handle_forces()
                state['nonfinger_mechanism_force_N'] = getattr(self,'nonfinger_mechanism_force',0.)
            self.states.append(state)
            return value
        self.c.step = step

    def joint_q(self):
        return float(self.a.env.sim.data.qpos[self.info['q_address']])

    def handle_forces(self):
        import mujoco
        m, d = self.a.env.sim.model, self.a.env.sim.data
        forces = dict(left=0., right=0.)
        nonfinger=0.
        for i in range(d.ncon):
            con = d.contact[i]
            names = [m.geom_id2name(int(g)) for g in (con.geom1, con.geom2)]
            if not any(n in self.info['handle_geom_names'] for n in names):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(getattr(m, '_model', m), getattr(d, '_data', d), i, force)
            for g in (con.geom1, con.geom2):
                body = m.body_id2name(int(m.geom_bodyid[int(g)])) or ''
                if 'leftfinger' in body or 'finger_joint1_tip' in body:
                    forces['left'] += max(0., float(force[0]))
                if 'rightfinger' in body or 'finger_joint2_tip' in body:
                    forces['right'] += max(0., float(force[0]))
                if body.startswith('robot0_') or (body.startswith('gripper0_') and 'finger' not in body):
                    nonfinger += max(0.,float(force[0]))
        self.nonfinger_mechanism_force=nonfinger
        return forces

    def native_plan(self, pose, phase, contact=False, monitor=None, free_rotation_axis_world=None):
        out = self.out/'plans'/(str(len(self.audit))+'_'+phase)
        reply = plan_cartesian(self.a, self.c.monitor if monitor is None else monitor,
                               pose, out, allow_finger_target=contact,
                               free_rotation_axis_world=free_rotation_axis_world)
        if monitor is not None:reply['coupled_articulation_preview'] = True
        self.audit.append(dict(phase=phase, reply=reply))
        progress('contact_preview', phase=phase, success=reply['success'], reason=reply.get('reason'))
        if reply.get('state_restored') is not True:
            raise RuntimeError('Contact preview did not restore physical state')
        return reply

    def global_plan(self, pose, phase):
        if self.planner is None:
            import robosuite
            cfg = self.a.config
            self.planner = WorkerClient(cfg['planner_python'], 'graspgen_libero_bridge.planning.planner_worker',
                self.out/'planner_worker', arguments=['--robosuite-assets',str(Path(robosuite.__file__).parent/'models/assets')],
                environment={'PYTHONPATH':str(Path(__file__).resolve().parents[2]),'CUDA_VISIBLE_DEVICES':str(cfg['planner_gpu'])})
            self.planner.request(dict(command='configure', scene=self.job.scene(),
                calibration_samples=self.a.calibration_samples(), output_dir=str(self.out/'planner_config'),
                max_robot_spheres=300, attachment_slots=128))
        reply = self.planner.request(dict(command='plan_pose', scene=self.job.scene(), goal_pose=pose.tolist(),
            out_dir=str(self.out/'plans'/(str(len(self.audit))+'_'+phase)), timeout_s=12,
            allow_target_contact=False, ignore_obstacles=[]))
        self.audit.append(dict(phase=phase, reply=reply))
        if reply['success']:
            with np.load(reply['path_file']) as path:
                check = self.c.monitor.validate_path(path['q_arm'], False)
            if not check['passed']:
                return dict(success=False, reason='actual_geometry_preflight', check=check)
        return reply

    def approach(self, candidates, gripper=0):
        # Exhaust cheap native approaches before starting the global planner.
        for global_fallback in (False, True):
            for index, (pre, grip) in enumerate(candidates):
                self.free_handle_roll = (getattr(self, 'candidate_specs', [{}] * len(candidates))[index]
                                         .get('free_handle_roll', False))
                app = self.global_plan(pre, 'global_approach') if global_fallback else self.native_plan(pre, 'approach')
                if global_fallback and app.get('reason')=='invalid_start_state':
                    raise ExecutionError('cuRobo rejected the current state before evaluating a target; see its proxy-collision diagnostics',
                        code='contact_global_start_invalid',category='planning',phase='approach',details=app)
                if not app['success']:
                    continue
                snap = _snapshot(self.a.env.sim)
                try:
                    with np.load(app['path_file']) as path:
                        self.a.env.sim.data.qpos[self.a.robot._ref_joint_pos_indexes] = path['q_arm'][-1]
                    self.a.env.sim.forward()
                    engage = self.native_plan(grip, 'engage', True)
                finally:
                    _restore(self.a.env.sim, snap)
                    self.a.robot.controller.update(force=True)
                    self.a.robot.controller.reset_goal()
                    self.a.task_env._update_observables(force=True)
                if not engage['success']:
                    continue
                if hasattr(self,'sweep_goal') and not self.preview_sweep(engage):
                    continue
                if hasattr(self,'push_specs') and not self.preview_push(engage,grip,index):
                    continue
                self.c.follow_path(app['path_file'], gripper=gripper, phase='approach')
                self.c.follow_path(engage['path_file'], gripper=gripper, phase='engage', allow_finger_target=True)
                return index
        raise ExecutionError('No executable approach/engagement in shared skill candidates',
            code='contact_approach_exhausted',category='planning',phase='approach')

    def preview_sweep(self, engage):
        """Check the intended joint motion before choosing a handle grasp.

        Both the mechanism and robot are moved only in a saved/restored static
        preview. This filters approaches that put the wrist against a limit
        halfway through the required turn. Physical execution still uses forces.
        """
        a,info=self.a,self.info
        snap=_snapshot(a.env.sim)
        try:
            with np.load(engage['path_file']) as path:
                a.env.sim.data.qpos[a.robot._ref_joint_pos_indexes]=path['q_arm'][-1]
            a.env.sim.forward()
            if getattr(self,'contact_mode','pinch')=='pinch':
                grip = preview_handle_contacts(a,self.c.monitor,
                    np.asarray(a.env.sim.data.qpos[a.robot._ref_joint_pos_indexes]))
                self.audit.append(dict(phase='handle_closure_preflight',reply=grip))
                if not grip['passed']:
                    return False
                a.env.sim.data.qpos[a.robot._ref_gripper_joint_pos_indexes]=grip['q_fingers']
                a.env.sim.forward()
            reference=np.asarray(a.state()['world_T_eef'])
            initial=self.joint_q()
            goal=self.sweep_goal
            radius=1. if info['joint_type']==2 else max(.04,float(np.linalg.norm(np.cross(info['axis'],reference[:3,3]-info['anchor']))))
            count=max(3,int(np.ceil(abs(goal-initial)*radius/.025))+1)
            for q in np.linspace(initial,goal,count)[1:]:
                previous_q=self.joint_q()
                previous_pose=np.asarray(a.state()['world_T_eef'])
                desired=handle_motion_pose(reference,info,q-initial,getattr(self,'free_handle_roll',False))
                coupled=ArticulationPreviewMonitor(a,self.c.monitor,info,previous_pose,previous_q,q)
                reply=self.native_plan(desired,'full_sweep_preflight',True,monitor=coupled,
                    free_rotation_axis_world=self.free_handle_axis())
                if not reply['success']:return False
                with np.load(reply['path_file']) as path:
                    a.env.sim.data.qpos[a.robot._ref_joint_pos_indexes]=path['q_arm'][-1]
                a.env.sim.data.qpos[info['q_address']]=q
                a.env.sim.forward()
                if self.c.monitor.check(True):return False
            return True
        finally:
            _restore(a.env.sim,snap)
            a.robot.controller.update(force=True);a.robot.controller.reset_goal()
            a.task_env._update_observables(force=True)

    def preview_push(self, engage, grip, index):
        """Check the robot's next short push before committing to its contact.

        This does not predict object dynamics. It only rejects pusher contacts
        that would drive the robot into a cabinet over the next bounded segment.
        """
        a=self.a;snap=_snapshot(a.env.sim)
        try:
            with np.load(engage['path_file']) as path:
                a.env.sim.data.qpos[a.robot._ref_joint_pos_indexes]=path['q_arm'][-1]
            a.env.sim.forward()
            direction,length=self.push_specs[index]
            desired=grip.copy();desired[:3,3]+=direction*length
            reply=self.native_plan(desired,'short_push_preflight',True)
            return bool(reply['success'])
        finally:
            _restore(a.env.sim,snap)
            a.robot.controller.update(force=True);a.robot.controller.reset_goal()
            a.task_env._update_observables(force=True)

    def free_handle_axis(self):
        if not (getattr(self,'free_handle_roll',False) and self.info.get('cylindrical_handle')
                and not self.info.get('rotary_knob') and self.info['joint_type']==3):
            return None
        bar = np.asarray(self.info['handle_axis'],dtype=float)
        axis = np.asarray(self.info['axis'],dtype=float)
        if abs(float(bar@axis)/(np.linalg.norm(bar)*np.linalg.norm(axis))) < .999:
            return None
        return axis

    def joint_motion_plan(self,desired,phase,goal_q):
        # The execution-side short preview must move the mechanism too, just
        # like the initial sweep preview. Otherwise the robot is checked
        # against a door frozen at its OLD angle and may be falsely rejected.
        # This wrapper is preview-only; the live contact monitor is unchanged.
        current=np.asarray(self.a.state()['world_T_eef'])
        coupled=ArticulationPreviewMonitor(self.a,self.c.monitor,self.info,current,self.joint_q(),goal_q)
        return self.native_plan(desired,phase,True,monitor=coupled,
            free_rotation_axis_world=self.free_handle_axis())

    def articulate(self):
        try:
            return self._articulate_attempt()
        except ExecutionError as error:
            if (error.code not in ('contact_approach_exhausted','contact_global_start_invalid') or self.stage['relation']!='open'
                    or self.info is None or self.info['joint_type']!=3 or self.info.get('rotary_knob')):
                raise
        # A door may be reachable in successive grasps even when no single
        # wrist branch can cover its full opening arc. Each leg is independently
        # previewed, physically driven, released and re-observed.
        if self.planner is not None:
            self.planner.close();self.planner=None
        legs=[]
        for index in range(4):
            child=ContactSkill(self.job,self.c.recorder,self.stage,self.out/'regrasp'/str(index))
            try:
                result=child._articulate_attempt(max_joint_segment=.45)
            finally:
                child.close()
            legs.append(result)
            write_json(self.out/'REGRASP_LEGS.json',serializable(legs))
            if self.a.task_env._eval_predicate(self.stage['predicate']):
                return dict(contact_mode='segmented_handle_open',legs=legs,official_subgoal=True)
            if abs(result['final_q']-result['initial_q'])<.025:
                raise ExecutionError('Regrasp leg made insufficient mechanism progress',
                    code='regrasp_no_progress',category='control',phase='articulate')
        raise ExecutionError('Door did not reach the original predicate within four regrasp legs',
            code='regrasp_budget_exhausted',category='control',phase='articulate')

    def _articulate_attempt(self,max_joint_segment=None):
        stage, a, c = self.stage, self.a, self.c
        region, relation = stage['region'], stage['relation']
        self.info = info = discover(a, region, self.out)
        c.monitor = HandleMonitor(a, stage['target'], info['handle_geom_names'])
        target = stage['target']
        obj = a.task_env.fixtures_dict.get(target) or a.task_env.objects_dict.get(target)
        properties = obj.object_properties['articulation']
        q0 = self.joint_q()
        qgoal = joint_goal(info,properties,relation,q0,self.stage.get('requires_access',False))
        full_goal=qgoal
        if max_joint_segment is not None:
            qgoal=q0+float(np.clip(qgoal-q0,-max_joint_segment,max_joint_segment))
        def reached():
            return bool(a.task_env._eval_predicate(stage['predicate']) or (
                max_joint_segment is not None and (qgoal-self.joint_q())*np.sign(qgoal-q0)<=.01))
        self.sweep_goal=qgoal
        write_json(self.out/'ARTICULATION.json',serializable(dict(info, initial_q=q0,target_q=qgoal,
            full_task_target_q=full_goal,max_joint_segment=max_joint_segment)))
        c.hold(20, -1, 'prepare_open', False)
        if info.get('rotary_knob'):
            candidates = []
            z = np.array([0.,0.,-1.])
            for flip in (1.,-1.):
                x = np.cross(info['handle_axis'],z)*flip
                x /= np.linalg.norm(x)
                grip = np.eye(4)
                grip[:3,:3] = np.column_stack((x,np.cross(z,x),z))
                grip[:3,3] = info['handle_center']
                pre = grip.copy();pre[2,3] += .08
                candidates.append((pre,grip))
            gap = .075
        else:
            # Closing approaches the external side of the handle too. Reversing
            # the desired joint direction must not put the pregrasp in the box.
            opening = float(np.mean(properties.get('default_open_ranges', [qgoal])))
            closing = float(np.mean(properties.get('default_close_ranges', [q0])))
            candidates = list(grip_candidates(info, opening, closing))
            initial_candidates = list(candidates)
            self.candidate_specs = [dict(group='original',free_handle_roll=False) for _ in candidates]
            if info.get('cylindrical_handle') and info['joint_type'] == 3:
                original = list(candidates)
                candidates.extend(original)
                self.candidate_specs.extend(dict(group='round_bar_roll',free_handle_roll=True) for _ in original)
            extra = list(grip_candidates(info,opening,closing,tilts=(5.,15.,25.,45.,65.,75.),
                axial_fractions=(0.,.5,-.5)))
            extra += list(grip_candidates(info,opening,closing,axial_fractions=(.5,-.5)))
            for roll in ((False,True) if info.get('cylindrical_handle') and info['joint_type']==3 else (False,)):
                candidates.extend(extra)
                self.candidate_specs.extend(dict(group='geometry_fallback',free_handle_roll=roll) for _ in extra)
            write_json(self.out/'HANDLE_CANDIDATES_ALL.json',serializable([
                dict(index=i,pre=pre,grip=grip,**spec)
                for i,((pre,grip),spec) in enumerate(zip(candidates,self.candidate_specs))]))
            thickness = max(2*float(np.sum(np.abs(grip[:3,0]@info['handle_rotation'])*info['handle_half_sizes'])) for _,grip in initial_candidates)
            gap = min(.075, thickness+.012)
        m = a.env.sim.model
        ranges = np.array([m.jnt_range[m.joint_name2id(n)] for n in a.robot.gripper.joints])
        a.robot.gripper.current_action = 2*(np.array([gap/2,-gap/2])-ranges[:,0])/(ranges[:,1]-ranges[:,0])-1
        c.hold(20, 0, 'prepare_width', False)
        if hasattr(self,'candidate_specs'):
            # Keep the original six approaches first for existing successes.
            # Extra geometric samples must be clear at the actual open width
            # before spending time on arm IK or full-motion previews.
            extras = [dict(world_T_eef=grip.tolist(),score=float(len(candidates)-i),generation_index=i)
                for i,((_,grip),spec) in enumerate(zip(candidates,self.candidate_specs))
                if spec['group']!='original']
            survivors, screen = screen_grasp_environment(a,extras,
                allowed_finger_target_geoms=info['handle_geoms'])
            screen['candidate_source']='Geometry search order; these scores are not GraspGen predictions'
            write_json(self.out/'HANDLE_ENVIRONMENT_SCREEN.json',screen)
            indexes={x['generation_index'] for x in survivors}
            chosen=[i for i,spec in enumerate(self.candidate_specs) if spec['group']=='original' or i in indexes]
            candidates=[candidates[i] for i in chosen]
            self.candidate_specs=[dict(self.candidate_specs[i],original_index=i) for i in chosen]
            write_json(self.out/'HANDLE_CANDIDATES.json',serializable([
                dict(index=i,pre=pre,grip=grip,**spec)
                for i,((pre,grip),spec) in enumerate(zip(candidates,self.candidate_specs))]))
        try:
            selected = self.approach(candidates)
        except ExecutionError as error:
            if relation!='close' or error.code not in ('contact_approach_exhausted','contact_global_start_invalid'):
                raise
            return self.close_with_surface_push(qgoal,q0)
        c.hold(30, 1, 'close_on_handle', True)
        sustained = sum(min(s.get('finger_forces',{}).values(),default=0.) > .1 for s in self.states[-10:])
        if sustained < 3:
            raise ExecutionError('No sustained bilateral compression on handle',
                code='handle_engagement_failed',category='contact',phase='close_on_handle')
        reference = np.asarray(a.state()['world_T_eef'])
        reference_q = self.joint_q()
        radius = 1. if info['joint_type']==2 else max(.04,float(np.linalg.norm(np.cross(info['axis'],reference[:3,3]-info['anchor']))))
        count = max(2,int(np.ceil(abs(qgoal-reference_q)*radius/.004))+1)
        for q in np.linspace(reference_q,qgoal,count)[1:]:
            desired = handle_motion_pose(reference,info,q-reference_q,getattr(self,'free_handle_roll',False))
            # A cylindrical handle permits rotation about its axis. Keep the
            # jaw orientation for a knob (which must transmit rotation).
            local = self.joint_motion_plan(desired,'joint_motion',q)
            if not local['success']:
                raise ExecutionError('Joint-constrained contact IK rejected',code='contact_ik_failed',
                    category='planning',phase='articulate',details=local)
            with np.load(local['path_file']) as path:
                c.set_joint_reference(path['q_arm'][-1])
                if self.free_handle_axis() is not None:
                    desired = path['world_T_eef'][-1].copy()
            c.move_to(desired,gripper=1,phase='articulate',allow_finger_target=True,
                position_tolerance=.0025,rotation_tolerance=.04,max_steps=100)
            if reached():
                break
        # A gripper reaching its pose is not a certificate that the mechanism
        # reached its joint target: compliance/slip can leave a rotary switch
        # short of the threshold. Correct using measured joint progress.
        stalled=0
        for correction_index in range(20):
            c.hold(10,1,'verify_held',True)
            if reached():break
            before=self.joint_q()
            delta=float(np.clip(qgoal-before,-(.004 if info['joint_type']==2 else .05),(.004 if info['joint_type']==2 else .05)))
            desired=handle_motion_pose(np.asarray(a.state()['world_T_eef']),info,delta,getattr(self,'free_handle_roll',False))
            local=self.joint_motion_plan(desired,'measured_joint_feedback',before+delta)
            if not local['success']:
                raise ExecutionError('Mechanism feedback correction rejected',code='mechanism_feedback_ik_failed',
                    category='planning',phase='articulate_feedback',details=local)
            with np.load(local['path_file']) as path:
                c.set_joint_reference(path['q_arm'][-1])
                if self.free_handle_axis() is not None:
                    desired=path['world_T_eef'][-1].copy()
            c.move_to(desired,gripper=1,phase='articulate_feedback',allow_finger_target=True,
                position_tolerance=.0015,rotation_tolerance=.004,max_steps=100)
            c.hold(10,1,'mechanism_progress',True)
            moved=abs(self.joint_q()-before)
            stalled=stalled+1 if moved<.00005 else 0
            if stalled>=3:
                raise ExecutionError('Mechanism stopped progressing under feedback',code='mechanism_stalled',category='control',phase='articulate_feedback')
        if not reached():
            raise ExecutionError('Mechanism did not reach official predicate within feedback budget',
                code='mechanism_goal_not_reached',category='control',phase='articulate_feedback')
        c.hold(20,1,'verify_held',True)
        c.hold(25,-1,'release_handle',True)
        # Retrace outward relative to the current mechanism, not the original
        # pregrasp, to avoid pulling a just-closed drawer open again.
        retreat = np.asarray(a.state()['world_T_eef'])
        direction = np.array([0.,0.,1.])
        if not info.get('rotary_knob'):
            # A round bar can roll inside the jaws. The current wrist axis is
            # therefore not a reliable record of the original approach side.
            direction=candidates[selected][0][:3,3]-candidates[selected][1][:3,3]
            direction=direction/np.linalg.norm(direction)
            if info['joint_type']==3:
                direction=Rotation.from_rotvec(np.asarray(info['axis'])*(self.joint_q()-q0)).apply(direction)
        retreat[:3,3] += direction*.07
        reply = self.native_plan(retreat, 'retreat', True)
        if not reply['success']:
            # Released jaws no longer require the wrist's original roll. Try
            # bounded orientation/translation alternatives before reporting a
            # failed retreat; every path and its contact-free endpoint is checked.
            current=np.asarray(a.state()['world_T_eef'])
            axis=np.asarray(info['axis']) if info['joint_type']==3 else current[:3,2]
            up=np.array([0.,0.,1.]);blend=direction+up
            blend=blend/np.linalg.norm(blend) if np.linalg.norm(blend)>1e-8 else up
            for vector,length in ((direction,.07),(up,.07),(blend,.07),(direction,.03),(up,.03)):
                candidate=current.copy();candidate[:3,3]+=vector*length
                option=self.native_plan(candidate,'retreat_alternative',True,free_rotation_axis_world=axis)
                if not option['success']:continue
                with np.load(option['path_file']) as path:
                    clear=c.monitor.validate_path(path['q_arm'][-1:],False)
                if clear['passed']:
                    reply=option;break
        if not reply['success']:
            raise ExecutionError('Post-release retreat rejected',code='contact_retreat_failed',
                category='planning',phase='retreat',details=reply)
        c.follow_path(reply['path_file'],gripper=-1,phase='retreat',allow_finger_target=True)
        c.hold(30,-1,'settle',False)
        return dict(selected_candidate=selected,free_handle_roll=getattr(self,'free_handle_roll',False),initial_q=q0,final_q=self.joint_q(),
            official_subgoal=bool(a.task_env._eval_predicate(stage['predicate'])),sustained_force_samples=sustained)

    def close_with_surface_push(self,qgoal,q0):
        """A close-only fallback: push an exposed moving face under feedback.

        Pinch candidates are exhausted first. No mechanism state is written
        during execution; finger contact and measured joint progress move it.
        """
        from graspgen_libero_bridge.adapters.libero.surface_contact import closing_push_candidates
        a,c,info=self.a,self.c,self.info
        self.contact_mode='closing_surface';self.free_handle_roll=False
        c.hold(25,1,'prepare_closed_mechanism_pusher',False)
        raw,moving=closing_push_candidates(a,info,qgoal,self.out)
        names=[a.env.sim.model.geom_id2name(g) for g in moving]
        c.monitor=HandleMonitor(a,self.stage['target'],names)
        self.info=dict(info,handle_geom_names=names)
        endpoint=[dict(world_T_eef=grip.tolist(),score=float(-i),generation_index=i)
                  for i,(_,grip,_) in enumerate(raw)]
        kept,audit=screen_grasp_environment(a,endpoint,allowed_finger_target_geoms=moving)
        chosen=[x['generation_index'] for x in kept[:32]]
        self.candidate_specs=[raw[i][2] for i in chosen]
        write_json(self.out/'CLOSING_SURFACE_SCREEN.json',audit)
        write_json(self.out/'CLOSING_SURFACE_CANDIDATES.json',serializable([
            dict(pre=pre,grip=grip,**spec) for pre,grip,spec in raw]))
        selected=self.approach([(raw[i][0],raw[i][1]) for i in chosen],gripper=1)
        spec=raw[chosen[selected]][2]
        geom=int(a.env.sim.model.geom_name2id(spec['geom']))
        relative=np.asarray(spec['geom_T_eef'])
        stalled=0;progress_samples=[]
        for iteration in range(180):
            if a.task_env._eval_predicate(self.stage['predicate']):break
            before=self.joint_q();current=np.asarray(a.state()['world_T_eef'])
            # A pusher is not attached to the panel. Re-register its chosen
            # contact against the CURRENT moving geometry instead of drifting
            # the contact forward from the measured wrist at every iteration.
            face=np.eye(4);face[:3,:3]=np.asarray(a.env.sim.data.geom_xmat[geom]).reshape(3,3)
            face[:3,3]=a.env.sim.data.geom_xpos[geom]
            contact=face@relative
            radius=1. if info['joint_type']==2 else max(.04,float(np.linalg.norm(np.cross(info['axis'],contact[:3,3]-info['anchor']))))
            maximum=.004/radius
            delta=float(np.clip(qgoal-before,-maximum,maximum))
            desired=handle_motion_pose(contact,info,delta)
            local=self.joint_motion_plan(desired,'surface_joint_feedback',before+delta)
            if not local['success']:
                raise ExecutionError('Moving-surface push continuation rejected',code='surface_push_ik_failed',
                    category='planning',phase='articulate_feedback',details=local)
            with np.load(local['path_file']) as path:c.set_joint_reference(path['q_arm'][-1])
            linear=float(np.linalg.norm(desired[:3,3]-current[:3,3]))
            angular=float(Rotation.from_matrix(desired[:3,:3]@current[:3,:3].T).magnitude())
            c.move_to(desired,gripper=1,phase='surface_joint_feedback',allow_finger_target=True,
                position_tolerance=min(.0015,max(.0001,.2*linear)),
                rotation_tolerance=min(.04,max(.0008,.2*angular)),max_steps=80,
                stop_when=lambda:a.task_env._eval_predicate(self.stage['predicate']))
            c.hold(4,1,'observe_surface_progress',True)
            moved=(self.joint_q()-before)*np.sign(qgoal-before)
            forces=self.handle_forces()
            progress_samples.append(dict(before=before,after=self.joint_q(),directed_progress=moved,forces=forces))
            write_json(self.out/'SURFACE_PUSH_PROGRESS.json',progress_samples)
            stalled=stalled+1 if moved<.00005 else 0
            if stalled>=5:
                raise ExecutionError('Moving surface made no joint progress',code='surface_push_stalled',category='control',phase='articulate_feedback')
        if not a.task_env._eval_predicate(self.stage['predicate']):
            raise ExecutionError('Surface push exhausted its feedback budget',code='surface_push_goal_not_reached',category='control',phase='articulate_feedback')
        # Keep fingers closed while disengaging, avoiding an opening wedge in a
        # shut door. Retreat opposite the measured closing tangent.
        current=np.asarray(a.state()['world_T_eef']);direction=np.asarray(info['axis'])*np.sign(qgoal-q0)
        if info['joint_type']==3:direction=np.cross(info['axis'],current[:3,3]-info['anchor'])*np.sign(qgoal-q0)
        direction/=np.linalg.norm(direction)
        retreat=current.copy();retreat[:3,3]-=direction*.07
        local=self.native_plan(retreat,'surface_retreat',True)
        if not local['success']:
            raise ExecutionError('Closed-pusher retreat rejected',code='contact_retreat_failed',category='planning',phase='retreat',details=local)
        c.follow_path(local['path_file'],gripper=1,phase='surface_retreat',allow_finger_target=True)
        c.hold(30,1,'settle',False)
        return dict(contact_mode='closing_surface',selected_candidate=selected,initial_q=q0,final_q=self.joint_q(),
            official_subgoal=bool(a.task_env._eval_predicate(self.stage['predicate'])),
            measured_progress_samples=len(progress_samples),allowed_contact_geoms=names)

    def push(self):
        a,c,stage = self.a,self.c,self.stage
        m,d = a.env.sim.model,a.env.sim.data
        site = int(m.site_name2id(stage['region']))
        center = np.asarray(d.site_xpos[site]).copy()
        from graspgen_libero_bridge.perception.pointcloud import sample_mujoco_object_points
        object0 = a.target_pose(stage['target']).copy()
        c.hold(25,1,'prepare_closed_pusher',False)
        (self.out/'pusher_geometry').mkdir(parents=True,exist_ok=True)
        segments=[]
        for segment in range(8):
            if a.task_env._eval_predicate(stage['predicate']):break
            target_pose=a.target_pose(stage['target']).copy()
            points=sample_mujoco_object_points(a.env,stage['target'],count=4096,seed=0,geometry='collision')
            direction=center-target_pose[:3,3];direction[2]=0.
            distance=float(np.linalg.norm(direction))
            if distance<.002:break
            direction/=distance
            current=np.asarray(a.state()['world_T_eef'])
            finger_points=[]
            for g in range(m.ngeom):
                body=m.body_id2name(int(m.geom_bodyid[g])) or ''
                if 'finger' in body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    world=_geometry(m,d,g,self.out/'pusher_geometry',20.)[1]
                    finger_points.append((world-current[:3,3])@current[:3,:3])
            local_points=np.concatenate(finger_points)
            candidates=[];candidate_directions=[];angles=[];self.push_specs=[]
            # If the direct rear contact is blocked by a fixture, a bounded
            # oblique push can clear it, followed by a fresh observed replan.
            for angle in (0.,45.,-45.,75.,-75.):
                push_direction=Rotation.from_euler('z',angle,degrees=True).apply(direction)
                extent=float(np.max((points-target_pose[:3,3])@(-push_direction)))
                for tilt in (0.,25.,45.):
                    z=-push_direction*np.sin(np.deg2rad(tilt))+np.array([0.,0.,-1.])*np.cos(np.deg2rad(tilt))
                    for flip in (1.,-1.):
                        x=np.cross([0.,0.,1.],push_direction)*flip
                        pose=np.eye(4);pose[:3,:3]=np.column_stack((x,np.cross(z,x),z))
                        local=local_points@pose[:3,:3].T
                        front=float(np.max(local@push_direction))
                        pose[:3,3]=target_pose[:3,3]-push_direction*(extent+front+.008)
                        pose[2,3]=float(points[:,2].min())+.0015-float(local[:,2].min())
                        pre=pose.copy();pre[2,3]+=.20
                        candidates.append((pre,pose));candidate_directions.append(push_direction);angles.append(angle)
                        self.push_specs.append((push_direction,min(.06,distance+.02)))
            selected=self.approach(candidates,gripper=1)
            reference=candidates[selected][1].copy();push_direction=candidate_directions[selected]
            maximum=self.push_specs[selected][1]
            segments.append(dict(angle_degrees=angles[selected],start=target_pose.tolist(),max_push_m=maximum))
            for length in np.arange(.004,maximum,.004):
                desired=reference.copy();desired[:3,3]+=push_direction*length
                reply=self.native_plan(desired,'push',True)
                if not reply['success']:
                    # No physical command was issued for this rejected preview.
                    # Preserve the current contact-free state, retreat and
                    # choose another short contact from the new observation.
                    segments[-1]['preview_stop']=reply
                    break
                with np.load(reply['path_file']) as path:c.set_joint_reference(path['q_arm'][-1])
                c.move_to(desired,gripper=1,phase='push',allow_finger_target=True,position_tolerance=.0025,max_steps=60)
                self.states[-1]['object_pose']=a.target_pose(stage['target']).tolist()
                if a.task_env._eval_predicate(stage['predicate']):break
            retreat=np.asarray(a.state()['world_T_eef'])
            # Leave the fixture at the height that qualified on entry; the
            # old fixed 10cm retreat could strand cuRobo's conservative hand
            # spheres inside a cabinet even when MuJoCo contacts were clear.
            retreat[2,3]=max(retreat[2,3]+.10,float(candidates[selected][0][2,3]))
            reply=self.native_plan(retreat,'retreat',True)
            if not reply['success']:raise RuntimeError('Push retreat rejected')
            c.follow_path(reply['path_file'],gripper=1,phase='retreat',allow_finger_target=True)
            c.hold(15,1,'observe_after_push',False)
            segments[-1]['end']=a.target_pose(stage['target']).tolist()
            write_json(self.out/'PUSH_SEGMENTS.json',segments)
        c.hold(40,1,'settle',False)
        return dict(official_subgoal=bool(a.task_env._eval_predicate(stage['predicate'])),segments=segments,
            object_translation=(a.target_pose(stage['target'])[:3,3]-object0[:3,3]).tolist())

    def close(self):
        if self.planner is not None:self.planner.close()
        write_json(self.out/'CONTACT_STATES.json',serializable(self.states))
        write_json(self.out/'PLANS.json',serializable(self.audit))
        write_json(self.out/'EXECUTION_CHECKS.json',serializable(self.c.events))
