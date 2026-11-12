"""Physical handle closure, goal selection, and access-precondition regressions."""
import importlib.util
from types import SimpleNamespace
import unittest
import numpy as np
from graspgen_libero_bridge.planning.grasp_contact import preview_handle_contacts
from graspgen_libero_bridge.adapters.libero.articulation import HandleMonitor, joint_goal, access_reached, stage_ready
from unittest.mock import patch
from graspgen_libero_bridge.policy.task_policy import schedule


class Model:
    def __init__(self, native):self._model=native
    def __getattr__(self,name):return getattr(self._model,name)
    def body_name2id(self,name):return self._model.body(name).id
    def body_id2name(self,index):return self._model.body(index).name
    def geom_id2name(self,index):return self._model.geom(index).name


class Sim:
    def __init__(self,model,data):self.model,self.data=Model(model),data
    def get_state(self):return SimpleNamespace(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),time=self.data.time)
    def set_state(self,state):
        self.data.qpos[:]=state.qpos;self.data.qvel[:]=state.qvel;self.data.time=state.time
    def forward(self):
        import mujoco
        mujoco.mj_forward(self.model._model,self.data)
    def step(self):raise AssertionError('Handle preview cannot step dynamics')


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'CPU MuJoCo required')
class HandlePreviewTests(unittest.TestCase):
    def adapter(self, target_x=0., neighbour=''):
        import mujoco
        xml=['<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>',neighbour,
             '<body name="robot0_base">']
        for i in range(7):
            kind='slide' if i<3 else 'hinge'
            axis=('1 0 0','0 1 0','0 0 1')[i%3]
            xml.append(f'<body name="link{i}"><joint name="joint{i}" type="{kind}" axis="{axis}" range="-1 1"/><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>')
        xml.append('''<site name="eef"/>
            <body name="leftfinger"><joint name="left" type="slide" axis="-1 0 0" range="0 .04"/><geom name="left_pad" type="box" size=".003 .01 .012" mass=".1"/></body>
            <body name="rightfinger"><joint name="right" type="slide" axis="-1 0 0" range="-.04 0"/><geom name="right_pad" type="box" size=".003 .01 .012" mass=".1"/></body>''')
        xml.extend(['</body>']*8)
        xml.append(f'<body name="fixture" pos="{target_x} 0 0"><geom name="bar" type="capsule" size=".008 .04"/></body></worldbody></mujoco>')
        m=mujoco.MjModel.from_xml_string(''.join(xml));d=mujoco.MjData(m)
        d.qpos[7:]=[.02,-.02];d.time=2.5
        sim=Sim(m,d);sim.forward();d.qacc_warmstart[:]=np.linspace(.01,.02,m.nv)
        robot=SimpleNamespace(_ref_joint_pos_indexes=np.arange(7),_ref_gripper_joint_pos_indexes=np.array([7,8]),eef_site_id=m.site('eef').id)
        a=SimpleNamespace(env=SimpleNamespace(sim=sim),task_env=SimpleNamespace(robots=[robot],obj_body_id={'fixture':m.body('fixture').id}))
        return a,HandleMonitor(a,'fixture',['bar'])

    def check(self,adapter,monitor):
        d=adapter.env.sim.data
        fields=['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied']
        before={name:np.asarray(getattr(d,name)).copy() for name in fields}
        result=preview_handle_contacts(adapter,monitor,np.zeros(7))
        for name,value in before.items():np.testing.assert_array_equal(getattr(d,name),value,err_msg=name)
        self.assertEqual(d.time,2.5)
        return result

    def test_bilateral_grip_uses_closed_fingers_and_restores_state(self):
        a,m=self.adapter();result=self.check(a,m)
        self.assertTrue(result['passed'],result)
        self.assertTrue(all(abs(q)<.02 for q in result['q_fingers']))
        self.assertLess(result['contact_normal_along_closing_axis']['left']*result['contact_normal_along_closing_axis']['right'],-.25)

    def test_preview_does_not_open_into_a_neighbour_wall(self):
        a,m=self.adapter(neighbour='<geom name="wall" type="box" pos="-.04 0 0" size=".004 .02 .02"/>')
        result=self.check(a,m)
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['measured_open_q_fingers'],[.02,-.02])

    def test_empty_and_unbalanced_grips_are_rejected(self):
        for offset in [.15,.006]:
            a,m=self.adapter(target_x=offset)
            self.assertFalse(self.check(a,m)['passed'])

    def test_world_collision_during_closure_is_not_exempted(self):
        a,m=self.adapter(neighbour='<geom name="obstacle" type="sphere" pos="-.009 0 .016" size=".006"/>')
        result=self.check(a,m)
        self.assertFalse(result['passed'],result)
        self.assertTrue(result['world_contacts'])


class JointGoalTests(unittest.TestCase):
    def test_standalone_door_does_not_require_the_interval_midpoint(self):
        info=dict(joint_type=3,limits=[-2.094,0.])
        props={'default_open_ranges':[-2.094,-1.3],'default_close_ranges':[-.005,0.]}
        self.assertAlmostEqual(joint_goal(info,props,'open',0.),-1.35)
        self.assertAlmostEqual(joint_goal(info,props,'open',0.,True),-1.697)

    def test_access_opening_does_not_close_an_already_wider_door(self):
        info=dict(joint_type=3,limits=[-2.094,0.])
        props={'default_open_ranges':[-2.094,-1.3],'default_close_ranges':[-.005,0.]}
        self.assertAlmostEqual(joint_goal(info,props,'open',-1.9,True),-1.9)
        self.assertFalse(access_reached(info,props,-1.32))
        self.assertTrue(access_reached(info,props,-1.9))

    def test_official_open_predicate_cannot_skip_an_insufficient_access_aperture(self):
        info=dict(joint_type=3,limits=[-2.094,0.],q_address=0)
        props={'default_open_ranges':[-2.094,-1.3],'default_close_ranges':[-.005,0.]}
        obj=SimpleNamespace(object_properties={'articulation':props})
        adapter=SimpleNamespace(task_env=SimpleNamespace(_eval_predicate=lambda _:True,
            fixtures_dict={'door':obj},objects_dict={}),env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=np.array([-1.32])))))
        stage=dict(skill='articulate',target='door',region='door',predicate=['open','door'],requires_access=True)
        with patch('graspgen_libero_bridge.adapters.libero.articulation.discover',return_value=info):
            self.assertFalse(stage_ready(adapter,stage,'/unused'))
            adapter.env.sim.data.qpos[0]=-1.8
            self.assertTrue(stage_ready(adapter,stage,'/unused'))

    def test_drawer_and_switch_targets_remain_unchanged(self):
        info=dict(joint_type=2,limits=[-.16,.01])
        props={'default_open_ranges':[-.16,-.14],'default_close_ranges':[0.,.005]}
        self.assertAlmostEqual(joint_goal(info,props,'open',0.),-.15)
        self.assertAlmostEqual(joint_goal(info,props,'close',-.15),.0025)
        knob=dict(joint_type=3,limits=[-.1,2.2])
        self.assertAlmostEqual(joint_goal(knob,{'default_turnon_ranges':[.5,2.1]},'turnon',0.),.65)

    def test_opening_for_insertion_is_marked_as_access(self):
        goals=[['Open','door'],['In','mug','door_inside']]
        stages=schedule(goals,{'door','mug'},{'door_inside':{'target':'door'}})
        self.assertTrue(stages[0]['requires_access'])
        self.assertEqual(goals,[['Open','door'],['In','mug','door_inside']])


if __name__=='__main__':unittest.main()
