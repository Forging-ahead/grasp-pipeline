"""Moving-panel contacts must follow geometry and preserve the live state."""
import importlib.util
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from graspgen_libero_bridge.adapters.libero.surface_contact import closing_push_candidates


class Model:
    def __init__(self,model):self._model=model
    def __getattr__(self,name):return getattr(self._model,name)
    def body_id2name(self,index):return self._model.body(index).name
    def geom_id2name(self,index):return self._model.geom(index).name


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'CPU MuJoCo required')
class SurfaceContactTests(unittest.TestCase):
    def sample(self,axis='0 1 0',joint='slide',target=.1):
        import mujoco
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <body name="housing"><geom name="fixed_wall" type="box" pos="0 .4 .1" size=".2 .01 .1"/>
            <body name="panel"><joint type="{joint}" axis="{axis}"/><geom name="moving_face" type="box" pos=".1 0 .1" size=".1 .01 .1" mass="1"/></body></body>
          <body name="leftfinger" pos="-.3 0 .2"><geom name="finger_pad" type="box" size=".01 .01 .02"/></body>
          </worldbody></mujoco>''')
        data=mujoco.MjData(model);data.qpos[0]=0.;data.time=2.5;mujoco.mj_forward(model,data)
        adapter=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(model=Model(model),data=data)),state=lambda:{'world_T_eef':np.eye(4)})
        info=dict(body_id=model.body('panel').id,joint_type=2 if joint=='slide' else 3,
            q_address=0,axis=np.asarray(list(map(float,axis.split()))),anchor=np.zeros(3))
        before={name:np.asarray(getattr(data,name)).copy() for name in ('qpos','qvel','geom_xpos','geom_xmat')}
        with tempfile.TemporaryDirectory() as directory:raw,moving=closing_push_candidates(adapter,info,target,directory)
        for name,value in before.items():np.testing.assert_array_equal(getattr(data,name),value)
        self.assertEqual(data.time,2.5)
        self.assertEqual(moving,[model.geom('moving_face').id])
        return raw

    def test_slide_push_uses_facing_moving_surface_and_preserves_state(self):
        raw=self.sample();self.assertTrue(raw)
        for pre,grip,spec in raw:
            self.assertEqual(spec['geom'],'moving_face')
            np.testing.assert_allclose(spec['normal'],[0.,1.,0.])
            self.assertLess(pre[1,3],grip[1,3])
            np.testing.assert_allclose(grip[:3,:3].T@grip[:3,:3],np.eye(3),atol=1e-12)

    def test_hinge_contact_uses_the_measured_axis_and_tangent(self):
        raw=self.sample(axis='0 0 1',joint='hinge',target=.2)
        self.assertTrue(raw)
        for _,_,spec in raw:np.testing.assert_allclose(spec['normal'],[0.,1.,0.])

    def test_box_face_without_a_usable_motion_normal_is_rejected(self):
        self.assertFalse(self.sample(axis='1 0 0'))
