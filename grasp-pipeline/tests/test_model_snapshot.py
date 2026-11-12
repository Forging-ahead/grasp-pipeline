"""A reset-time fixed-fixture edit must survive model replay."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from graspgen_libero_bridge.adapters.libero.model_snapshot import save_reset_model


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'CPU MuJoCo required')
class ModelSnapshotTests(unittest.TestCase):
    def test_compiled_fixture_edits_survive_without_mutating_live_state(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="fixture"><geom type="box" size=".1 .02 .05"/>
            <body><joint type="slide" axis="0 1 0"/><geom type="sphere" size=".02" mass="1"/>
            <site name="handle" pos="0 .1 0"/></body></body>
          </worldbody></mujoco>''')
        model.body_pos[1] = [.123456789123, -.31, .92]
        model.geom_pos[0] = [.00123456789123, 0., 0.]
        data = mujoco.MjData(model);data.qpos[0]=.03456789123;data.time=2.5
        mujoco.mj_forward(model,data)
        fields = ('qpos','qvel','qacc_warmstart','geom_xpos','geom_xmat','site_xpos')
        before = {name: np.asarray(getattr(data,name)).copy() for name in fields}
        sim = SimpleNamespace(model=model,data=data)
        with tempfile.TemporaryDirectory() as directory:
            metadata=save_reset_model(sim,directory)
            fresh=mujoco.MjModel.from_binary_path(str(Path(directory)/'reset_model.mjb'))
            replay=mujoco.MjData(fresh);replay.qpos[:]=data.qpos;mujoco.mj_forward(fresh,replay)
            for name in ('geom_xpos','geom_xmat','site_xpos'):
                np.testing.assert_array_equal(getattr(replay,name),before[name])
            with np.load(Path(directory)/'MODEL_PARAMETERS.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['body_pos'],model.body_pos)
            self.assertFalse(metadata['live_physics_modified'])
        for name,value in before.items():np.testing.assert_array_equal(getattr(data,name),value)
        self.assertEqual(data.time,2.5)
