"""Physical geometry tests for a planner-independent candidate screen."""
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np
from graspgen_libero_bridge.planning.grasp_environment import screen_grasp_environment


class Model:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def body_name2id(self, name):
        return self._model.body(name).id

    def body_id2name(self, index):
        return self._model.body(index).name

    def geom_id2name(self, index):
        return self._model.geom(index).name


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "CPU MuJoCo required")
class GraspEnvironmentTests(unittest.TestCase):
    def make_adapter(self, obstacle="", target_position="0 0 .10"):
        import mujoco
        xml = f'''<mujoco><worldbody>{obstacle}
          <body name="robot" pos="-1 0 0"><body name="hand">
            <geom name="palm" type="box" size=".03 .04 .02"/>
            <site name="grip" pos="0 0 .10"/>
            <body name="leftfinger" pos=".04 0 .08"><geom name="leftpad" type="box" size=".01 .01 .03"/></body>
            <body name="rightfinger" pos="-.04 0 .08"><geom name="rightpad" type="box" size=".01 .01 .03"/></body>
          </body></body>
          <body name="target" pos="{target_position}"><freejoint/>
            <geom name="object" type="box" size=".012 .012 .012" mass=".1"/>
          </body></worldbody></mujoco>'''
        native = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(native)
        data.time = 2.5
        data.qvel[:] = .003
        mujoco.mj_forward(native, data)
        robot = SimpleNamespace(gripper=SimpleNamespace(root_body="hand"),
            robot_model=SimpleNamespace(root_body="robot"), eef_site_id=native.site("grip").id)
        adapter = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(model=Model(native), data=data)),
            task_env=SimpleNamespace(robots=[robot], obj_body_id={"target": native.body("target").id}),
            target_object="target", state=lambda: {"gripper_width": .08})
        return adapter

    def candidate(self, rank=0, x=0., rotation=None):
        pose = np.eye(4)
        pose[:3, 3] = [x, 0., .10]
        if rotation is not None:
            pose[:3, :3] = rotation
        return dict(world_T_eef=pose.tolist(), score=1.-rank*.1, generation_index=rank)

    def snapshot(self, adapter):
        d = adapter.env.sim.data
        return {name: np.asarray(getattr(d, name)).copy() for name in
                ("qpos", "qvel", "ctrl", "qacc_warmstart", "geom_xpos", "geom_xmat", "time")}

    def assert_unchanged(self, adapter, before):
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(adapter.env.sim.data, name), value, err_msg=name)

    def test_blocked_high_score_is_removed_and_survivor_order_is_preserved(self):
        adapter = self.make_adapter('<geom name="wall" type="box" size=".025 .05 .025"/>')
        candidates = [self.candidate(0), self.candidate(1, .3), self.candidate(2, .5)]
        before = self.snapshot(adapter)
        kept, audit = screen_grasp_environment(adapter, candidates)
        self.assertEqual([x["generation_index"] for x in kept], [1, 2])
        self.assertIs(kept[0], candidates[1])
        self.assertEqual(audit["candidates"][0]["blocking_contact"]["obstacle_geom"], "wall")
        self.assert_unchanged(adapter, before)

    def test_open_finger_target_contact_is_permitted_but_palm_target_contact_is_not(self):
        adapter = self.make_adapter(target_position=".04 0 .08")
        kept, audit = screen_grasp_environment(adapter, [self.candidate()])
        self.assertEqual(len(kept), 1)
        self.assertGreater(audit["permitted_finger_target_pairs"], 0)
        adapter = self.make_adapter(target_position="0 0 0")
        kept, audit = screen_grasp_environment(adapter, [self.candidate()])
        self.assertFalse(kept)
        self.assertEqual(audit["candidates"][0]["blocking_contact"]["gripper_geom"], "palm")
        self.assertEqual(audit["candidates"][0]["blocking_contact"]["obstacle_geom"], "object")

    def test_fingers_cannot_pass_through_a_neighbour_object(self):
        adapter = self.make_adapter('<geom name="neighbour" type="sphere" pos=".04 0 .08" size=".012"/>')
        kept, audit = screen_grasp_environment(adapter, [self.candidate()])
        self.assertFalse(kept)
        self.assertEqual(audit["candidates"][0]["blocking_contact"]["obstacle_geom"], "neighbour")

    def test_handle_permissions_can_be_narrowed_to_specific_target_geoms(self):
        adapter = self.make_adapter(target_position=".04 0 .08")
        target = adapter.env.sim.model._model.geom('object').id
        kept, _ = screen_grasp_environment(adapter,[self.candidate()],allowed_finger_target_geoms=[])
        self.assertFalse(kept)
        kept, audit = screen_grasp_environment(adapter,[self.candidate()],allowed_finger_target_geoms=[target])
        self.assertEqual(len(kept),1)
        self.assertEqual(audit['allowed_finger_target_geom_ids'],[target])

    def test_handle_permissions_cannot_exempt_a_neighbour_obstacle(self):
        adapter = self.make_adapter('<geom name="neighbour" type="sphere" pos=".04 0 .08" size=".012"/>')
        neighbour = adapter.env.sim.model._model.geom('neighbour').id
        with self.assertRaisesRegex(ValueError,'stay on target'):
            screen_grasp_environment(adapter,[self.candidate()],allowed_finger_target_geoms=[neighbour])

    def test_rotation_and_measured_site_offset_are_applied(self):
        adapter = self.make_adapter('<geom name="neighbour" type="sphere" pos="0 .04 .08" size=".012"/>')
        turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        kept, audit = screen_grasp_environment(adapter, [self.candidate(), self.candidate(1, rotation=turn)])
        self.assertEqual([x["generation_index"] for x in kept], [0])
        self.assertFalse(audit["candidates"][1]["passed"])

    def test_visual_only_mesh_is_not_an_obstacle(self):
        adapter = self.make_adapter('<geom name="visual" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>')
        kept, _ = screen_grasp_environment(adapter, [self.candidate()])
        self.assertEqual(len(kept), 1)

    def test_plane_is_checked(self):
        adapter = self.make_adapter('<geom name="table" type="plane" pos="0 0 .01" size="2 2 .1"/>')
        kept, audit = screen_grasp_environment(adapter, [self.candidate()])
        self.assertFalse(kept)
        self.assertEqual(audit["candidates"][0]["blocking_contact"]["obstacle_geom"], "table")

    def test_invalid_later_candidate_cannot_mutate_live_state(self):
        adapter = self.make_adapter()
        before = self.snapshot(adapter)
        invalid = self.candidate(1)
        invalid["world_T_eef"][0][0] = 2.
        with self.assertRaisesRegex(ValueError, "Invalid grasp"):
            screen_grasp_environment(adapter, [self.candidate(), invalid])
        self.assert_unchanged(adapter, before)


if __name__ == "__main__":
    unittest.main()
