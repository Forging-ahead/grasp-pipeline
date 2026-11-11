import unittest
import numpy as np
from graspgen_libero_bridge.policy.placement import container_release_candidates, can_retrace_after_ik_failure


class ContainerReleaseTests(unittest.TestCase):
    def test_retrace_forbids_corrupt_state_and_technical_errors(self):
        self.assertTrue(can_retrace_after_ik_failure({'state_restored':True,'reason':'cartesian_ik_iteration_limit'}))
        self.assertFalse(can_retrace_after_ik_failure({'state_restored':False,'reason':'cartesian_ik_iteration_limit'}))
        self.assertFalse(can_retrace_after_ik_failure({'state_restored':True,'reason':'cartesian_state_restore_failed'}))

    def fixture(self):
        return np.array([[x, y, z] for x in (-.02, .02) for y in (-.03, .03) for z in (0, .04)])

    def test_lowest_first_and_grasp_relation_preserved(self):
        goal=np.eye(4);goal[2,3]=.02
        relation=np.eye(4);relation[2,3]=.08
        rows=container_release_candidates(goal,relation,self.fixture(),np.eye(4),[.06,.06,.07],[.1,0.,.04])
        self.assertEqual([r['offset_m'] for r in rows],[0.,.04,.1])
        for r in rows:
            np.testing.assert_allclose(r['eef_pose']@relation,r['object_pose'])

    def test_object_footprint_must_clear_opening(self):
        self.assertEqual(container_release_candidates(np.eye(4),np.eye(4),self.fixture(),np.eye(4),[.02,.06,.07],[0.,.1]),[])

    def test_opening_coordinates_handle_yaw(self):
        points=np.array([[x,y,0] for x in (-.05,.05) for y in (-.02,.02)])
        region=np.eye(4);region[:2,:2]=[[0,-1],[1,0]]
        self.assertEqual(len(container_release_candidates(np.eye(4),np.eye(4),points,region,[.04,.08,.07],[0.])),1)
        self.assertEqual(container_release_candidates(np.eye(4),np.eye(4),points,np.eye(4),[.04,.08,.07],[0.]),[])

    def test_excessive_release_height_and_tilted_container_rejected(self):
        for offsets in ([.13],[-.01],[float('nan')]):
            with self.assertRaises(ValueError):
                container_release_candidates(np.eye(4),np.eye(4),self.fixture(),np.eye(4),[.06,.06,.07],offsets)
        region=np.eye(4);region[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
        with self.assertRaises(ValueError):
            container_release_candidates(np.eye(4),np.eye(4),self.fixture(),region,[.06,.06,.07],[0.])

    def test_small_physical_container_lean_uses_gravity_corridor(self):
        a=np.deg2rad(3.);region=np.eye(4)
        region[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
        self.assertEqual(len(container_release_candidates(np.eye(4),np.eye(4),self.fixture(),region,[.06,.06,.07],[.1])),1)
