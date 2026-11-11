import unittest
import numpy as np
from graspgen_libero_bridge.planning.planner_worker import finger_values


class PassiveFingerHandoff(unittest.TestCase):
    def test_observed_drawer_handoff_is_retained_exactly(self):
        measured=[.03887012818576937,-.04004159031143542]
        np.testing.assert_array_equal(finger_values({'q_fingers':measured}),measured)

    def test_empty_closure_soft_stop_is_not_an_isaac_sign_error(self):
        measured=[-.00002,.00001]
        np.testing.assert_array_equal(finger_values({'q_fingers':measured}),measured)

    def test_large_excursions_and_wrong_joint_convention_remain_invalid(self):
        for measured in [[-.003,.003],[.03,.03],[.043,-.043]]:
            with self.assertRaises(ValueError):finger_values({'q_fingers':measured})
