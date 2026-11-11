import unittest
import numpy as np
from graspgen_libero_bridge.surface_packing import pack_rectangles


class SharedSupportPacking(unittest.TestCase):
    def test_two_asymmetric_origins_keep_complete_footprints_supported(self):
        boxes=np.array([[[-.03,-.05],[.05,.05]],[[-.04,-.05],[.04,.05]]])
        positions=pack_rectangles(boxes,[-.095,-.095],[.095,.095],[.075,.075])
        self.assertIsNotNone(positions)
        placed=[box+xy for box,xy in zip(boxes,positions)]
        for rectangle,xy in zip(placed,positions):
            self.assertTrue(np.all(rectangle[0]>=-.092));self.assertTrue(np.all(rectangle[1]<=.092))
            self.assertTrue(np.all(np.abs(xy)<=.072))
        self.assertTrue(any(placed[0][1,axis]<=placed[1][0,axis]-.0059 for axis in (0,1)))

    def test_oversized_pair_has_no_fake_packing_solution(self):
        boxes=np.array([[[-.06,-.06],[.06,.06]],[[-.06,-.06],[.06,.06]]])
        self.assertIsNone(pack_rectangles(boxes,[-.095,-.095],[.095,.095],[.075,.075]))
