import unittest
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from graspgen_libero_bridge.policy.task_policy import schedule
from graspgen_libero_bridge.adapters.libero.placement_geometry import gravity_frame, orientation_candidates, fits_opening, opening_offsets


class GoalCompositionRegression(unittest.TestCase):
    def test_in_then_close_preserves_original_goals(self):
        goals=[['Close','cabinet_top'],['In','bottle','cabinet_top']]
        original=copy.deepcopy(goals)
        plan=schedule(goals,{'cabinet','bottle'},{'cabinet_top':{'target':'cabinet'}})
        self.assertEqual([x['relation'] for x in plan],['in','close'])
        self.assertEqual(goals,original)

    def test_compound_moka_switch_is_supported(self):
        plan=schedule([['On','pot','stove_cook'],['Turnon','stove']],{'pot','stove'},
            {'stove_cook':{'target':'stove'}})
        self.assertEqual([x['skill'] for x in plan],['articulate','pick_place'])

    def test_close_owner_waits_for_placement_in_its_region(self):
        plan=schedule([['In','mug','microwave_heating'],['Close','microwave']],{'mug','microwave'},
            {'microwave_heating':{'target':'microwave'}})
        self.assertEqual([x['relation'] for x in plan],['in','close'])

    def test_support_is_placed_before_upper_object(self):
        plan=schedule([['On','top','bottom'],['In','bottom','tray_region']],{'top','bottom','tray'},
            {'tray_region':{'target':'tray'}})
        self.assertEqual([x['target'] for x in plan],['bottom','top'])

    def test_push_instruction_routes_contact_skill(self):
        self.assertEqual(schedule([['On','plate','table_front']],{'plate','table'},
            {'table_front':{'target':'table'}},'Push the plate')[0]['skill'],'push')


class PlacementGeometryRegression(unittest.TestCase):
    def test_cabinet_region_vertical_axis_is_local_x(self):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        frame,size=gravity_frame(pose,[.03,.075,.10])
        self.assertGreater(frame[2,2],.99)
        self.assertAlmostEqual(size[2],.03)
        np.testing.assert_allclose(frame[:3,:3]@frame[:3,:3].T,np.eye(3),atol=1e-10)

    def test_flat_book_requires_rotation_into_narrow_opening(self):
        points=np.array([[x,y,z] for x in [-.09,.09] for y in [-.05,.05] for z in [-.005,.005]])
        pose=np.eye(4);frame=np.eye(4);size=np.array([.06,.02,.12])
        self.assertFalse(fits_opening(pose,points,frame,size))
        candidates=orientation_candidates(pose,points,frame,size,-.12)
        self.assertTrue(candidates)
        self.assertTrue(all(fits_opening(x,points,frame,size) for x in candidates))
        self.assertFalse(np.allclose(candidates[0][:3,:3],np.eye(3)))

    def test_oversized_object_cannot_fit(self):
        points=np.array([[x,y,z] for x in [-.10,.10] for y in [-.10,.10] for z in [-.10,.10]])
        self.assertEqual(orientation_candidates(np.eye(4),points,np.eye(4),np.array([.05,.05,.05]),-.05),[])

    def test_rotated_asset_and_offset_origin_fit_region(self):
        points=np.array([[x,y,z] for x in [-.09,.09] for y in [-.05,.05] for z in [-.005,.005]])
        points=points@Rotation.from_euler('xyz',[17,23,31],degrees=True).as_matrix().T+[.02,-.03,.04]
        frame=np.eye(4);frame[:3,:3]=Rotation.from_euler('z',37,degrees=True).as_matrix()
        candidates=orientation_candidates(np.eye(4),points,frame,np.array([.06,.02,.12]),-.12)
        self.assertTrue(candidates)
        self.assertTrue(all(fits_opening(x,points,frame,np.array([.06,.02,.12])) for x in candidates))

    def test_container_shift_stays_inside_opening(self):
        points=np.array([[x,y,z] for x in [-.02,.02] for y in [-.015,.015] for z in [-.04,.04]])
        frame=np.eye(4);size=np.array([.10,.08,.06]);pose=np.eye(4)
        shifts=opening_offsets(pose,points,frame,size)
        self.assertGreater(len(shifts),4)
        for shift in shifts:
            candidate=pose.copy();candidate[:3,3]+=shift
            self.assertTrue(fits_opening(candidate,points,frame,size))

    def test_bottom_origin_book_needs_tilt_to_meet_containment(self):
        points=np.array([[x,y,z] for x in [-.0035,.0035] for y in [-.06,.06] for z in [0.,.1344]])
        frame=np.eye(4);frame[2,3]=.07177
        size=np.array([.02775,.06216,.06046])
        candidates=orientation_candidates(np.eye(4),points,frame,size,0.,require_origin_containment=True)
        self.assertTrue(candidates)
        for candidate in candidates:
            rest_origin=(candidate[:3,3]-np.array([0.,0.,.015])-frame[:3,3])@frame[:3,:3]
            self.assertTrue(np.all(np.abs(rest_origin)<=size-.001))


if __name__=='__main__':unittest.main()
