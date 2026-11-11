"""Geometry and boundary regressions for arena goals and articulation paths."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from graspgen_libero_bridge.env_adapter import EnvAdapter
from graspgen_libero_bridge.articulation import constrained_pose


def adapter(body=4, name='arena'):
    a=EnvAdapter.__new__(EnvAdapter)
    a.destination='arena';a.config={'goal_destination':'arena_goal'}
    a.task_env=SimpleNamespace(obj_body_id={'cup':7},object_sites_dict={'arena_goal':object()},parsed_problem={'regions':{'arena_goal':{'target':'arena'}}})
    a.obs={};a.env=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(
        site_name2id=lambda n:1,site_bodyid=[0,body],body_id2name=lambda n:name)))
    return a


def test_region_support_not_movable_registry():
    a=adapter();assert a.body_id('arena')==4 and a.body_id('cup')==7


@pytest.mark.parametrize('body,name',[(0,'world')])
def test_region_resolution_does_not_substitute_other_bodies(body,name):
    with pytest.raises(KeyError):adapter(body,name).body_id('arena')


def test_unknown_destination_stays_an_error():
    with pytest.raises(KeyError):adapter().body_id('missing')


def test_slide_target_tracks_axis_in_any_world_rotation():
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.4,-.2,1.1]).as_matrix();pose[:3,3]=[.2,.3,1.]
    axis=np.array([1.,2.,-.5]);axis/=np.linalg.norm(axis)
    result=constrained_pose(pose,axis,[.1,.2,.3],-.12,2)
    np.testing.assert_allclose(result[:3,3]-pose[:3,3],-.12*axis)
    np.testing.assert_array_equal(result[:3,:3],pose[:3,:3])


def test_hinge_target_stays_on_circle_and_rotates_grip():
    p=np.eye(4);p[:3,3]=[1.,0.,.3]
    q=constrained_pose(p,[0,0,1],[0,0,0],np.pi/2,3)
    np.testing.assert_allclose(q[:3,3],[0,1,.3],atol=1e-12)
    np.testing.assert_allclose(q[:3,:3]@np.array([1,0,0]),[0,1,0],atol=1e-12)
    np.testing.assert_array_equal(p[:3,3],[1,0,.3])


def test_free_joint_is_not_treated_as_an_articulation():
    with pytest.raises(ValueError):constrained_pose(np.eye(4),[1,0,0],[0,0,0],.1,0)


def test_arena_alias_uses_declared_region_target_not_body_spelling():
    assert adapter(4,'native_table_body').body_id('arena')==4
    a=adapter();a.task_env.parsed_problem['regions']['arena_goal']['target']='another_support'
    with pytest.raises(KeyError):a.body_id('arena')


def test_support_uses_physical_tabletop_and_excludes_movable_plate():
    from graspgen_libero_bridge.regions import support_surface
    m=SimpleNamespace(ngeom=3,nbody=4,body_parentid=np.array([0,0,0,0]),
        geom_bodyid=np.array([1,2,0]),geom_contype=np.ones(3),geom_conaffinity=np.ones(3),
        geom_type=np.array([6,6,0]),geom_size=np.array([[.4,.4,.02],[.1,.1,.03],[3,3,.1]]),
        site_name2id=lambda n:0,body_name2id=lambda n:3,geom_id2name=lambda g:None if g==0 else 'plate')
    d=SimpleNamespace(site_xpos=np.array([[0.,0.,.41]]),geom_xpos=np.array([[0,0,.42],[0,0,.44],[0,0,0]]),geom_xmat=np.tile(np.eye(3).reshape(9),(3,1)))
    a=SimpleNamespace(config={'goal_destination':'area'},env=SimpleNamespace(sim=SimpleNamespace(model=m,data=d)),task_env=SimpleNamespace(obj_body_id={'plate':2}),robot=SimpleNamespace(robot_model=SimpleNamespace(root_body='robot')))
    result=support_surface(a)
    assert result['physical_surface_z_m']==pytest.approx(.44)
    assert result['geom_names']==['geom_0']
    d.site_xpos[0,2]=1.
    with pytest.raises(ValueError):support_surface(a)


def test_anonymous_arena_contact_names_match_exported_scene_permissions():
    from graspgen_libero_bridge.controller import ContactMonitor
    m=SimpleNamespace(geom_bodyid=np.array([10,20]),geom_id2name=lambda g:'target' if g==0 else None)
    d=SimpleNamespace(ncon=1,contact=[SimpleNamespace(dist=-.002,geom1=0,geom2=1)])
    c=ContactMonitor.__new__(ContactMonitor);c.adapter=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(model=m,data=d)))
    c.robot_bodies={99};c.target_bodies={10};c.held=True;c.tolerance=.001;c.allowed_support_geoms={'geom_1'}
    assert c.check(True)==[]
    c.allowed_support_geoms.clear()
    assert c.check(True)[0]['held_object_collision'] is True
