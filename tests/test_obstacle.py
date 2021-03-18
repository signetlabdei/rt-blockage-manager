# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.obstacle import SphereObstacle
from src.geometry import Point, Segment
from src.ray import Ray
import pytest


def test_sphere_obstacle():
    o = SphereObstacle(center=Point(0, 0, 0),
                       radius=10,
                       reflection_loss=20,
                       transmission_loss=30)
    assert o.center == Point(0, 0, 0)
    assert o.center == o.location()
    assert o.radius == 10
    assert o.reflection_loss() == 20
    assert o.transmission_loss() == 30

    # check repr string
    o_repr = eval(str(o))
    assert o.center == o_repr.center
    assert o.radius == o_repr.radius
    assert o.reflection_loss() == o_repr.reflection_loss()
    assert o.transmission_loss() == o_repr.transmission_loss()


def test_sphere_obstacle_obstructs():
    o = SphereObstacle(center=Point(0, 0, 0),
                       radius=5)

    # Point inside sphere: should be obstructed as obstacle is assumed to be solid
    assert o.obstructs(Segment(Point(0, 0, 0), Point(10, 10, 10)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(5, 5, 5), Point(10, 10, 10)))

    # Obstructed segment
    assert o.obstructs(Segment(Point(-10, -10, -10), Point(10, 10, 10)))

    # Unobstructed ray
    assert not o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(0, 10, 0), Point(-10, 0, 0)]))

    # Obstructed ray
    assert o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(-10, 0, 0), Point(0, 10, 0)]))


def test_sphere_obstacle_specular_reflection():
    o = SphereObstacle(center=Point(0, 0, 0),
                       radius=5)

    # Invalid points (pa inside sphere): no reflection
    p = o.specular_reflection(Point(1, 1, 1), Point(10, 10, 10))
    assert p is None

    # Reflection on xy-plane
    p = o.specular_reflection(Point(10, 10, 0), Point(-10, 10, 0))
    assert (p - Point(0, 5, 0)).length() == pytest.approx(0)

    # Reflection on yz-plane
    p = o.specular_reflection(Point(0, 10, 10), Point(0, 10, -10))
    assert (p - Point(0, 5, 0)).length() == pytest.approx(0)

    # Reflection on xz-plane
    p = o.specular_reflection(Point(10, 0, 10), Point(-10, 0, 10))
    assert (p - Point(0, 0, 5)).length() == pytest.approx(0)
