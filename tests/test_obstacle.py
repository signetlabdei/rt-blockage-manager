# AUTHOR(S):
# Paolo Testolina <paolo.testolina@dei.unipd.it>
# Alessandro Traspadini <alessandro.traspadini@dei.unipd.it>
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

import functools
from src.diffraction_models import atan_diffraction, dke_kunisch
from src.obstacle import OrthoScreenObstacle, SphereObstacle
from src.geometry import Line, Point, Segment, distance, project
from src.mobility_model import ConstantPositionMobilityModel
from src.ray import Ray
import pytest

# SphereObstacle
def test_sphere_obstacle():
    pos = Point(0, 0, 0)
    o = SphereObstacle(mm=ConstantPositionMobilityModel(pos),
                       radius=10,
                       reflection_loss=20,
                       transmission_loss=30)

    assert o.location() == Point(0, 0, 0)
    assert o.radius == 10
    assert o.reflection_loss() == 20
    assert o.transmission_loss() == 30

    # check repr string
    o_repr = eval(str(o))
    assert o.location() == o_repr.location()
    assert o.radius == o_repr.radius
    assert o.reflection_loss() == o_repr.reflection_loss()
    assert o.transmission_loss() == o_repr.transmission_loss()

    # check update of mobility model
    o.update(1)
    assert o.location() == Point(0, 0, 0)


def test_sphere_obstacle_obstructs():
    o = SphereObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                       radius=5)

    # Point inside sphere: should be obstructed as obstacle is assumed to be solid
    assert o.obstructs(Segment(Point(0, 0, 0), Point(10, 10, 10)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(5, 5, 5), Point(10, 10, 10)))

    # Obstructed segment
    assert o.obstructs(Segment(Point(-10, -10, -10), Point(10, 10, 10)))

    # Unobstructed ray
    assert not o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(0, 10, 0), Point(-10, 0, 0)]))

    # Unobstructed ray with corner reflection
    assert not o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(0, 10, 0), Point(0, 10, 0), Point(-10, 0, 0)]))

    # Obstructed ray
    assert o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(-10, 0, 0), Point(0, 10, 0)]))

    # Obstructed ray with corner reflection
    assert o.obstructs(Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(-10, 0, 0), Point(-10, 0, 0), Point(0, 10, 0)]))

    # Type error
    with pytest.raises(TypeError):
        o.obstructs(0)  # type: ignore


def test_sphere_obstacle_specular_reflection():
    o = SphereObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                       radius=5)

    # Invalid points (pa inside sphere): no reflection
    p = o.specular_reflection(Point(1, 1, 1), Point(10, 10, 10))
    assert p is None

    # Reflection on xy-plane
    p = o.specular_reflection(Point(10, 10, 0), Point(-10, 10, 0))
    assert p is not None
    assert (p - Point(0, 5, 0)).length() == pytest.approx(0)

    # Reflection on yz-plane
    p = o.specular_reflection(Point(0, 10, 10), Point(0, 10, -10))
    assert p is not None
    assert (p - Point(0, 5, 0)).length() == pytest.approx(0)

    # Reflection on xz-plane
    p = o.specular_reflection(Point(10, 0, 10), Point(-10, 0, 10))
    assert p is not None
    assert (p - Point(0, 0, 5)).length() == pytest.approx(0)


# OrthoScreenObstacle
def test_ortho_screen_obstacle():
    pos = Point(0, 0, 0)
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(pos),
                            width=0.7,
                            height=1.8,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    assert o.location() == Point(0, 0, 0)
    assert o.width == 0.7
    assert o.height == 1.8
    assert o.reflection_loss() == 5
    assert o.transmission_loss() == 30
    assert o._diffraction_loss_model is None
    assert o._distance_threshold == 0

    # check update of mobility model
    o.update(1)
    assert o.location() == Point(0, 0, 0)


@pytest.mark.parametrize("segment", [Segment(Point(-10, 0, 0), Point(10, 0, 0)),
                                    Segment(Point(0, -10, 0), Point(0, 10, 0)),
                                    Segment(Point(0, 0, -10), Point(0, 0, 10)),
                                    Segment(Point(-10, 0, 0), Point(0, 0, 10))
                                    ])
def test_ortho_screen_obstacle_get_ortho_screen_edges(segment):
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                            width=0.7,
                            height=1.8,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    par = o._get_ortho_screen(segment)

    assert (par.adj2 - par.p0).length() == pytest.approx(o.height)
    assert (par.adj1 - par.p0).length() == pytest.approx(o.width)

    left_proj, _ = project(par.p0, Line(segment.start, segment.end - segment.start))
    right_proj, _ = project(par.adj1, Line(segment.start, segment.end - segment.start))
    assert (left_proj-right_proj).length() == pytest.approx(0)


def test_ortho_screen_obstacle_obstructs():
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                            width=1,
                            height=2)

    # pass from center
    assert o.obstructs(Segment(Point(-10, 0, 1), Point(10, 0, 1)))
    assert o.obstructs(Segment(Point(0, -10, 1), Point(0, 10, 1)))
    assert o.obstructs(Segment(Point(-10, -10, 1), Point(10, 10, 1)))

    # pass low
    assert o.obstructs(Segment(Point(-10, 0, 0.1), Point(10, 0, 0.1)))
    # pass high
    assert o.obstructs(Segment(Point(-10, 0, 1.9), Point(10, 0, 1.9)))
    # pass left
    assert o.obstructs(Segment(Point(-10, -0.4, 1), Point(10, -0.4, 1)))
    # pass right
    assert o.obstructs(Segment(Point(-10, 0.4, 1), Point(10, 0.4, 1)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(5, 5, 5), Point(10, 10, 10)))
    assert not o.obstructs(Segment(Point(-10, 0, -1), Point(10, 0, -1)))
    assert not o.obstructs(Segment(Point(-10, 0, 3), Point(10, 0, 3)))
    assert not o.obstructs(Segment(Point(-10, -1, 1), Point(10, -1, 1)))
    assert not o.obstructs(Segment(Point(-10, 1, 1), Point(10, 1, 1)))

    # Unobstructed segment: coming from above
    assert not o.obstructs(Segment(Point(0, 0, 10), Point(0, 0, -10)))

    # Unobstructed ray
    assert not o.obstructs(
        Ray(10e-9, 80, 0, [Point(10, 0, 1), Point(0, 10, 1), Point(-10, 0, 1)]))

    # Unobstructed ray with corner reflection
    assert not o.obstructs(Ray(
        10e-9, 80, 0, [Point(10, 0, 0), Point(0, 10, 0), Point(0, 10, 0), Point(-10, 0, 0)]))

    # Obstructed ray
    assert o.obstructs(
        Ray(10e-9, 80, 0, [Point(10, 0, 0), Point(-10, 0, 0), Point(0, 10, 0)]))

    # Obstructed ray with corner reflection
    assert o.obstructs(Ray(
        10e-9, 80, 0, [Point(10, 0, 0), Point(-10, 0, 0), Point(-10, 0, 0), Point(0, 10, 0)]))

    # Type error
    with pytest.raises(TypeError):
        o.obstructs(0)  # type: ignore


def test_ortho_screen_obstacle_type_error_in_distance():
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                            width=1,
                            height=2)

    # Type error
    with pytest.raises(TypeError):
        o.distance(0)  # type: ignore


@pytest.mark.parametrize("diffraction_model",
                         [atan_diffraction,
                          dke_kunisch
                         ])
def test_ortho_screen_obstacle_diffraction(diffraction_model):
    diffraction_loss_model = functools.partial(diffraction_model, wavelength=1e-9)
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                            width=1,
                            height=2,
                            diffraction_loss_model=diffraction_loss_model,
                            distance_threshold=0)

    # direct LoS obstruction
    r = Segment(Point(-10, 0, 1), Point(10, 0, 1))
    assert o.diffraction_loss(r) > 0

    # far away ray
    r = Segment(Point(-10, 1e100, 1), Point(10, 1e100, 1))
    assert o.diffraction_loss(r) == pytest.approx(0)


def test_ortho_screen_obstacle_vertical_segment():
    o = OrthoScreenObstacle(mm=ConstantPositionMobilityModel(Point(0, 0, 0)),
                            width=0.7,
                            height=1.8,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    segment = Segment(Point(10, 0, 0), Point(10, 0, 1))
    par = o._get_ortho_screen(segment)
    d_expected = 10
    assert distance(segment.start,par.plane)==distance(segment.end,par.plane)==pytest.approx(d_expected)