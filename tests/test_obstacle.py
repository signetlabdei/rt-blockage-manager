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
from sqlite3 import SQLITE_ANALYZE

from numpy import angle
from src.diffraction_models import atan_diffraction, dke_kunisch
from src.obstacle import ScreenObstacle, OrthoScreenObstacle, SphereObstacle
from src.geometry import Parallelogram3d, TransfMatrix, Vector, Line, Point, Segment, distance, project

from src.mobility_model import ConstantPositionMobilityModel
from src.ray import Ray
import numpy as np
import pytest
import math

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
                            distance_threshold=math.inf)

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

# ScreenObstacle
def test_screen_obstacle():
    pos = Point(0, 0, 0)
    elev_angle = 0
    azim_angle = 0
    width = 0.7
    height = 1.8
    reflection_loss = 5
    transmission_loss = 30


    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle)

    o = ScreenObstacle(mm=mobility_model,
                            width=width,
                            height=height,
                            reflection_loss=reflection_loss,
                            transmission_loss=transmission_loss,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    assert o.location() == pos
    assert o.elevation_angle() == elev_angle
    assert o.azimuth_angle() == azim_angle
    assert o.width == width
    assert o.height == height
    assert o.reflection_loss() == reflection_loss
    assert o.transmission_loss() == transmission_loss
    assert o._diffraction_loss_model is None
    assert o._distance_threshold == 0

    # check update of mobility model and angular mobility model
    o.update(1)
    assert o.location() == Point(0, 0, 0)
    assert o.elevation_angle() == 0
    assert o.azimuth_angle() == 0


@pytest.mark.parametrize("elev_angle",[0,45,90,180])
@pytest.mark.parametrize("azim_angle",[0,45,90,180])
def test_tilted_screen_obstacle_preserves_width_height(elev_angle, azim_angle):

    pos = Point(0, 0, 0)

    mobility_model = ConstantPositionMobilityModel(pos,angle_elev = elev_angle, angle_azim = azim_angle,angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model,
                            width=0.7,
                            height=1.8,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    par = o.get_screen()

    assert (par.adj2 - par.p0).length() == pytest.approx(o.height)
    assert (par.adj1 - par.p0).length() == pytest.approx(o.width)

@pytest.mark.parametrize("time", [9,18,27,36,45,54,63,72])
@pytest.mark.parametrize("cpmm",[ConstantPositionMobilityModel(pos=Point(0, 0, 0),angle_elev=0,velocity_elev=10,angle_unit='deg'),
                                 ConstantPositionMobilityModel(pos=Point(0, 0, 0),angle_azim=0,velocity_azim=10,angle_unit='deg')])
def test_tilted_screen_obstacle_time(time,cpmm):

    o = ScreenObstacle(mm=cpmm,
                            width=0.7,
                            height=1.8,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    o.update(t=time)
    par = o.get_screen()

    if cpmm.get_azim_mm() is None:
        if time in (9,27,45,63):
            # adj2 = Point(-0.9, -0.35, 0.9), adj1 = Point(0.9, 0.35, 0.9), p0 = Point(0.9, -0.35, 0.9)
            adj2 = Point(-0.9, -0.35, 0.9)
            adj1 = Point(0.9, 0.35, 0.9)
            p0 = Point(0.9, -0.35, 0.9)
        else:
            # adj2 = Point(0.0, -0.35, 1.8), adj1 = Point(0.0, 0.35, 0.0), p0 = Point(0.0, -0.35, 0.0)
            adj2 =  Point(0.0, -0.35, 1.8)
            adj1 = Point(0.0, 0.35, 0.0)
            p0 = Point(0.0, -0.35, 0.0)
            
    elif cpmm.get_elev_mm() is None:
        if time in (9,27,45,63):
            # adj2 = Point(0.35, 0.0, 1.8), adj1 = Point(-0.35, 0.0, 0.0), p0 = Point(0.35, 0.0, 0.0)
            adj2 = Point(0.35, 0.0, 1.8)
            adj1 = Point(-0.35, 0.0, 0.0)
            p0 = Point(0.35, 0.0, 0.0)
        else:
            # adj2 = Point(0.0, -0.35, 1.8), adj1 = Point(0.0, 0.35, 0.0), p0 = Point(0.0, -0.35, 0.0)
            adj2 = Point(0.0, -0.35, 1.8)
            adj1 = Point(0.0, 0.35, 0.0)
            p0 = Point(0.0, -0.35, 0.0)
    
    assert par == Parallelogram3d(p0,adj1,adj2)

@pytest.mark.parametrize("elev", [-450,-360,-270,-180,-90,-30,0,30,90,180,270,360,450])
@pytest.mark.parametrize("azim", [-450,-360,-270,-180,-90,-30,0,30,90,180,270,360,450])
def test_rotation_screen_elevation_and_azimuth(elev,azim):

    cpmm = ConstantPositionMobilityModel(pos=Point(0, 0, 0),angle_elev=elev,
                                         angle_azim=azim,angle_unit='deg')

    height = 1.8
    width = 0.7

    o = ScreenObstacle(mm=cpmm,
                            width=width,
                            height=height,
                            reflection_loss=5,
                            transmission_loss=30,
                            diffraction_loss_model=None,
                            distance_threshold=0)

    par = o.get_screen()

    bottom_center = Point(0,0,0)
    z_vector = Vector(0,0,1)
    base_vector = Vector(0,1,0)
    orth_vector  = base_vector.cross(z_vector)

    # find the center of the obstacle (unique stationary point after azimuth and elevation tilts)
    center = bottom_center + z_vector * height / 2

    # to keep the consistency of the azimuth rotation
    if elev in (-450,-180,-90,180,270):
        azim = -azim

    # elevation tilt

    elev = math.radians(elev)

    if abs(math.sin(elev)) < 1e-9:
        length_vector = z_vector
        width_vector = base_vector
        orto = orth_vector.normalize()
    elif abs(math.cos(elev)) < 1e-9:
        length_vector = -orth_vector.normalize()
        orto = z_vector
        width_vector = length_vector.cross(orto).normalize()
    else:
        # the elevation angle is not 0/90/180/270/360/450....
        shift = z_vector.cross(base_vector).normalize()
        shift_orto = shift * math.tan(elev - math.pi/2)
        orto = (z_vector + shift_orto).normalize()
        shift = shift * math.tan(elev)
        length_vector = (z_vector + shift).normalize()
        width_vector = length_vector.cross(orto).normalize()

    # azimuth tilt

    azim = math.radians(azim)

    if abs(math.cos(azim)) < 1e-9:
        width_vector, orto = -orto, width_vector
        length_vector = orto.cross(width_vector).normalize()
    elif abs(math.sin(azim)) > 1e-9:
        # the azimuth angle is not 0/90/180/270/360/450....
        shift = orto * math.tan(azim)
        orto = orto * math.tan(azim-math.pi/2)
        width_vector = (width_vector + shift).normalize()
        length_vector = orto.cross(width_vector).normalize()

    # obtain the points of the parallelogram

    bottom_right = center + width / 2 * width_vector - height / 2 * length_vector
    bottom_left  = bottom_right - width * width_vector
    top_left = bottom_left + height * length_vector

    assert par == Parallelogram3d(bottom_left,bottom_right,top_left)   


def test_screen_obstacle_no_tilted_obstructs():

    pos = Point(0, 0, 0)
    elev_angle = 0
    azim_angle = 0

    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle,angle_unit = 'deg')


    o = ScreenObstacle(mm=mobility_model, width=1,height=2)

    # pass from center
    assert o.obstructs(Segment(Point(-10, 0, 1), Point(10, 0, 1)))
    assert o.obstructs(Segment(Point(-10, -10, 1), Point(10, 10, 1)))
    assert o.obstructs(Segment(Point(10, 0, 11), Point(-1, 0, 0)))


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

def test_screen_obstacle_elev_tilted_obstructs():
    # with only elevation angle = 90

    pos = Point(0, 0, 0)
    elev_angle = 90
    azim_angle = 0

    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle, angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model, width=1,height=2)
    
    # pass from center
    assert o.obstructs(Segment(Point(0, 0, -10), Point(0, 0, 10)))
    assert o.obstructs(Segment(Point(-1, 0, 0), Point(5, 0, 4)))
    assert o.obstructs(Segment(Point(0, -1, 0), Point(0, 5, 4)))
    
    # pass low
    assert o.obstructs(Segment(Point(0.9, 0, -10), Point(0.9, 0, 10)))
    # pass high
    assert o.obstructs(Segment(Point(-0.9, 0, -10), Point(-0.9, 0, 10)))
    # pass left
    assert o.obstructs(Segment(Point(0, -0.4, -10), Point(0, -0.4, 10)))
    # pass right
    assert o.obstructs(Segment(Point(0, 0.4, -10), Point(0, 0.4, 10)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(-10, 0, 0.1), Point(10, 0, 0.1)))
    assert not o.obstructs(Segment(Point(-10, 0, 1.9), Point(10, 0, 1.9)))

def test_screen_obstacle_azim_tilted_obstructs():
    # with only azimuth angle = 90

    pos = Point(0, 0, 0)
    elev_angle = 0
    azim_angle = 90

    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle,angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model, width=1,height=2)
    
    # pass from center
    assert o.obstructs(Segment(Point(0, -10, 1), Point(0, 10, 1)))
    assert o.obstructs(Segment(Point(-5, -5, 1), Point(5, 5, 1)))
    assert o.obstructs(Segment(Point(0, 10, 6), Point(0, -6, -2)))
    
    # pass low
    assert o.obstructs(Segment(Point(0, -10, 0.1), Point(0, 10, 0.1)))

    # pass high
    assert o.obstructs(Segment(Point(0, -10, 1.9), Point(0, 10, 1.9)))

    # pass left
    assert o.obstructs(Segment(Point(-0.4, -10, 1), Point(-0.4, 10, 1)))

    # pass right
    assert o.obstructs(Segment(Point(0.4, -10, 1), Point(0.4, 10, 1)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(0.6, -10, 1), Point(0.6, 10, 1)))
    assert not o.obstructs(Segment(Point(-0.6, -10, 1), Point(-0.6, 10, 1)))

def test_screen_obstacle_azim_elev_tilted_obstructs():
    # with both inclination
    pos = Point(0, 0, 0)
    elev_angle = 90
    azim_angle = 90

    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle, angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model, width=1,height=2)
    
    # pass from center
    assert o.obstructs(Segment(Point(0, -10, 1), Point(0, 10, 1)))
    assert o.obstructs(Segment(Point(-5, -5, 1), Point(5, 5, 1)))
    assert o.obstructs(Segment(Point(0, 10, 6), Point(0, -6, -2)))
    
    # pass low
    assert o.obstructs(Segment(Point(0, -10, 0.6), Point(0, 10, 0.6)))

    # pass high
    assert o.obstructs(Segment(Point(0, -10, 1.4), Point(0, 10, 1.4)))

    # pass left
    assert o.obstructs(Segment(Point(-0.9, -10, 1), Point(-0.9, 10, 1)))

    # pass right
    assert o.obstructs(Segment(Point(0.9, -10, 1), Point(0.9, 10, 1)))

    # Unobstructed segment
    assert not o.obstructs(Segment(Point(0, -10, 1.6), Point(0, 10, 1.6)))
    assert not o.obstructs(Segment(Point(-2, -10, 1), Point(-2, 10, 1)))
    assert not o.obstructs(Segment(Point(0, -10, 0.1), Point(0, 10, 0.1)))


def test_screen_different_initial_orientation_obstructs():
    #base along x and height along y
    pos = Point(0, 0, 0)
    base_vector = Vector(1,0,0)
    height_vector = Vector(0,1,0)

    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = 0, angle_azim = 0, angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model,
                       width=1,
                       height=2,
                       base_vector = base_vector,
                       height_vector = height_vector)
    
    #Obstructed segments
    assert o.obstructs(Segment(Point(0,1,-10),Point(0,1,10)))
    assert o.obstructs(Segment(Point(0,1.9,-10),Point(0,1.9,10)))
    assert o.obstructs(Segment(Point(0,0.1,-10),Point(0,0.1,10)))
    assert o.obstructs(Segment(Point(-0.4,1,-10),Point(-0.4,1,10)))
    assert o.obstructs(Segment(Point(0.4,1,-10),Point(0.4,1,10)))
    assert o.obstructs(Segment(Point(-0.4,1.9,-10),Point(-0.4,1.9,10)))
    assert o.obstructs(Segment(Point(-0.4,0.1,-10),Point(-0.4,0.1,10)))
    assert o.obstructs(Segment(Point(0.4,1.9,-10),Point(0.4,1.9,10)))
    assert o.obstructs(Segment(Point(0.4,0.1,-10),Point(0.4,0.1,10)))

    #Unobstructed segments
    assert not o.obstructs(Segment(Point(0.4,1.9,-10),Point(0.4,1.9,-1)))
    assert not o.obstructs(Segment(Point(0.4,0.1,1),Point(0.4,0.1,10)))
    assert not o.obstructs(Segment(Point(0.6,0.1,1),Point(0.6,0.1,10)))
    assert not o.obstructs(Segment(Point(0.4,2.1,1),Point(0.4,2.1,10)))

    # Type error
    with pytest.raises(TypeError):
        o.obstructs(0)  # type: ignore


def test_screen_obstacle_type_error_in_distance():
    
    pos = Point(0, 0, 0)
    elev_angle = 0
    azim_angle = 0
    
    mobility_model = ConstantPositionMobilityModel(pos, angle_elev = elev_angle, angle_azim = azim_angle)

    o = ScreenObstacle(mm=mobility_model, width=1,height=2)

    # Type error
    with pytest.raises(TypeError):
        o.distance(0)  # type: ignore

@pytest.mark.parametrize("diffraction_model",
                         [atan_diffraction,
                          dke_kunisch
                         ])                             
def test_tilted_screen_obstacle_diffraction(diffraction_model):
    diffraction_loss_model = functools.partial(diffraction_model, wavelength=1e-9)

    mobility_model = ConstantPositionMobilityModel(Point(0, 0, 0), angle_elev = 10, angle_azim = 10,angle_unit='deg')

    o = ScreenObstacle(mm=mobility_model,
                            width=1,
                            height=2,
                            diffraction_loss_model=diffraction_loss_model,
                            distance_threshold=math.inf)

    # direct LoS obstruction
    r = Segment(Point(-10, 0, 1), Point(10, 0, 1))
    assert o.diffraction_loss(r) > 0

    # far away ray
    r = Segment(Point(-10, 1e100, 1), Point(10, 1e100, 1))
    assert o.diffraction_loss(r) == pytest.approx(0)