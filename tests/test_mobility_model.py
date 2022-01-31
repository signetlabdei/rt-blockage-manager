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

from src.mobility_model import PositionAllocation, ConstantPositionMobilityModel, ConstantVelocityMobilityModel
from src.mobility_model import ConstantAccelerationMobilityModel, RandomWaypointMobilityModel, WaypointMobilityModel
from src.mobility_model import ConstantAngularPositionMobilityModel, ConstantAngularVelocityMobilityModel
from src.mobility_model import  ConstantAngularAccelerationMobilityModel , get_angular_mobility_model
from src.mobility_model import check_angle_unit_init, check_returned_angle_unit
from src.geometry import Point, Vector, distance, Rectangle
import numpy as np
import pytest
import math


# PositionAllocator
def test_position_allocation():
    rpa = PositionAllocation(x=lambda: 1,
                             y=lambda: 2,
                             z=lambda: 3)

    p = rpa()
    assert p == Point(1, 2, 3)


# ConstantPositionMobilityModel
def test_constant_position_mobility_model():
    p = Point(1, 2, 3)
    mm = ConstantPositionMobilityModel(p)
    assert mm.location(0) == p

    eval(str(mm)) # test repr


# ConstantVelocityMobilityModel
def test_constant_velocity_mobility_model():
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    mm = ConstantVelocityMobilityModel(p, v)
    assert mm.location(0) == p
    assert mm.location(1) == p + v
    assert mm.location(10) == p + 10 * v

    eval(str(mm))  # test repr


# ConstantAccelerationMobilityModel
def test_constant_acceleration_mobility_model():
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    a = Vector(0, 1, 0)
    mm = ConstantAccelerationMobilityModel(p, v, a)
    assert mm.location(0) == p
    assert mm.location(1) == p + v + 0.5 * a
    assert mm.location(10) == p + 10 * v + 0.5 * 10 ** 2 * a

    eval(str(mm))  # test repr


# RandomWaypointMobilityModel
def test_random_waypoint_mobility_model():
    mm = RandomWaypointMobilityModel(bounding_box=Rectangle(0, 0, 10, 10),
                                     position_allocation=PositionAllocation(lambda: np.random.uniform(0, 11),
                                                                            lambda: np.random.uniform(0, 11),
                                                                            lambda: 0),
                                     speed_rv=lambda: np.random.uniform(2, 6),
                                     pause_rv=lambda: np.random.uniform(2, 6))

    # Just check location at 100 consecutive time steps
    for t in range(0, 100, 1):
        p = mm.location(t)
        assert isinstance(p, Point)


# WaypointMobilityModel
def test_waypoint_mobility_model():
    mm = WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                               speeds=2,
                               pauses=2)

    assert mm.max_mobility_duration() == pytest.approx(28)
    assert distance(mm.location(0), Point(0, 0, 0)) == pytest.approx(0)
    assert distance(mm.location(1), Point(2, 0, 0)) == pytest.approx(0)
    assert distance(mm.location(5), Point(10, 0, 0)) == pytest.approx(0)
    assert distance(mm.location(7), Point(10, 0, 0)) == pytest.approx(0)
    assert distance(mm.location(8), Point(10, 2, 0)) == pytest.approx(0)
    assert distance(mm.location(12), Point(10, 10, 0)) == pytest.approx(0)
    assert distance(mm.location(14), Point(10, 10, 0)) == pytest.approx(0)
    assert distance(mm.location(15), Point(8, 10, 0)) == pytest.approx(0)
    assert distance(mm.location(19), Point(0, 10, 0)) == pytest.approx(0)
    assert distance(mm.location(21), Point(0, 10, 0)) == pytest.approx(0)
    assert distance(mm.location(22), Point(0, 8, 0)) == pytest.approx(0)
    assert distance(mm.location(26), Point(0, 0, 0)) == pytest.approx(0)
    assert distance(mm.location(28), Point(0, 0, 0)) == pytest.approx(0)


def test_waypoint_mobility_fails():
    # Invalid number of speeds
    with pytest.raises(AssertionError):
        WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                              speeds=[2, 1],
                              pauses=2)

    # Invalid number of pauses
    with pytest.raises(AssertionError):
        WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                              speeds=2,
                              pauses=[2, 1])


def test_waypoint_mobility_sequence_input():
    # Invalid number of speeds
    WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                          speeds=[2, 2, 2, 2],
                          pauses=[2, 2, 2, 2])

# Angular Mobility Models

# ConstantAngularPositionMobilityModel
def test_constant_angular_position_mobility_model():
    alpha = 45
    time = 10
    alpha_rad = math.radians(alpha)
    amm_deg = ConstantAngularPositionMobilityModel(alpha,'deg')
    amm_rad = ConstantAngularPositionMobilityModel(alpha_rad, 'rad')
    assert (amm_deg.get_angle(time, angle_unit='deg')) == pytest.approx(alpha)
    assert (amm_deg.get_angle(time, angle_unit='rad')) == pytest.approx(alpha_rad)
    assert (amm_rad.get_angle(time, angle_unit='deg')) == pytest.approx(alpha)
    assert (amm_rad.get_angle(time, angle_unit='rad')) == pytest.approx(alpha_rad)
    eval(str(amm_deg))  # test repr
    eval(str(amm_rad))  # test repr


# ConstantAngularVelocityMobilityModel
def test_constant_angular_velocity_mobility_model():
    alpha = 30
    alpha_rad = math.radians(alpha)
    velocity = 2
    velocity_rad = math.radians(velocity)
    time = 3
    amm_deg = ConstantAngularVelocityMobilityModel(alpha, velocity,'deg')
    amm_rad = ConstantAngularVelocityMobilityModel(alpha_rad,velocity_rad,'rad')
    
    expected_value = 36  # deg

    assert (amm_deg.get_angle(time, angle_unit='deg')) == pytest.approx(expected_value)
    assert (amm_deg.get_angle(time, angle_unit='rad')) == pytest.approx(math.radians(expected_value))
    assert (amm_rad.get_angle(time, angle_unit='deg')) == pytest.approx(expected_value)
    assert (amm_rad.get_angle(time, angle_unit='rad')) == pytest.approx(math.radians(expected_value))
    eval(str(amm_deg))  # test repr
    eval(str(amm_rad))  # test repr


# ConstantAngularAccelerationMobilityModel
def test_constant_angular_acceleration_mobility_model():
    alpha = 90
    alpha_rad = math.radians(alpha)
    velocity = 7
    velocity_rad = math.radians(velocity)
    acceleration = -8
    acceleration_rad = math.radians(acceleration)
    time = 4
    amm_deg = ConstantAngularAccelerationMobilityModel(alpha, velocity, acceleration, 'deg')
    amm_rad = ConstantAngularAccelerationMobilityModel(alpha_rad,velocity_rad, acceleration_rad, 'rad')

    expected_value = 54  # deg

    assert (amm_deg.get_angle(time, angle_unit='deg')) == pytest.approx(expected_value)
    assert (amm_deg.get_angle(time, angle_unit='rad')) == pytest.approx(math.radians(expected_value))
    assert (amm_rad.get_angle(time, angle_unit='deg')) == pytest.approx(expected_value)
    assert (amm_rad.get_angle(time, angle_unit='rad')) == pytest.approx(math.radians(expected_value))

    eval(str(amm_deg))  # test repr
    eval(str(amm_rad))  # test repr

# get_azim_angle
def test_get_azim_angle():
    acc_azim = -1
    vel_azim = 2
    azim = 25
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    a = Vector(0, 1, 0)
    cpmm = ConstantPositionMobilityModel(pos=p,angle_azim=azim,velocity_azim=vel_azim,
                                         acceleration_azim=acc_azim,angle_unit='deg')
    
    # at time t = 2s, the azimuth angle is 27 (degree)
    assert (cpmm.get_azim_angle(t=2,angle_unit='deg')) == pytest.approx(27)
    assert (cpmm.get_azim_angle(t=2,angle_unit='rad')) == pytest.approx(math.radians(27))

    # at time t = 4s, the azimuth angle is 25 (degree)
    assert (cpmm.get_azim_angle(t=4,angle_unit='deg')) == pytest.approx(25)
    assert (cpmm.get_azim_angle(t=4,angle_unit='rad')) == pytest.approx(math.radians(25))

    cvmm = ConstantVelocityMobilityModel(start_pos=p,vel=v,angle_azim=azim,velocity_azim=vel_azim,
                                        acceleration_azim=acc_azim,angle_unit='deg')
    
    # at time t = 3s, the azimuth angle is 26.5 (degree)
    assert (cvmm.get_azim_angle(t=3,angle_unit='deg')) == pytest.approx(26.5)
    assert (cvmm.get_azim_angle(t=3,angle_unit='rad')) == pytest.approx(math.radians(26.5))

    # at time t = 6s, the azimuth angle is 19 (degree)
    assert (cvmm.get_azim_angle(t=6,angle_unit='deg')) == pytest.approx(19)
    assert (cvmm.get_azim_angle(t=6,angle_unit='rad')) == pytest.approx(math.radians(19))

    camm = ConstantAccelerationMobilityModel(start_pos=p,vel=v,accel=a,angle_azim=azim,velocity_azim=vel_azim,
                                             acceleration_azim=acc_azim,angle_unit='deg')
    
    # at time t = 5s, the azimuth angle is 22.5 (degree)
    assert (camm.get_azim_angle(t=5,angle_unit='deg')) == pytest.approx(22.5)
    assert (camm.get_azim_angle(t=5,angle_unit='rad')) == pytest.approx(math.radians(22.5))

    # at time t = 8s, the azimuth angle is 9 (degree)
    assert (camm.get_azim_angle(t=8,angle_unit='deg')) == pytest.approx(9)
    assert (camm.get_azim_angle(t=8,angle_unit='rad')) == pytest.approx(math.radians(9))

# get_elev_angle
def test_get_elev_angle():
    acc_elev = -1
    vel_elev = 2
    elev = 25
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    a = Vector(0, 1, 0)
    cpmm = ConstantPositionMobilityModel(pos=p,angle_elev=elev,velocity_elev=vel_elev,
                                         acceleration_elev=acc_elev,angle_unit='deg')
    
    # at time t = 2s, the elevation angle is 27 (degree)
    assert (cpmm.get_elev_angle(t=2,angle_unit='deg')) == pytest.approx(27)
    assert (cpmm.get_elev_angle(t=2,angle_unit='rad')) == pytest.approx(math.radians(27))

    # at time t = 4s, the elevation angle is 25 (degree)
    assert (cpmm.get_elev_angle(t=4,angle_unit='deg')) == pytest.approx(25)
    assert (cpmm.get_elev_angle(t=4,angle_unit='rad')) == pytest.approx(math.radians(25))

    cvmm = ConstantVelocityMobilityModel(start_pos=p,vel=v,angle_elev=elev,velocity_elev=vel_elev,
                                         acceleration_elev=acc_elev,angle_unit='deg')

    # at time t = 3s, the elevation angle is 26.5 (degree)
    assert (cvmm.get_elev_angle(t=3,angle_unit='deg')) == pytest.approx(26.5)
    assert (cvmm.get_elev_angle(t=3,angle_unit='rad')) == pytest.approx(math.radians(26.5))

    # at time t = 6s, the elevation angle is 19 (degree)
    assert (cvmm.get_elev_angle(t=6,angle_unit='deg')) == pytest.approx(19)
    assert (cvmm.get_elev_angle(t=6,angle_unit='rad')) == pytest.approx(math.radians(19))

    camm = ConstantAccelerationMobilityModel(start_pos=p,vel=v,accel=a,angle_elev=elev,velocity_elev=vel_elev,
                                             acceleration_elev=acc_elev,angle_unit='deg')
    
    # at time t = 5s, the elevation angle is 22.5 (degree)
    assert (camm.get_elev_angle(t=5,angle_unit='deg')) == pytest.approx(22.5)
    assert (camm.get_elev_angle(t=5,angle_unit='rad')) == pytest.approx(math.radians(22.5))

    # at time t = 8s, the elevation angle is 9 (degree)
    assert (camm.get_elev_angle(t=8,angle_unit='deg')) == pytest.approx(9)
    assert (camm.get_elev_angle(t=8,angle_unit='rad')) == pytest.approx(math.radians(9))

# get_angular_mobility_model
def test_get_angular_mobility_model():
    angle = 10
    velocity = 5
    acceleration = -1

    angular_mm = get_angular_mobility_model(acceleration = 0, velocity = 0,
                                            angle = angle,angle_unit='deg')
    assert isinstance(angular_mm,ConstantAngularPositionMobilityModel)

    assert (angular_mm.get_angle(t = 1,angle_unit = 'deg')) == pytest.approx(angle)

    angular_mm = get_angular_mobility_model(acceleration = 0, velocity = velocity,
                                            angle = angle, angle_unit='deg')

    assert isinstance(angular_mm,ConstantAngularVelocityMobilityModel)

    #after 1 second, angle is 15 (degree)
    assert (angular_mm.get_angle(t = 1,angle_unit='deg')) == pytest.approx(15)

    angular_mm = get_angular_mobility_model(acceleration = acceleration, velocity = velocity,
                                            angle = angle, angle_unit = 'deg')

    assert isinstance(angular_mm,ConstantAngularAccelerationMobilityModel)

    #after 1 second, angle is 14.5 (degree)
    assert (angular_mm.get_angle(t = 1,angle_unit='deg')) == pytest.approx(14.5)

    angular_mm = get_angular_mobility_model(acceleration = 0, velocity = 0,
                                            angle = 0, angle_unit = 'deg')
    
    assert angular_mm is None

#check_angle_unit
def test_check_angle_unit():
    angle = 30

    assert check_angle_unit_init(angle,'rad') == pytest.approx(angle)
    assert check_angle_unit_init(angle,'deg') == pytest.approx(math.radians(angle))

    assert check_returned_angle_unit(angle,'rad') == pytest.approx(angle)
    assert check_returned_angle_unit(angle,'deg') == pytest.approx(math.degrees(angle))
