# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.mobility_model import *
from src.geometry import distance
import pytest


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
    mm = ConstantPositionMobilityModelModel(p)
    assert mm.location(0) == p


# ConstantVelocityMobilityModel
def test_constant_velocity_mobility_model():
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    mm = ConstantVelocityMobilityModel(p, v)
    assert mm.location(0) == p
    assert mm.location(1) == p + v
    assert mm.location(10) == p + 10 * v


# ConstantAccelerationMobilityModel
def test_constant_acceleration_mobility_model():
    p = Point(1, 2, 3)
    v = Vector(1, 0, 0)
    a = Vector(0, 1, 0)
    mm = ConstantAccelerationMobilityModel(p, v, a)
    assert mm.location(0) == p
    assert mm.location(1) == p + v + 0.5 * a
    assert mm.location(10) == p + 10 * v + 0.5 * 10 ** 2 * a


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

    assert type(p) == Point


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
    try:
        mm = WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                                   speeds=[2, 1],
                                   pauses=2)
    except Exception as e:
        assert type(e) == AssertionError

    # Invalid number of pauses
    try:
        mm = WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                                   speeds=2,
                                   pauses=[2, 1])
    except Exception as e:
        assert type(e) == AssertionError


def test_waypoint_mobility_input_lists():
    wp = [Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)]
    mm = WaypointMobilityModel(wp,
                               speeds=[2] * (len(wp) - 1),
                               pauses=[2] * (len(wp) - 1))
