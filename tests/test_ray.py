# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.ray import Ray
from src.geometry import Point
import pytest
import numpy as np


def test_ray():
    r = Ray(delay=10e-9,
            path_gain=80,
            phase=0,
            vertices=[Point(0, 0, 1.5), Point(10, 0, 1.5)])

    assert r.delay == 10e-9
    assert r.path_gain == 80
    assert r.phase == 0
    assert len(r.vertices) == 2

    # Check that returned list is just a copy
    r.vertices.append(Point(10, 10, 1.5))
    assert len(r.vertices) == 2

    # Correctly set new vertex
    r.vertices = r.vertices + [Point(10, 10, 1.5)]
    assert len(r.vertices) == 3


def test_ray_aoa_aod():
    r = Ray(delay=10e-9,
            path_gain=80,
            phase=0,
            vertices=[Point(0, 0, 1.5), Point(10, 0, 1.5)])

    assert r.aod_azimuth() == pytest.approx(0)
    assert r.aod_elevation() == pytest.approx(0)
    assert r.aod_inclination() == pytest.approx(np.pi / 2)

    assert r.aoa_azimuth() == pytest.approx(np.pi)
    assert r.aoa_elevation() == pytest.approx(0)
    assert r.aoa_inclination() == pytest.approx(np.pi / 2)


def test_refl_order():
    r = Ray(delay=10e-9,
            path_gain=80,
            phase=0,
            vertices=[Point(0, 0, 1.5), Point(10, 0, 1.5)])

    assert r.refl_order() == 0

    r.vertices = r.vertices + [Point(10, 10, 1.5)]
    assert r.refl_order() == 1


def test_is_direct():
    r = Ray(delay=10e-9,
            path_gain=80,
            phase=0,
            vertices=[Point(0, 0, 1.5), Point(10, 0, 1.5)])
    assert r.is_direct()

    r.vertices = r.vertices + [Point(10, 10, 1.5)]
    assert not r.is_direct()
