# AUTHOR(S):
# Paolo Testolina <paolo.testolina@dei.unipd.it>
# Alessandro Traspadini <alessandro.traspadini@dei.unipd.it>
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: August 2021

import functools
import math
import numpy as np
import pytest
from scipy.special import fresnel

from src.diffraction_models import atan_diffraction, dke_itu, fast_fresnel_integral, ske_itu, dke_kunisch, ske_kunisch, lateral_atan_diffraction, empirical_itu
from src.geometry import Parallelogram3d, Point, Segment, Vector

DEEP_SHADOW_MIN_LOSS = 100 # dB


def test_fast_fresnel_integral():
    nu = np.arange(-10, 10, 0.1)

    S, C = fresnel(nu)
    for x, S_ref, C_ref in zip(nu, S, C):
        F = fast_fresnel_integral(x)
        assert C_ref == pytest.approx(F.real, rel=1e-5)
        assert S_ref == pytest.approx(F.imag, rel=1e-5)


@pytest.mark.parametrize("segment",
                         [(Segment(Point(1e100, -10, 1), Point(1e100, 10, 1))),  # right
                          (Segment(Point(-1e100, -10, 1), Point(-1e100, 10, 1))),  # left
                          (Segment(Point(0, -10, 1e100), Point(0, 10, 1e100))),  # above
                          (Segment(Point(0, -10, -1e100), Point(0, 10, -1e100))),  # below
                          (Segment(Point(1e100, -10, 1e100), Point(1e100, 10, 1e100))),  # above right
                          ])
def test_atan_diffraction_very_far(segment:Segment):
    bottom_right = Point(1, 0, 0)
    bottom_left = Point(-1, 0, 0)
    top_left = Point(-1, 0, 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction
    d = atan_diffraction(screen=screen, seg=segment, wavelength=wavelength)
    assert d == pytest.approx(0)


def test_atan_diffraction_edge2close():
# bugfix: segments (rays) very close to the edge of the screen may rise value error when computing math.sqrt()
    wavelength = 0.004996540966666667
    p1 = Point(9.2, 2.5999999999999996, 2.032826530612245)
    p2 = Point(9.2, 3.0, 2.032826530612245)
    seg = Segment(Point(0.2, 3.0, 2.5), Point(10.0, 3.0, 1.9913))

    try:
        _ = lateral_atan_diffraction(p1, p2, seg, wavelength)
    except ValueError:
        assert False
    

@pytest.mark.parametrize("edge_dir,segment",
                         [(Vector(-1, 0, 0), Segment(Point(1e100, -10, 1), Point(1e100, 10, 1))),  # right
                          (Vector(1, 0, 0), Segment(Point(-1e100, -10, 1), Point(-1e100, 10, 1))),  # left
                          (Vector(0, 0, -1), Segment(Point(0, -10, 1e100), Point(0, 10, 1e100))),  # above
                          (Vector(0, 0, 1), Segment(Point(0, -10, -1e100), Point(0, 10, -1e100))),  # below
                          (Vector(0, 0, 1), Segment(Point(5, -10, -1e100), Point(5, 10, -1e100))),  # below offset
                          ])
def test_ske_itu(edge_dir:Vector, segment:Segment):
    # If ray is very far off the screen it should not create any significant amount of diffraction
    d = ske_itu(edge_pos=Point(0, 0, 0),
                edge_dir=edge_dir,
                seg=segment,
                wavelength=1e-9)
    assert d == pytest.approx(0)

    # If ray is very far over the screen it should not create a deep shadow
    d = ske_itu(edge_pos=Point(0, 0, 0),
                edge_dir=-edge_dir,
                seg=segment,
                wavelength=1e-9)
    assert d > DEEP_SHADOW_MIN_LOSS


@pytest.mark.parametrize("segment",
                         [(Segment(Point(1.1, -10, 1), Point(1.1, 10, 1))),  # right
                          (Segment(Point(-1.1, -10, 1), Point(-1.1, 10, 1))),  # left
                          (Segment(Point(0, -10, 2.1), Point(0, 10, 2.1))),  # above
                          (Segment(Point(0, -10, -.1), Point(0, 10, -.1))),  # below
                          # above right
                          (Segment(Point(1.1, -10, 2.1), Point(1.1, 10, 2.1))),
                          ])
def test_dke_itu_no_shadow(segment:Segment):
    bottom_right = Point(1, 0, 0)
    bottom_left = Point(-1, 0, 0)
    top_left = Point(-1, 0, 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction
    with pytest.raises(RuntimeError):
        dke_itu(screen=screen,
                seg=segment,
                wavelength=wavelength,
                model='something_wrong')


def test_dke_itu_deep_shadow():
    screen_size = 1e10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction.
    segment = Segment(Point(0, -10, 0), Point(0, 10, 0))
    d = dke_itu(screen=screen,
                seg=segment,
                wavelength=wavelength,
                model='avg')
    assert d > DEEP_SHADOW_MIN_LOSS

    d = dke_itu(screen=screen,
                seg=segment,
                wavelength=wavelength,
                model='min')
    assert d > DEEP_SHADOW_MIN_LOSS

    with pytest.raises(ValueError):
        dke_itu(screen=screen,
                seg=segment,
                wavelength=wavelength,
                model='something_wrong')


@pytest.mark.parametrize("edge_dir,segment",
                         [(Vector(-1, 0, 0), Segment(Point(1e100, -10, 1), Point(1e100, 10, 1))),  # right
                          (Vector(-1, 0, 0), Segment(Point(1e20, -10, 1), Point(1e20, 10, 1))),  # right
                          (Vector(1, 0, 0), Segment(Point(-1e100, -10, 1), Point(-1e100, 10, 1))),  # left
                          (Vector(0, 0, -1), Segment(Point(0, -10, 1e100), Point(0, 10, 1e100))),  # above
                          (Vector(0, 0, 1), Segment(Point(0, -10, -1e100), Point(0, 10, -1e100))),  # below
                          (Vector(0, 0, 1), Segment(Point(5, -10, -1e100), Point(5, 10, -1e100))),  # below offset
                          ])
def test_ske_kunisch(edge_dir:Vector, segment:Segment):
    # If ray is very far off the screen it should not create any significant amount of diffraction
    d = ske_kunisch(edge_pos=Point(0, 0, 0),
                   edge_dir=edge_dir,
                   seg=segment,
                   wavelength=1e-9, unit='db')
    assert d == pytest.approx(0)

    # If ray is very far over the screen it should not create a deep shadow
    d = ske_kunisch(edge_pos=Point(0, 0, 0),
                   edge_dir=-edge_dir,
                   seg=segment,
                   wavelength=1e-9, unit='db')
    assert d > DEEP_SHADOW_MIN_LOSS


@pytest.mark.parametrize("segment",
                         [(Segment(Point(1e100, -10, 1), Point(1e100, 10, 1))),  # right
                          (Segment(Point(-1e100, -10, 1), Point(-1e100, 10, 1))),  # left
                          pytest.param(Segment(Point(0, -10, 1e100), Point(0, 10, 1e100)), marks=pytest.mark.xfail),  # above
                          pytest.param(Segment(Point(0, -10, -1e100), Point(0, 10, -1e100)), marks=pytest.mark.xfail), # below
                          (Segment(Point(1e100, -10, 1e100), Point(1e100, 10, 1e100))),  # above right
                          ])
def test_dke_kunisch_very_far(segment:Segment):
    bottom_right = Point(1, 0, 0)
    bottom_left = Point(-1, 0, 0)
    top_left = Point(-1, 0, 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction
    d = dke_kunisch(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d == pytest.approx(0)


@pytest.mark.parametrize("segment",
                         [(Segment(Point(1e100, -10, 1), Point(1e100, 10, 1))),  # right
                          (Segment(Point(-1e100, -10, 1), Point(-1e100, 10, 1))),  # left
                          (Segment(Point(0, -10, 1e100), Point(0, 10, 1e100))),  # above
                          (Segment(Point(0, -10, -1e100), Point(0, 10, -1e100))), # below
                          (Segment(Point(1e100, -10, 1e100), Point(1e100, 10, 1e100))),  # above right
                          ])
@pytest.mark.parametrize("diffraction_model",
                         [atan_diffraction,
                         empirical_itu])
def test_dke_very_far(segment:Segment, diffraction_model):
    bottom_right = Point(1, 0, 0)
    bottom_left = Point(-1, 0, 0)
    top_left = Point(-1, 0, 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d == pytest.approx(0)


@pytest.mark.parametrize("diffraction_model",
                         [atan_diffraction,
                         atan_diffraction,
                         empirical_itu,
                         dke_kunisch])
def test_dke_deep_shadow(diffraction_model):
    screen_size = 1e10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    # If ray is very far off the screen it should not create any significant amount of diffraction.
    segment = Segment(Point(0, -10, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d > DEEP_SHADOW_MIN_LOSS

    d = dke_kunisch(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d > DEEP_SHADOW_MIN_LOSS


@pytest.mark.parametrize("diffraction_model",
                         [atan_diffraction,
                         dke_kunisch])
def test_diffraction_close2node(diffraction_model):
    screen_size = 10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    segment = Segment(Point(0, -1e-12, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d > DEEP_SHADOW_MIN_LOSS

    segment = Segment(Point(0, 1e-12, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d==pytest.approx(0)

    segment = Segment(Point(0, 0, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d > DEEP_SHADOW_MIN_LOSS


def test_empirical_itu_diffraction_close2node():
    diffraction_model=empirical_itu
    screen_size = 10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    segment = Segment(Point(0, -1e-12, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                seg=segment,
                wavelength=wavelength)
    assert d > DEEP_SHADOW_MIN_LOSS

    segment = Segment(Point(0, 1e-12, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                   seg=segment,
                   wavelength=wavelength)
    assert d==pytest.approx(0)

    segment = Segment(Point(0, 0, 0), Point(0, 10, 0))
    d = diffraction_model(screen=screen,
                seg=segment,
                wavelength=wavelength)
    assert d == pytest.approx(-20*np.log10(0.5))


def test_empirical_itu_diffraction_edge_on_node():
    diffraction_model = empirical_itu
    screen_size = 10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    segment = Segment(Point(top_left.x, 0.0, 0.0), Point(top_left.x, 10, 0.0))

    d = diffraction_model(screen=screen,
                seg=segment,
                wavelength=wavelength)
    assert d == pytest.approx(-20*np.log10(0.5))


def test_atan_itu_diffraction_edge_on_node():
    screen_size = 10
    bottom_right = Point(screen_size / 2, 0, -screen_size / 2)
    bottom_left = Point(-screen_size / 2, 0, -screen_size / 2)
    top_left = Point(-screen_size / 2, 0, screen_size / 2)
    screen = Parallelogram3d(bottom_left, bottom_right, top_left)
    wavelength = 1e-9

    segment = Segment(Point(top_left.x, 0.0, 0.0), Point(top_left.x, 10, 0.0))
    d = atan_diffraction(screen=screen,
                   seg=segment,
                   wavelength=wavelength)

    assert not math.isnan(d)
    assert 0 <= d < math.inf

@pytest.mark.parametrize("ske_diffraction_model",
                         [functools.partial(ske_kunisch,unit='db'),
                         ske_itu])
def test_ske_diffraction_edge_on_node(ske_diffraction_model):
    segment = Segment(Point(0,0,0), Point(1,0,0))
    edge_pos = Point(0,0,0)
    edge_dir = Vector(0,0,-1)
    d = ske_diffraction_model(edge_pos=edge_pos,
                   edge_dir=edge_dir,
                   seg=segment,
                   wavelength=1e-9)

    assert d == pytest.approx(-20 * math.log10(0.5))
