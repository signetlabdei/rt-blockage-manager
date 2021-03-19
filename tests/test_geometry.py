# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.geometry import GeometryArithmeticError, Vector, Point, Segment, Line, Rectangle, distance, project
import pytest
import numpy as np
import math


# Vector
def test_vector():
    v = Vector(1, 2, 3)
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0

    # check repr string
    v_repr = eval(str(v))
    assert v == v_repr


def test_vector_from_array():
    coords = [1, 2, 3]
    v = Vector.from_array(coords)
    assert v.x == coords[0]
    assert v.y == coords[1]
    assert v.z == coords[2]

    v = Vector.from_array(np.array(coords))
    assert v.x == coords[0]
    assert v.y == coords[1]
    assert v.z == coords[2]


def test_vector_normalize():
    v = Vector(1, 2, 3)
    vn = v.normalize()

    assert vn.length() == pytest.approx(1)
    # vn must be parallel to v: dot product equal to product of lengths
    assert v.dot(vn) == pytest.approx(v.length() * vn.length())


@pytest.mark.parametrize("v,length", [(Vector(0, 0, 0), 0),
                                      (Vector(1, 0, 0), 1),
                                      (Vector(1, 1, 1), np.sqrt(3))])
def test_vector_length(v, length):
    assert v.length() == pytest.approx(length)


@pytest.mark.parametrize("v1,v2,res", [(Vector(1, 0, 0), Vector(0, 0, 0), 0),
                                       (Vector(1, 0, 0), Vector(1, 0, 0), 1),
                                       (Vector(1, 0, 0), Vector(0, 1, 0), 0),
                                       (Vector(1, 0, 0), Vector(0, 0, 1), 0),
                                       (Vector(1, 2, 3), Vector(1, 0, 0), 1),
                                       (Vector(1, 2, 3), Vector(0, 1, 0), 2),
                                       (Vector(1, 2, 3), Vector(0, 0, 1), 3),
                                       (Vector(1, 2, 3), Vector(1, 2, 3), 14)])
def test_dot(v1, v2, res):
    assert v1.dot(v2) == pytest.approx(res)


def test_vector_add():
    # Vector + Vector
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)

    v = v1 + v2
    v_expected = Vector(5, 7, 9)
    assert (v - v_expected).length() == pytest.approx(0)

    # Vector + Point
    v = Vector(1, 2, 3)
    p = Point(4, 5, 6)

    p_sum = v + p
    assert type(p_sum) == Point

    p_expected = Point(5, 7, 9)
    assert (p_sum - p_expected).length() == pytest.approx(0)

    # Vector + anything: error
    with pytest.raises(GeometryArithmeticError):
        v + 1  # type: ignore


def test_vector_sub():
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)

    v = v1 - v2
    v_expected = Vector(-3, -3, -3)
    assert (v - v_expected).length() == pytest.approx(0)

    # Vector - anything: error
    with pytest.raises(GeometryArithmeticError):
        v - 1  # type: ignore


def test_vector_mul():
    v1 = Vector(1, 2, 3)
    a = 10

    v = v1 * a
    v_expected = Vector(10, 20, 30)
    assert (v - v_expected).length() == pytest.approx(0)

    # Vector * Vector: error
    with pytest.raises(GeometryArithmeticError):
        v * v  # type: ignore


def test_vector_rmul():
    v1 = Vector(1, 2, 3)
    a = 10

    v = a * v1
    v_expected = Vector(10, 20, 30)
    assert (v - v_expected).length() == pytest.approx(0)


def test_vector_truediv():
    v1 = Vector(1, 2, 3)
    a = 10

    v = v1 / a
    v_expected = Vector(0.1, 0.2, 0.3)
    assert (v - v_expected).length() == pytest.approx(0)


def test_vector_eq():
    # Vector - Vector
    v1 = Vector(1, 2, 3)
    v2 = Vector(1, 2, 3)

    assert v1 == v2

    # Vector == Point: error
    with pytest.raises(GeometryArithmeticError):
        v1 == Point(1, 2, 3)


@pytest.mark.parametrize("v,az,el,incl", [(Vector(1, 0, 0), 0, 0, math.pi / 2),
                                          (Vector(-1, 0, 0), math.pi, 0, math.pi / 2),
                                          (Vector(0, 1, 0), math.pi / 2, 0, math.pi / 2),
                                          (Vector(0, -1, 0), -math.pi / 2, 0, math.pi / 2),
                                          (Vector(0, 0, 1), 0, math.pi / 2, 0),
                                          (Vector(0, 0, -1), 0, -math.pi / 2, math.pi)])
def test_vector_az_el_incl(v, az, el, incl):
    assert v.azimuth() == pytest.approx(az)
    assert v.elevation() == pytest.approx(el)
    assert v.inclination() == pytest.approx(incl)


# Point
def test_point():
    p = Point(1, 2, 3)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.z == 3.0

    # check repr string
    p_repr = eval(str(p))
    assert p == p_repr


def test_point_from_array():
    coords = [1, 2, 3]
    p = Point.from_array(coords)
    assert p.x == coords[0]
    assert p.y == coords[1]
    assert p.z == coords[2]
    p = Point.from_array(np.array(coords))
    assert p.x == coords[0]
    assert p.y == coords[1]
    assert p.z == coords[2]


def test_point_sub():
    p1 = Point(1, 2, 3)
    p2 = Point(0, 0, 0)

    v = p1 - p2
    v_expected = Vector(1, 2, 3)
    assert (v - v_expected).length() == pytest.approx(0)

    # Point - anything: error
    with pytest.raises(GeometryArithmeticError):
        p1 - 1  # type: ignore


def test_point_add():
    p = Point(0, 0, 0)
    v = Vector(1, 2, 3)

    p2 = p + v
    assert type(p2) == Point

    p_expected = Point(1, 2, 3)
    assert (p2 - p_expected).length() == pytest.approx(0)

    # Point + Point: error
    with pytest.raises(GeometryArithmeticError):
        p + p  # type: ignore


def test_point_eq():
    p1 = Point(1, 2, 3)
    p2 = Point(1, 2, 3)

    assert p1 == p2

    # Point == Vector: error
    with pytest.raises(GeometryArithmeticError):
        p1 == Vector(1, 2, 3)


# Segment
def test_segment():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 3)
    s = Segment(p1, p2)

    assert p1 == s.start
    assert p2 == s.end

    # check repr string
    s_repr = eval(str(s))
    assert s == s_repr


def test_segment_length():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 3)
    s = Segment(p1, p2)

    assert s.length() == pytest.approx(np.sqrt(14))


def test_segment_eq():
    p1 = Point(1, 2, 3)
    p2 = Point(1, 2, 3)

    s1 = Segment(p1, p2)
    s2 = Segment(p1, p2)

    assert s1 == s2

    # Segment == anything: error
    with pytest.raises(GeometryArithmeticError):
        s1 == p1


@pytest.mark.parametrize("s,aod_az,aod_el,aod_incl,aoa_az,aoa_el,aoa_incl",
                         [(Segment(Point(0, 0, 0), Point(1, 0, 0)), 0, 0, math.pi / 2, math.pi, 0, math.pi / 2),
                          (Segment(Point(0, 0, 0), Point(0, 1, 0)), math.pi / 2, 0, math.pi / 2, -math.pi / 2, 0, math.pi / 2),
                          (Segment(Point(0, 0, 0), Point(0, 0, 1)), 0, math.pi / 2, 0, 0, -math.pi / 2, math.pi)])
def test_segment_aoa_aod(s, aod_az, aod_el, aod_incl, aoa_az, aoa_el, aoa_incl):
    assert s.aod_azimuth() == pytest.approx(aod_az)
    assert s.aod_elevation() == pytest.approx(aod_el)
    assert s.aod_inclination() == pytest.approx(aod_incl)

    assert s.aoa_azimuth() == pytest.approx(aoa_az)
    assert s.aoa_elevation() == pytest.approx(aoa_el)
    assert s.aoa_inclination() == pytest.approx(aoa_incl)


# Line
def test_line():
    p = Point(0, 0, 0)
    v = Vector(1, 2, 3)
    ll = Line(p, v)

    assert p == ll.p
    assert v == ll.v

    # check repr string
    ll_repr = eval(str(ll))
    assert ll == ll_repr


def test_line_eq():
    p = Point(0, 0, 0)
    v = Vector(1, 2, 3)

    l1 = Line(p, v)
    l2 = Line(p, v)

    assert l1 == l2

    # Line == anything: error
    with pytest.raises(GeometryArithmeticError):
        l1 == v


# Rectangle
def test_rectangle():
    r = Rectangle(0, 10, 20, 30)
    assert r.x0 == 0
    assert r.y0 == 10
    assert r.width == 20
    assert r.height == 30

    # check repr string
    r_repr = eval(str(r))
    assert r == r_repr


def test_rectangle_eq():
    r1 = Rectangle(0, 10, 20, 30)
    r2 = Rectangle(0, 10, 20, 30)

    assert r1 == r2

    # Rectangle == anything: error
    with pytest.raises(GeometryArithmeticError):
        r1 == 1


@pytest.mark.parametrize("p,inside",
                         [(Point(0, 0, 0), True),
                          (Point(0, 0, 100), True),
                          (Point(10, 0, 0), True),
                          (Point(0, 10, 0), True),
                          (Point(10, 10, 0), True),
                          (Point(5, 5, 0), True),
                          (Point(0, 11, 0), False),
                          (Point(11, 0, 0), False),
                          (Point(11, 11, 0), False),
                          (Point(-1, 0, 0), False),
                          (Point(0, -1, 0), False)])
def test_rectangle_is_inside(p, inside):
    r = Rectangle(0, 0, 10, 10)
    assert r.is_inside(p) == inside


# functions
@pytest.mark.parametrize("p,ll,proj_expected,t_expected",
                         [(Point(0, 0, 0), Line(Point(0, 1, 0), Vector(1, 0, 0)), Point(0, 1, 0), 0),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(1, 0, 0)), Point(0, 1, 0), -10),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(-1, 0, 0)), Point(0, 1, 0), 10),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(-10, 0, 0)), Point(0, 1, 0), 1)])
def test_project(p, ll, proj_expected, t_expected):
    proj, t = project(p, ll)
    assert type(proj) == Point
    assert (proj - proj_expected).length() == pytest.approx(0)
    assert t == pytest.approx(t_expected)


def test_distance():
    p = Point(0, 0, 0)

    # Point to point
    p2 = Point(1, 1, 1)
    assert distance(p, p2) == pytest.approx((p - p2).length())

    # Point to Line
    ll = Line(Point(1, 1, 0), Vector(1, 0, 0))
    proj, _ = project(p, ll)
    assert distance(p, ll) == pytest.approx((p - proj).length())

    # Point to Segment
    s = Segment(Point(1, 1, 0), Point(10, 1, 0))
    assert distance(p, s) == pytest.approx((p - s.start).length())
    s = Segment(Point(-10, 1, 0), Point(-1, 1, 0))
    assert distance(p, s) == pytest.approx((p - s.end).length())
    s = Segment(Point(-10, 1, 0), Point(10, 1, 0))
    assert distance(p, s) == pytest.approx((p - Point(0, 1, 0)).length())

    # Distance with anything else: error
    with pytest.raises(TypeError):
        distance(p, Vector(0, 0, 0))  # type: ignore
