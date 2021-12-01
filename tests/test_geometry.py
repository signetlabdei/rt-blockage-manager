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

from src.geometry import GeometryArithmeticError, Parallelogram3d, Plane, Vector, Point, Segment, Line, Rectangle, TransfMatrix, distance, project
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


@pytest.mark.parametrize("v1,v2,res", [(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)),
                                       (Vector(0, 1, 0), Vector(1, 0, 0), Vector(0, 0, -1)),
                                       (Vector(1, 2, 3), Vector(3, 4, 5), Vector(-2, 4, -2))])
def test_cross(v1, v2, res):
    v3 = v1.cross(v2)
    assert (v3 - res).length() == pytest.approx(0)


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
    assert isinstance(p_sum, Point)

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

def test_vector_neg():
    v = Vector(1, 2, 3)
    v_neg = -v
    assert (v_neg + v).length() == pytest.approx(0)

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
    assert isinstance(p2, Point)

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


# Plane
def test_plane():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)
    assert p.p1 == p1
    assert p.p2 == p2
    assert p.p3 == p3

    # check repr string
    p_repr = eval(str(p))
    assert p == p_repr


@pytest.mark.parametrize("test_plane,result",
                         [(Plane(Point(1, -2, 0), Point(3, 1, 4), Point(0, -1, 2)), True),
                          (Plane(Point(18/2, 0, 0), Point(0, -18/8, 0), Point(0, 0, 18/5)), True),
                          (Plane(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)), False)
                          ])
def test_plane_equals(test_plane, result):
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    assert (p == test_plane) == result


def test_plane_equals_raise():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    with pytest.raises(TypeError):
        assert p == Point(0,0,0)


def test_plane_normal():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    n = p.normal
    expected = Vector(2, -8, 5)
    assert abs(n.normalize().dot(expected.normalize())) == pytest.approx(1)


def test_plane_equation():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    eq = p.equation
    expected = (2, -8, 5, -18)

    # equations can be offset by a multiplicative term
    ratio = eq[0] / expected[0]

    for eq_term, expected_term in zip(eq, expected):
        assert eq_term == pytest.approx(expected_term * ratio)


def test_plane_intersection_line():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    l = Line(Point(0,0,0), Vector(1,0,0))
    intersection = p.intersection(l)
    expected = Point(9, 0, 0)

    assert intersection is not None
    assert (intersection - expected).length() == pytest.approx(0)


@pytest.mark.parametrize("segment,expected",
                         [(Segment(Point(0, 0, 0), Point(10, 0, 0)), Point(9, 0, 0)),
                          (Segment(Point(0, 0, 0), Point(8, 0, 0)), None),
                          (Segment(Point(10, 0, 0), Point(100, 0, 0)), None)
                         ])
def test_plane_intersection_segment(segment, expected):
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    intersection = p.intersection(segment)

    if expected is None:
        assert intersection is None
    else:
        assert intersection is not None
        assert (intersection - expected).length() == pytest.approx(0)


def test_plane_intersection_parallel():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    parallel = p2 - p1

    # Line
    p0 = Point(0, 0, 0,)
    l = Line(p0, parallel)
    assert p.intersection(l) is None

    # Segment
    s = Segment(p0, p0 + parallel)
    assert p.intersection(s) is None


def test_plane_intersection_error():
    p1 = Point(1, -2, 0)
    p2 = Point(3, 1, 4)
    p3 = Point(0, -1, 2)
    p = Plane(p1, p2, p3)

    with pytest.raises(TypeError):
        p.intersection(Point(0, 0, 0))  # type: ignore


# Parallelogram3D
def test_parallelogram3d():
    p0 = Point(1, -2, 0)
    adj1 = Point(3, 1, 4)
    adj2 = Point(0, -1, 2)
    p = Parallelogram3d(p0, adj1, adj2)
    assert p.p0 == p0
    assert p.adj1 == adj1
    assert p.adj2 == adj2

    # check repr string
    p_repr = eval(str(p))
    assert p == p_repr


@pytest.mark.parametrize("test_parallelogram3d,result",
                         [(Parallelogram3d(Point(1, -2, 0), Point(3, 1, 4), Point(0, -1, 2)), True),
                          (Parallelogram3d(Point(1, -2, 0),
                           Point(0, -1, 2), Point(3, 1, 4)), True),
                          (Parallelogram3d(Point(0, -1, 2),
                           Point(1, -2, 0), Point(2, 2, 6)), True),
                          (Parallelogram3d(Point(0, 0, 0), Point(
                              1, 0, 0), Point(0, 1, 0)), False)
                          ])
def test_parallelogram3d_equals(test_parallelogram3d, result):
    p0 = Point(1, -2, 0)
    adj1 = Point(3, 1, 4)
    adj2 = Point(0, -1, 2)
    p = Parallelogram3d(p0, adj1, adj2)

    assert (p == test_parallelogram3d) == result


def test_parallelogram3d_equals_raise():
    p0 = Point(1, -2, 0)
    adj1 = Point(3, 1, 4)
    adj2 = Point(0, -1, 2)
    p = Parallelogram3d(p0, adj1, adj2)

    with pytest.raises(TypeError):
        assert p == Point(0, 0, 0)


@pytest.mark.parametrize("segment,expected",
                         [(Segment(Point(3, -1, 2), Point(3, 1, 2)), Point(3, 0, 2)),
                          (Segment(Point(3, -2, 2), Point(3, -1, 2)), None),
                          (Segment(Point(5, -1, 2), Point(5, 1, 2)), None),
                          (Segment(Point(3, -2, 2), Point(3, -1, 2)), None),
                          (Segment(Point(1, -2, -2), Point(1, -2, 5)), None),
                          (Segment(Point(0, 0, 0), Point(0, 0, 0)), Point(0, 0, 0))
                          ])
def test_parallelogram3d_intersection_segment(segment, expected):
    p0 = Point(0, 0, 0)
    adj1 = Point(3, 0, 0)
    adj2 = Point(2, 0, 4)
    p = Parallelogram3d(p0, adj1, adj2)

    intersection = p.intersection(segment)

    if expected is None:
        assert intersection is None
    else:
        assert intersection is not None
        assert (intersection - expected).length() == pytest.approx(0)


def test_parallelogram3d_intersection_error():
    p0 = Point(1, -2, 0)
    adj1 = Point(3, 1, 4)
    adj2 = Point(0, -1, 2)
    p = Parallelogram3d(p0, adj1, adj2)

    with pytest.raises(TypeError):
        p.intersection(Point(0, 0, 0))  # type: ignore


def test_parallelogram3d_in_parallelogram():
    # test from bugfix
    p0 = Point(3.94616, 2.8, 0.0)
    adj1 = Point(3.94616, 3.2, 0.0)
    adj2 = Point(3.94616, 2.8, 1.7)
    par = Parallelogram3d(p0, adj1, adj2)

    point = Point(3.9461599999999994, 3.0, 1.6)
    assert par.in_parallelogram(point)


# functions
@pytest.mark.parametrize("p,ll,proj_expected,t_expected",
                         [(Point(0, 0, 0), Line(Point(0, 1, 0), Vector(1, 0, 0)), Point(0, 1, 0), 0),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(1, 0, 0)), Point(0, 1, 0), -10),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(-1, 0, 0)), Point(0, 1, 0), 10),
                          (Point(0, 0, 0), Line(Point(10, 1, 0), Vector(-10, 0, 0)), Point(0, 1, 0), 1)])
def test_project_on_line(p, ll, proj_expected, t_expected):
    proj, t = project(p, ll)
    assert isinstance(proj, Point)
    assert (proj - proj_expected).length() == pytest.approx(0)
    assert t == pytest.approx(t_expected)


@pytest.mark.parametrize("p,pp,proj_expected,t_expected",
                         [(Point(0, 0, 0), Plane(Point(0, 1, 0), Point(1, 0, 0), Point(0,1,1)), Point(1/2, 1/2, 0), np.sqrt(2)/2),
                          (Point(0, 0, 0), Plane(Point(0, 1, 0), Point(1, 0, 0), Point(0,0,1)), Point(1/3, 1/3, 1/3), np.sqrt(3)/3),
                          (Point(0, 0, 0), Plane(Point(10, 1, 0), Point(10, -1, 0), Point(10, 1, 1)), Point(10, 0, 0), 10),
                          (Point(0, 0, 0), Plane(Point(10, 1, -1), Point(10, -1, -1), Point(-10, 1, -1)), Point(0, 0, -1), 1),
                          (Point(0, 0, 0), Plane(Point(10, 1, 0), Point(-10, 1, 0), Point(10, 1, 1)), Point(0, 1, 0), 1)])
def test_project_on_plane(p, pp, proj_expected, t_expected):
    proj, t = project(p, pp)
    assert isinstance(proj, Point)
    assert (proj - proj_expected).length() == pytest.approx(0)
    assert t == pytest.approx(t_expected)


def test_distance():
    point = Point(0, 0, 0)

    # Point to point
    p2 = Point(1, 1, 1)
    assert distance(point, p2) == pytest.approx((point - p2).length())

    # Point to Line
    ll = Line(Point(1, 1, 0), Vector(1, 0, 0))
    proj, _ = project(point, ll)
    assert distance(point, ll) == pytest.approx((point - proj).length())

    # Point to Segment
    s = Segment(Point(1, 1, 0), Point(10, 1, 0))
    assert distance(point, s) == pytest.approx((point - s.start).length())
    s = Segment(Point(-10, 1, 0), Point(-1, 1, 0))
    assert distance(point, s) == pytest.approx((point - s.end).length())
    s = Segment(Point(-10, 1, 0), Point(10, 1, 0))
    assert distance(point, s) == pytest.approx((point - Point(0, 1, 0)).length())

    # Point to Plane
    plane = Plane(Point(1, -2, 0), Point(3, 1, 4), Point(0, -1, 2))
    assert distance(point, plane) == pytest.approx(18 / math.sqrt(93))
    plane = Plane(Point(0, 0, 0), Point(3, 1, 4), Point(0, -1, 2))
    assert distance(point, plane) == pytest.approx(0)

    # Distance with anything else: error
    with pytest.raises(TypeError):
        distance(point, Vector(0, 0, 0))  # type: ignore


@pytest.mark.parametrize("p, translation, rotation, expected_p1",
                         [(Point(10, -2, 5), Vector(0, 0, 0), (Vector(1,0,0), Vector(0,1,0), Vector(0,0,1)), Point(10, -2, 5)),  # identity
                          (Point(0, 0, 0), Vector(1, 1, 1), (Vector(1,0,0), Vector(0,1,0), Vector(0,0,1)), Point(-1, -1, -1)),  # positive translation of origin
                          (Point(0, 0, 0), Vector(-1, -1, -1), (Vector(1,0,0), Vector(0,1,0), Vector(0,0,1)), Point(1, 1, 1)),  # negative translation of origin
                          (Point(0, 0, 0), Vector(0, 0, 0), (Vector(1,0,0), Vector(0,1/np.sqrt(2),-1/np.sqrt(2)), Vector(0,1/np.sqrt(2),1/np.sqrt(2))), Point(0, 0, 0)),  # 45 deg rotation on x axis, origin
                          (Point(0, 0, 0), Vector(0, 0, 0), (Vector(-1/np.sqrt(2),0,1/np.sqrt(2)), Vector(0,1,0), Vector(1/np.sqrt(2),0,1/np.sqrt(2))), Point(0, 0, 0)),  # 45 deg rotation on y axis, origin
                          (Point(0, 0, 0), Vector(0, 0, 0), (Vector(1/np.sqrt(2),1/np.sqrt(2),0), Vector(-1/np.sqrt(2),1/np.sqrt(2),0), Vector(0,0,1)), Point(0, 0, 0)),  # 45 deg rotation on z axis, origin
                          (Point(10, -2, 5), Vector(0, 0, 0), (Vector(1,0,0), Vector(0,1/np.sqrt(2),-1/np.sqrt(2)), Vector(0,1/np.sqrt(2),1/np.sqrt(2))), Point(10, -np.sqrt(2)/2*(5+2), np.sqrt(2)/2*(5-2))),  # 45 deg rotation on x axis, origin
                          ])
def test_TransfMatrix(p, translation, rotation, expected_p1):
    tf = TransfMatrix(translation_vec=translation, rotation_basis=rotation)

    p1 = tf.change_coord_syst(p)

    assert distance(expected_p1, p1) == pytest.approx(0)
