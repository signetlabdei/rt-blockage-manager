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

import numpy as np
from typing import Optional, Tuple, Union, Sequence, overload
import math


class GeometryArithmeticError(Exception):
    """Exception raised for arithmetic errors of geometric classes"""

    def __init__(self, other):
        self.other = other
        super().__init__(f"Operation with type {type(other)} is not supported or defined")


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def z(self) -> float:
        return self.coord[2]

    @staticmethod
    def from_array(xx: Sequence) -> 'Point':
        assert len(xx) == 3, f"Invalid array: {xx}"
        return Point(xx[0], xx[1], xx[2])

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"

    def __add__(self, other: 'Vector') -> 'Point':
        if isinstance(other, Vector):
            s = self.coord + other.coord
            return Point.from_array(s)

        raise GeometryArithmeticError(other)

    @overload
    def __sub__(self, other: 'Vector') -> 'Point':
        ...

    @overload
    def __sub__(self, other: 'Point') -> 'Vector':
        ...

    def __sub__(self, other: Union['Vector', 'Point']) -> Union['Vector', 'Point']:
        if isinstance(other, Point):
            diff = self.coord - other.coord
            return Vector.from_array(diff)

        if isinstance(other, Vector):
            s = self.coord - other.coord
            return Point.from_array(s)

        raise GeometryArithmeticError(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            raise GeometryArithmeticError(other)

        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def z(self) -> float:
        return self.coord[2]

    @staticmethod
    def from_array(xx: Sequence) -> 'Vector':
        assert len(xx) == 3, f"Invalid array: {xx}"
        return Vector(xx[0], xx[1], xx[2])

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    @overload
    def __add__(self, other: 'Vector') -> 'Vector':
        ...

    @overload
    def __add__(self, other: Point) -> Point:
        ...

    def __add__(self, other: Union['Vector', Point]) -> Union['Vector', Point]:
        if isinstance(other, Vector):
            s = self.coord + other.coord
            return Vector.from_array(s)

        if isinstance(other, Point):
            s = self.coord + other.coord
            return Point.from_array(s)

        raise GeometryArithmeticError(other)

    def __sub__(self, other: 'Vector') -> 'Vector':
        if isinstance(other, Vector):
            s = self.coord - other.coord
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __neg__(self) -> 'Vector':
        return Vector.from_array(-self.coord)

    def __mul__(self, other: Union[float, int]) -> 'Vector':
        if np.issubdtype(type(other), np.integer) \
                or np.issubdtype(type(other), np.float):
            other = float(other)

        if isinstance(other, float):
            s = self.coord * other
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __rmul__(self, other: Union[float, int]) -> 'Vector':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'Vector':
        return self * (1 / other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            raise GeometryArithmeticError(other)

        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

    def azimuth(self) -> float:
        return math.atan2(self.y, self.x)

    def inclination(self) -> float:
        return math.acos(self.z / self.length())

    def elevation(self) -> float:
        return math.asin(self.z / self.length())

    def length(self) -> float:
        return np.linalg.norm(self.coord)

    def dot(self, other: 'Vector') -> float:
        return float(np.dot(self.coord, other.coord))  # type: ignore

    def normalize(self) -> 'Vector':
        return Vector.from_array(self.coord / self.length())

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector.from_array(np.cross(self.coord, other.coord))

    def as_array(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Segment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def length(self) -> float:
        return (self.end - self.start).length()

    def __repr__(self) -> str:
        return f"Segment({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            raise GeometryArithmeticError(other)

        return self.start == other.start and self.end == other.end

    def aod_azimuth(self) -> float:
        v = self.end - self.start
        return v.azimuth()

    def aod_inclination(self) -> float:
        v = self.end - self.start
        return v.inclination()

    def aod_elevation(self) -> float:
        v = self.end - self.start
        return v.elevation()

    def aoa_azimuth(self) -> float:
        v = self.start - self.end
        return v.azimuth()

    def aoa_inclination(self) -> float:
        v = self.start - self.end
        return v.inclination()

    def aoa_elevation(self) -> float:
        v = self.start - self.end
        return v.elevation()


class Line:
    def __init__(self, p: Point, v: Vector):
        assert v.length() > 0, f"{v=} is not a well-defined direction"
        self.p = p
        self.v = v

    def __repr__(self) -> str:
        return f"Line({self.p}, {self.v})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Line):
            raise GeometryArithmeticError(other)

        return self.p == other.p and self.v == other.v


class Rectangle:
    def __init__(self, x0: float, y0: float, width: float, height: float):
        self._x0 = x0
        self._y0 = y0
        self._width = width
        self._height = height

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def y0(self) -> float:
        return self._y0

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def is_inside(self, p: Point) -> bool:
        if (self.x0 <= p.x <= self.x0 + self.width) and (self.y0 <= p.y <= self.y0 + self.height):
            return True
        return False

    def __repr__(self) -> str:
        return f"Rectangle({self.x0}, {self.y0}, {self.width}, {self.height})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rectangle):
            raise GeometryArithmeticError(other)

        return self.x0 == other.x0 and self.y0 == other.y0 and \
               self.width == other.width and self.height == other.height


class Plane:
    def __init__(self, p1: Point, p2: Point, p3: Point):
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        self._equation = self._get_equation()  # TODO remove points?
        self._normal = self._get_normal()

    def _get_equation(self) -> Tuple[float, float, float, float]:
        plane_normal = (self.p2 - self.p1).cross(self.p3 - self.p1)
        plane_constant = -np.dot(plane_normal.coord, self.p1.coord)  # type: ignore
        return plane_normal.x, plane_normal.y, plane_normal.z, plane_constant

    def _get_normal(self) -> Vector:
        return Vector.from_array(self.equation[0:3]).normalize()

    @property
    def p1(self) -> Point:
        return self._p1

    @property
    def p2(self) -> Point:
        return self._p2

    @property
    def p3(self) -> Point:
        return self._p3

    @property
    def equation(self) -> Tuple[float, float, float, float]:
        return self._equation

    @property
    def normal(self) -> Vector:
        return self._normal

    def __repr__(self) -> str:
        return f"Plane({self.p1}, {self.p2}, {self.p3})"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Plane):
            raise TypeError(f"{type(o)=}")
        
        # planes are equal if their equations are linearly dependent
        dot = np.dot(self.equation, o.equation)
        norm_dot = dot / (np.linalg.norm(self.equation) * np.linalg.norm(o.equation))  # type: ignore
        return abs(norm_dot) > 1 - 1e-9

    def intersection(self, x: Union[Segment, Line]) -> Optional[Point]:
        if isinstance(x, Line):
            denom = self.normal.dot(x.v)
            if abs(denom) < 1e-9:
                # plane and line are parallel
                return None

            num = self.normal.dot(self.p1 - x.p)
            d = num / denom

            return x.p + x.v * d

        if isinstance(x, Segment):
            denom = self.normal.dot(x.end - x.start)

            if abs(denom) < 1e-9:
                # plane and segment are parallel
                return None

            num = self.normal.dot(self.p1-x.start)  # type: ignore
            d = num / denom

            if d < 0 or d > 1:
                # segment intersects plane outside its boundaries
                return None

            # segment intersects plane
            return x.start + d * (x.end - x.start)

        raise TypeError(f"{type(x)=}")


class Parallelogram3d:
    def __init__(self, p0: Point, adj1: Point, adj2: Point):
        self._p0 = p0
        self._adj1 = adj1
        self._adj2 = adj2

        assert (adj1 - p0).cross(adj2 - p0).length() > 0

        self._plane = Plane(self.p0, self.adj1, self.adj2)

    @property
    def p0(self) -> Point:
        return self._p0

    @property
    def adj1(self) -> Point:
        return self._adj1

    @property
    def adj2(self) -> Point:
        return self._adj2

    @property
    def plane(self) -> Plane:
        return self._plane
        
    def __repr__(self) -> str:
        return f"Parallelogram3d({self.p0}, {self.adj1}, {self.adj2})"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Parallelogram3d):
            raise TypeError(f"{type(o)=}")

        self_v = self.get_vertices()
        o_v = list(o.get_vertices())

        common_vertices = [sv
                                 for sv in self_v
                                 for ov in o_v
                                 if (sv-ov).length() < 1e-9]
        return len(common_vertices) == 4
    
    def get_vertices(self) -> Tuple[Point, Point, Point, Point]:
        p4 = self.adj1 + (self.adj2 - self.p0)
        return self.p0, self.adj1, self.adj2, p4

    def intersection(self, x: Union[Segment, Line]) -> Optional[Point]:
        if isinstance(x, Line):
            intersection = self._plane.intersection(x)

            if intersection is None:
                return None

            if self.in_parallelogram(intersection):
                return intersection

            return None

        if isinstance(x, Segment):
            if x.start == x.end:
                # zero-length segment
                if self.in_parallelogram(x.start):
                    return x.start
                
                return None
                
            else:
                intersection = self.intersection(Line(x.start, x.end - x.start))

            if intersection is None:
                return None

            t = (intersection - x.start).dot(x.end - x.start)
            if 0 <= t <= x.length() ** 2:
                # if intersection is within the segment
                return intersection
            else:
                return None

        raise TypeError(f"{type(x)=}")

    def in_parallelogram(self, p: Point) -> bool:
        # change of base to check whether the intersection is within the parallelogram
        v1 = self.adj1 - self.p0
        v2 = self.adj2 - self.p0
        v3 = v1.cross(v2)
        w = p - self.p0

        P = np.stack([v1.coord, v2.coord, v3.coord]).T
        new_w_coord = np.linalg.inv(P).dot(w.coord)

        if abs(new_w_coord[2]) > 1e-9:
            # does not lie on the parallelogram's plane
            return False

        if (0 <= new_w_coord[0] <= 1) and (0 <= new_w_coord[1] <= 1):
            # The intersection is expressed as function of the coordinates dictated by the parallelogram.
            # To be within the parallelogram, the new coordinates should be a linear combination of the two
            # sides of the parallelogram
            return True

        return False

class TransfMatrix:
    # From: https://math.stackexchange.com/questions/2306319/transforming-point-between-euclidean-coordinate-systems

    def __init__(self, rotation_basis: Tuple[Vector, Vector, Vector], translation_vec: Vector):
        assert all([abs(v.length()-1)<1e-9 for v in rotation_basis])
        self._rot_basis = rotation_basis
        self._rot = np.linalg.inv(
            np.array([[r.x, r.y, r.z] for r in rotation_basis]).transpose())
        self._translation = translation_vec

    def __str__(self) -> str:
        return f'{self._rot.shape} matrix:\n rotation {self._rot}, translation {self._translation}'

    def __repr__(self) -> str:
        return f'TransfMatrix({self._rot_basis}, {self._translation})'

    def change_coord_syst(self, p: Point) -> Point:
        transl = p-self._translation
        p1 = self._rot @ transl.coord

        return Point.from_array(p1)
        

# UTILITIES

def distance(p: Point, x: Union[Point, Line, Segment, Plane]) -> float:
    if isinstance(x, Point):
        return (p - x).length()

    if isinstance(x, Line):
        projection, _ = project(p, x)
        return (p - projection).length()

    if isinstance(x, Segment):
        line = Line(x.start, x.end - x.start)
        projection, t = project(p, line)

        if t > 1.0:
            # Shortest distance from segment end
            return (p - x.end).length()
        elif t < 0.0:
            # Shortest distance from segment start
            return (p - x.start).length()
        else:
            # Shortest distance from projection
            return (p - projection).length()

    if isinstance(x, Plane):
        projection, d = project(p, x)
        return d  # type: ignore

    raise TypeError(f"Type of x={type(x)} not supported")

def project(p: Point, x: Union[Line, Plane]) -> Tuple[Point, float]:
    if isinstance(x, Line):
        # From: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
        # distance(x=a+t*n, p) = ||(a-p) - ((a-p).dot(n))n||
        # p: p
        # x = a + t * n: x = x.p + t * x.v
        t1 = (p - x.p).dot(x.v.normalize())
        t = t1 / x.v.length()
        projection = x.p + t * x.v
        return projection, t
    
    if isinstance(x, Plane):
        # From: https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
        # given a plane described by point p and normal n, the projection of q onto the plane is
        # q_proj = q - dot(q - p, n) * n
        # q: p
        # n: x.normal
        # p: x.p1 (or, equivalently, x.p2, x.p3)
        projection = p-(p-x.p1).dot(x.normal)*x.normal
        return projection, (p-projection).length()
    
    raise TypeError(f"Type of x={type(x)} not supported")
