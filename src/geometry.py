# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

import numpy as np
from typing import Tuple, Union, Sequence


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
        if type(other) == Vector:
            s = self.coord + other.coord
            return Point.from_array(s)

        raise GeometryArithmeticError(other)

    def __sub__(self, other: 'Point') -> 'Vector':
        if type(other) == type(self):
            diff = self.coord - other.coord
            return Vector.from_array(diff)

        raise GeometryArithmeticError(other)

    def __eq__(self, other: 'Point') -> bool:
        if type(other) != type(self):
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

    def __add__(self, other: Union['Vector', Point]) -> Union['Vector', Point]:
        if type(other) == type(self):
            s = self.coord + other.coord
            return Vector.from_array(s)

        if type(other) == Point:
            s = self.coord + other.coord
            return Point.from_array(s)

        raise GeometryArithmeticError(other)

    def __sub__(self, other: 'Vector') -> 'Vector':
        if type(other) == type(self):
            s = self.coord - other.coord
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __mul__(self, other: Union[float, int]) -> 'Vector':
        if np.issubdtype(type(other), np.integer) \
                or np.issubdtype(type(other), np.float):
            other = float(other)

        if type(other) == float:
            s = self.coord * other
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __rmul__(self, other: Union[float, int]) -> 'Vector':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'Vector':
        return self * (1 / other)

    def __eq__(self, other: 'Vector') -> bool:
        if type(other) != type(self):
            raise GeometryArithmeticError(other)

        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

    def azimuth(self) -> float:
        return np.arctan2(self.y, self.x)

    def inclination(self) -> float:
        return np.arccos(self.z / self.length())

    def elevation(self) -> float:
        return np.arcsin(self.z / self.length())

    def length(self) -> float:
        return np.linalg.norm(self.coord)

    def dot(self, other: 'Vector') -> float:
        return np.dot(self.coord, other.coord).item()

    def normalize(self) -> 'Vector':
        return Vector.from_array(self.coord / self.length())


class Segment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def length(self) -> float:
        return (self.end - self.start).length()

    def __repr__(self) -> str:
        return f"Segment({self.start}, {self.end})"

    def __eq__(self, other: 'Segment') -> bool:
        if type(other) != type(self):
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
        assert v.length() > 0, f"v={v} is not a well-defined direction"
        self.p = p
        self.v = v

    def __repr__(self) -> str:
        return f"Line({self.p}, {self.v})"

    def __eq__(self, other: 'Line') -> bool:
        if type(other) != type(self):
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

    def __eq__(self, other: 'Rectangle') -> bool:
        if type(other) != type(self):
            raise GeometryArithmeticError(other)

        return self.x0 == other.x0 and self.y0 == other.y0 and \
               self.width == other.width and self.height == other.height


def distance(p: Point, x: Union[Point, Line, Segment]) -> float:
    if type(x) == Point:
        return (p - x).length()

    if type(x) == Line:
        projection, _ = project(p, x)
        return (p - projection).length()

    if type(x) == Segment:
        line = Line(x.start, x.end - x.start)
        projection, t = project(p, line)

        if t > x.length():
            # Shortest distance from segment end
            return (p - x.end).length()
        elif t < 0.0:
            # Shortest distance from segment start
            return (p - x.start).length()
        else:
            # Shortest distance from projection
            return (p - projection).length()

    raise TypeError(f"Type of x={type(x)} not supported")


def project(p: Point, x: Line) -> Tuple[Point, float]:
    # From: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
    # distance(x=a+t*n, p) = ||(a-p) - ((a-p).dot(n))n||
    # p: p
    # x = a + t * n: x = x.p + t * x.v
    norm_v = x.v.normalize()
    t = -(x.p - p).dot(norm_v)
    projection = x.p + t * norm_v
    return projection, t
