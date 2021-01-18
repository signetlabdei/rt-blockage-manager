import numpy as np
from typing import Tuple, Union


class GeometryArithmeticError(Exception):
    """Exception raised for arithmetic errors of geometric classes"""

    def __init__(self, other):
        self.other = other
        super().__init__(f"Operation with type {type(other)} is not supported or defined")


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    def __add__(self, other):
        raise GeometryArithmeticError(other)

    def __sub__(self, other):
        if type(other) == type(self):
            diff = other.coord - self.coord
            return Vector.from_array(diff)

        raise GeometryArithmeticError(other)


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    @staticmethod
    def from_array(xx) -> 'Vector':
        assert len(xx) == 3, f"Invalid array: {xx}"
        return Vector(xx[0], xx[1], xx[2])

    def __add__(self, other):
        if type(other) == type(self):
            s = self.coord + other.coord
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __sub__(self, other):
        if type(other) == type(self):
            s = self.coord - other.coord
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __mul__(self, other):
        if type(other) == float:
            s = self.coord * other
            return Vector.from_array(s)

        raise GeometryArithmeticError(other)

    def __rmul__(self, other):
        return self * other

    def azimuth(self) -> float:
        return np.arctan2(self.coord[1], self.coord[0])

    def inclination(self) -> float:
        return np.arccos(self.coord[2] / self.length())

    def elevation(self) -> float:
        return np.arcsin(self.coord[2] / self.length())

    def length(self) -> float:
        return np.linalg.norm(self.coord)

    def dot(self, other: 'Vector') -> float:
        return np.dot(self.coord, other.coord).item()


class Segment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

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
        self.p = p
        self.v = v


def distance(p: Point, x: Union[Line, Segment]) -> float:
    if type(x) == Line:
        projection, _ = project(p, x)
        return (p - projection).length()

    if type(x) == Segment:
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


def project(p: Point, x: Line) -> Tuple[Point, float]:
    # From: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
    # distance(x=a+t*n, p) = ||(a-p) - ((a-p).dot(n))n||
    # p: p
    # x = a + t * n: x = x.p + t * x.v
    t = -(x.p - p).dot(x.v)
    projection = x.p + t * x.v
    return projection, t
