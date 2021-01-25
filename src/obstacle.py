# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from abc import ABC, abstractmethod
import numpy as np
from src import geometry as geom
from src.ray import Ray
from typing import Union, Optional


class Obstacle(ABC):

    @abstractmethod
    def location(self) -> geom.Point:
        pass

    @abstractmethod
    def reflection_loss(self) -> float:
        pass

    @abstractmethod
    def transmission_loss(self) -> float:
        pass

    @abstractmethod
    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        pass

    @abstractmethod
    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        pass


class SphereObstacle(Obstacle):
    def __init__(self,
                 location: geom.Point,
                 radius: float,
                 reflection_loss=np.inf,
                 transmission_loss=np.inf):
        self._location = location
        self._radius = radius
        self._reflection_loss = reflection_loss
        self._transmission_loss = transmission_loss

    @property
    def location(self):
        return self._location

    @property
    def radius(self):
        return self._radius

    @property
    def reflection_loss(self):
        return self._reflection_loss

    @property
    def transmission_loss(self):
        return self._transmission_loss

    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        if type(r) is geom.Segment:
            d = geom.distance(self.location, r)
            if d < self.radius:
                return True

        elif type(r) is Ray:
            for p1, p2 in zip(r.vertices[:-1], r.vertices[1:]):
                segment = geom.Segment(p1, p2)
                d = geom.distance(self.location, segment)
                if d < self.radius:
                    return True

        return False

    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        # https://www.geometrictools.com/Documentation/SphereReflections.pdf

        # check if a reflection is possible
        if self.obstructs(geom.Segment(pa, pb)):
            # pb in shadow cone of pb
            return None

        # Use the center of the sphere as the origin
        L = pa - self.location
        S = pb - self.location

        a = S.dot(S)
        b = S.dot(L)
        c = L.dot(L)

        poly = [4 * c * (a * c - b ** 2),
                -4 * (a * c - b ** 2),
                a + 2 * b + c - 4 * a * c,
                2 * (a - b),
                a - 1]
        r = np.roots(poly)

        for y in filter(lambda root: root > 0, r):
            x = (-2 * c * y ** 2 + y + 1) / (2 * b * y + 1)
            if x > 0:
                print(x, y)
                # valid point on surface only if both x and y are >0
                # rescale it to sphere radius
                N = (x * S + y * L) * self.radius

                # Return point in space
                return self.location + N

        raise ArithmeticError("Specular reflection not found")

    def __repr__(self):
        return f"SphereObstacle({self.location}, {self.radius}, reflection_loss={self.reflection_loss}, " \
               f"transmission_loss={self.transmission_loss})"


if __name__ == '__main__':
    s = SphereObstacle(geom.Point(0, 0, 0), 0.2)

    pointa = geom.Point(0, 3, 0)
    pointb = geom.Point(3, 0, 0)

    print('s.obstructs(geom.Segment(pa, pb))=', s.obstructs(geom.Segment(pointa, pointb)))
    print('reflection: ', s.specular_reflection(pointa, pointb))
