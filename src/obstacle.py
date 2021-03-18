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
        """
        Get location of the obstacle as a Point
        """

    @abstractmethod
    def reflection_loss(self) -> float:
        """
        Reflection loss [dB]
        """

    @abstractmethod
    def transmission_loss(self) -> float:
        """
        Transmission loss [dB]
        """

    @abstractmethod
    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        """
        Returns whether the obstacle obstructs the Segmet or Ray
        """

    @abstractmethod
    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        """
        Return the single Point of specular reflection on the surface of the obstacle, given start and end points.
        """


class SphereObstacle(Obstacle):
    def __init__(self,
                 center: geom.Point,
                 radius: float,
                 reflection_loss=np.inf,
                 transmission_loss=np.inf):
        self._center = center
        self._radius = radius
        self._reflection_loss = reflection_loss
        self._transmission_loss = transmission_loss

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def location(self):
        return self.center

    def reflection_loss(self):
        return self._reflection_loss

    def transmission_loss(self):
        return self._transmission_loss

    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        if type(r) is geom.Segment:
            d = geom.distance(self.center, r)
            if d < self.radius:
                return True

        elif type(r) is Ray:
            for p1, p2 in zip(r.vertices[:-1], r.vertices[1:]):
                segment = geom.Segment(p1, p2)
                d = geom.distance(self.center, segment)
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
        L = pa - self.center
        S = pb - self.center

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
                # valid point on surface only if both x and y are >0
                # rescale it to sphere radius
                N = (x * S + y * L) * self.radius

                # Return point in space
                return self.center + N

        raise ArithmeticError("Specular reflection not found")  # pragma: no cover

    def __repr__(self):
        return f"SphereObstacle({self.center}, {self.radius}, reflection_loss={self._reflection_loss}, " \
               f"transmission_loss={self._transmission_loss})"


if __name__ == '__main__':
    s = SphereObstacle(geom.Point(0, 0, 0), 0.2)

    pointa = geom.Point(0, 3, 0)
    pointb = geom.Point(3, 0, 0)

    print('s.obstructs(geom.Segment(pa, pb))=', s.obstructs(geom.Segment(pointa, pointb)))
    print('reflection: ', s.specular_reflection(pointa, pointb))
