from abc import ABC, abstractmethod
import numpy as np
from src import geometry as geom
from src.ray import Ray
from typing import Union


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
    def obstructs(self, ray: Ray) -> bool:
        pass

    @abstractmethod
    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> geom.Point:
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

    def obstructs(self, ray: Ray) -> bool:
        for p1, p2 in zip(ray.vertices[:-1], ray.vertices[1:]):
            segment = geom.Segment(p1, p2)
            d = geom.distance(self._location, segment)
            if d < self._radius:
                return True

        return False

    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Union[geom.Point, None]:
        # https://www.geometrictools.com/Documentation/SphereReflections.pdf
        return None
