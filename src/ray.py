# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from src import geometry as geom
from typing import List


class Ray:
    def __init__(self, delay: float, path_gain: float, phase: float, vertices: List[geom.Point]):
        self.delay = delay
        self.path_gain = path_gain
        self.phase = phase
        self._vertices = vertices

    @property
    def vertices(self) -> List[geom.Point]:
        return self._vertices.copy()

    @vertices.setter
    def vertices(self, vertices: List[geom.Point]):
        assert len(vertices) >= 2, "Vertices do should also include origin and destination"
        self._vertices = vertices

    def aod_azimuth(self) -> float:
        s = geom.Segment(self.vertices[0], self.vertices[1])
        return s.aod_azimuth()

    def aod_inclination(self) -> float:
        s = geom.Segment(self.vertices[0], self.vertices[1])
        return s.aod_inclination()

    def aod_elevation(self) -> float:
        s = geom.Segment(self.vertices[0], self.vertices[1])
        return s.aod_elevation()

    def aoa_azimuth(self) -> float:
        s = geom.Segment(self.vertices[-2], self.vertices[-1])
        return s.aoa_azimuth()

    def aoa_inclination(self) -> float:
        s = geom.Segment(self.vertices[-2], self.vertices[-1])
        return s.aoa_inclination()

    def aoa_elevation(self) -> float:
        s = geom.Segment(self.vertices[-2], self.vertices[-1])
        return s.aoa_elevation()
