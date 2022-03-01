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

import math
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional, Callable
from src import geometry as geom
from src.mobility_model import MobilityModel, ConstantPositionMobilityModel as cpmm
from src.ray import Ray


class Obstacle(ABC):

    @abstractmethod
    def update(self, t: float) -> None:
        """
    Updates the internal state of the obstacle, in necessary.
    It should be performed at every change in time
    """

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
    def diffraction_loss(self, ray_segment: geom.Segment) -> float:
        """
        Diffraction loss [dB]
        """

    @abstractmethod
    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        """
        Returns whether the obstacle obstructs the Segment or Ray
        """

    @abstractmethod
    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        """
        Return the single Point of specular reflection on the surface of the obstacle, given start and end points.
        """


class SphereObstacle(Obstacle):
    def __init__(self,
                 mm: MobilityModel,
                 radius: float,
                 reflection_loss: float = math.inf,
                 transmission_loss: float = math.inf):
        self._mm = mm
        self._radius = radius
        self._reflection_loss = reflection_loss
        self._transmission_loss = transmission_loss

        self._last_update_t: float = 0.0

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def mobility_model(self) -> MobilityModel:
        return self._mm

    def update(self, t: float) -> None:
        if t != self._last_update_t:
            # update time
            self._last_update_t = t

    def location(self) -> geom.Point:
        return self.mobility_model.location(self._last_update_t)

    def reflection_loss(self) -> float:
        return self._reflection_loss

    def transmission_loss(self) -> float:
        return self._transmission_loss

    def diffraction_loss(self, r: geom.Segment) -> float:
        raise NotImplementedError

    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        center = self.location()

        if isinstance(r, geom.Segment):
            d = geom.distance(center, r)
            if d < self.radius:
                return True

        elif isinstance(r, Ray):
            for p1, p2 in zip(r.vertices[:-1], r.vertices[1:]):
                if geom.distance(p1, p2) == 0:
                    # Support corner case: check
                    # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                    # for more information
                    continue

                if self.obstructs(geom.Segment(p1, p2)):
                    return True

        else:
            raise TypeError(f"{type(r)=}")

        return False

    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        # https://www.geometrictools.com/Documentation/SphereReflections.pdf

        # check if a reflection is possible
        if self.obstructs(geom.Segment(pa, pb)):
            # pb in shadow cone of pb
            return None

        # Use the location of the sphere as the origin
        center = self.location()
        L = pa - center
        S = pb - center

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
                return center + N

        raise ArithmeticError("Specular reflection not found")  # pragma: no cover

    def __repr__(self):
        return f"SphereObstacle({self._mm!r}, " \
               f"{self.radius}, " \
               f"reflection_loss={self._reflection_loss}, " \
               f"transmission_loss={self._transmission_loss})"


class OrthoScreenObstacle(Obstacle):
    def __init__(self,
                 mm: MobilityModel,
                 width: float,
                 height: float,
                 reflection_loss: float = math.inf,
                 transmission_loss: float = math.inf,
                 diffraction_loss_model: Optional[Callable[[geom.Parallelogram3d, geom.Segment], float]] = None,
                 distance_threshold: float = 0):
        self._mm = mm
        self._width = width
        self._height = height
        self._reflection_loss = reflection_loss
        self._transmission_loss = transmission_loss
        self._diffraction_loss_model = diffraction_loss_model
        self._distance_threshold = distance_threshold

        assert self._distance_threshold >= 0

        self._last_update_t: float = 0.0

    @property
    def height(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    @property
    def mobility_model(self) -> MobilityModel:
        return self._mm

    def update(self, t: float) -> None:
        if t != self._last_update_t:
            self._last_update_t = t

    def location(self) -> geom.Point:
        return self.mobility_model.location(self._last_update_t)

    def reflection_loss(self) -> float:
        return self._reflection_loss

    def diffraction_loss(self, ray_segment: geom.Segment) -> float:
        assert self._diffraction_loss_model is not None

        ortho_screen = self._get_ortho_screen(ray_segment)
        return self._diffraction_loss_model(ortho_screen, ray_segment)
    
    def _get_ortho_screen(self, seg: geom.Segment) -> geom.Parallelogram3d:
        bottom_center = self.location()

        ray_line = geom.Line(seg.start, seg.end - seg.start)

        z_vector = geom.Vector(0, 0, 1)
        if abs(ray_line.v.normalize().dot(z_vector)) > 1 - 1e-9:
            # segment is vertical: cannot reliably make the obstacle orthogonal
            proj, _ = geom.project(bottom_center, ray_line)  # make the obstacle's plane orthogonal to the bottom_center-ray_line segment
            obs_ray_vector = proj-bottom_center
            if obs_ray_vector.length()>0:
                base_vector = z_vector.cross(obs_ray_vector).normalize()
            else:
                base_vector = geom.Vector(1,0,0)  # if ray_line belongs to the plane of the obstacle, obs_ray_vector is a null vector. Align to x-axis

        else:
            # aligning rectangle orthogonally to the given segment
            base_vector = z_vector.cross(ray_line.v).normalize()

        bottom_right = bottom_center + self.width / 2 * base_vector
        bottom_left = bottom_center - self.width / 2 * base_vector
        top_left = bottom_left + self.height * z_vector

        return geom.Parallelogram3d(bottom_left, bottom_right, top_left)

    def transmission_loss(self) -> float:
        return self._transmission_loss

    def distance(self, s: geom.Segment) -> float:
        if not isinstance(s, geom.Segment):
            raise TypeError(f"{type(s)}")

        ortho_screen = self._get_ortho_screen(s)
        if ortho_screen.intersection(s) is not None:
            return 0

        # else: no intersection
        plane = geom.Plane(ortho_screen.p0, ortho_screen.adj1, ortho_screen.adj2)

        intersection = plane.intersection(s)
        if intersection is None:
            # segment is parallel to plane
            return math.inf  # TODO is this ok?

        # segment intersects plane outside the rectangle
        p0, p1, p2, p3 = ortho_screen.get_vertices()

        d01 = geom.distance(intersection, geom.Segment(p0, p1))
        d02 = geom.distance(intersection, geom.Segment(p0, p2))
        d31 = geom.distance(intersection, geom.Segment(p3, p1))
        d32 = geom.distance(intersection, geom.Segment(p3, p2))

        return min(d01, d02, d31, d32)

    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        if isinstance(r, geom.Segment):
            if abs(geom.distance(r.start, r.end))<1e-9:
                # Support corner case: check
                # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                # for more information
                return False
            else:
                return self.distance(r) <= self._distance_threshold

        elif isinstance(r, Ray):
            for p1, p2 in zip(r.vertices[:-1], r.vertices[1:]):
                if abs(geom.distance(p1, p2))<1e-9:
                    # Support corner case: check
                    # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                    # for more information
                    continue

                if self.obstructs(geom.Segment(p1, p2)):
                    return True

            return False

        else:
            raise TypeError(f"{type(r)}")

    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        raise NotImplementedError("Specular reflection not yet implemented")  # pragma: no cover

    def __repr__(self):
        raise NotImplementedError


class ScreenObstacle(Obstacle):

    def __init__(self,
                    mm: MobilityModel,
                    width: float,
                    height: float,
                    base_vector: geom.Vector = geom.Vector(0,1,0),
                    height_vector: geom.Vector = geom.Vector(0,0,1),
                    reflection_loss: float = math.inf,
                    transmission_loss: float = math.inf,
                    diffraction_loss_model: Optional[Callable[[geom.Parallelogram3d, geom.Segment], float]] = None,
                    distance_threshold: float = 0):
        self._mm = mm

        # base_vector and height_vector define respectively the directions of the geometrical structure
        # of the obstacle(base and height), thus they need to be orthogonal for screen obstacles.

        # check othogonality of base_vector and height_vector

        assert base_vector.dot(height_vector) == 0, "given base_vector and height_vector are not orthogonal"
        self._base_vector = base_vector
        self._height_vector = height_vector
        self._width = width
        self._height = height
        self._reflection_loss = reflection_loss
        self._transmission_loss = transmission_loss
        self._diffraction_loss_model = diffraction_loss_model
        self._distance_threshold = distance_threshold

        assert self._distance_threshold >= 0

        self._last_update_t: float = 0.0

    @property
    def height(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    @property
    def mobility_model(self) -> MobilityModel:
        return self._mm

    @property
    def base_vector(self) -> geom.Vector:
        return self._base_vector

    @property
    def height_vector(self) -> geom.Vector:
        return self._height_vector


    def update(self, t: float) -> None:
        if t != self._last_update_t:
            self._last_update_t = t


    def location(self) -> geom.Point:
        return self._mm.location(self._last_update_t)

    def elevation_angle(self, angle_unit :str  = 'rad') -> float:
        #get the elevation angle in rad
        return self._mm.get_elev_angle(self._last_update_t, angle_unit = angle_unit)

    def azimuth_angle(self, angle_unit :str  = 'rad') -> float:
        #get the azimuth angle in rad
        return self._mm.get_azim_angle(self._last_update_t, angle_unit=angle_unit)

    def reflection_loss(self) -> float:
        return self._reflection_loss

    def transmission_loss(self) -> float:
        return self._transmission_loss

    def get_screen(self) -> geom.Parallelogram3d:

        # start from the origin

        origin = geom.Point(0,0,0)
        bottom_center_origin = origin - self._height_vector * self._height / 2
        p0 = bottom_center_origin - self.base_vector * self._width / 2
        adj1 = p0 + self.base_vector * self._width
        adj2 = p0 + self._height_vector * self._height

        # elevation rotation

        sin_elev = math.sin(self.elevation_angle())
        cos_elev = math.cos(self.elevation_angle())
        rot_axis = (p0-adj1).normalize()
        rot_x = rot_axis.x
        rot_y = rot_axis.y
        rot_z = rot_axis.z

        # Rotaton matrix R
        R_elev = np.array([[rot_x**2 + (1-rot_x**2)*cos_elev , (1-cos_elev)*rot_x*rot_y - sin_elev * rot_z , (1-cos_elev)*rot_x*rot_z + sin_elev * rot_y],
        [(1-cos_elev)*rot_x*rot_y + sin_elev*rot_z, rot_y**2 + (1-rot_y**2)*cos_elev, (1-cos_elev)*rot_y*rot_z-sin_elev * rot_x],
        [(1-cos_elev)*rot_x*rot_z - sin_elev*rot_y, (1-cos_elev)*rot_y*rot_z + sin_elev * rot_x , rot_z**2 + (1-rot_z**2)*cos_elev]])

        #Apply R to edge points
        adj2 = R_elev @ [adj2.x,adj2.y,adj2.z]
        adj2 = geom.Point(adj2[0],adj2[1],adj2[2])
        adj1 = R_elev @ [adj1.x,adj1.y,adj1.z]
        adj1 = geom.Point(adj1[0],adj1[1],adj1[2])
        p0 = R_elev @ [p0.x,p0.y,p0.z]
        p0 = geom.Point(p0[0],p0[1],p0[2])

        # azimuth rotation

        sin_azim = math.sin(self.azimuth_angle())
        cos_azim = math.cos(self.azimuth_angle())
        rot_axis = (p0-adj2).normalize()
        rot_x = rot_axis.x
        rot_y = rot_axis.y
        rot_z = rot_axis.z

        # Rotaton matrix R
        R_azim = np.array([[rot_x**2 + (1-rot_x**2)*cos_azim , (1-cos_azim)*rot_x*rot_y - sin_azim * rot_z , (1-cos_azim)*rot_x*rot_z + sin_azim * rot_y],
            [(1-cos_azim)*rot_x*rot_y + sin_azim*rot_z, rot_y**2 + (1-rot_y**2)*cos_azim, (1-cos_azim)*rot_y*rot_z-sin_azim * rot_x],
            [(1-cos_azim)*rot_x*rot_z - sin_azim*rot_y, (1-cos_azim)*rot_y*rot_z + sin_azim * rot_x , rot_z**2 + (1-rot_z**2)*cos_azim]])

        #Apply R to edge points
        adj2 = R_azim @ [adj2.x,adj2.y,adj2.z]
        adj1 = R_azim @ [adj1.x,adj1.y,adj1.z]
        p0 = R_azim @ [p0.x,p0.y,p0.z]

        # shift the tilted obstacle to its location

        bottom_center = self.location()

        shift_vector = [bottom_center.x-bottom_center_origin.x,bottom_center.y-bottom_center_origin.y,bottom_center.z-bottom_center_origin.z]
        
        adj2 = geom.Point(adj2[0] + shift_vector[0], adj2[1] + shift_vector[1], adj2[2] + shift_vector[2])
        adj1 = geom.Point(adj1[0] + shift_vector[0], adj1[1] + shift_vector[1], adj1[2] + shift_vector[2])
        p0 = geom.Point(p0[0] + shift_vector[0], p0[1] + shift_vector[1], p0[2] + shift_vector[2])


        return geom.Parallelogram3d(p0, adj1, adj2)

    def diffraction_loss(self, ray_segment: geom.Segment) -> float:
        assert self._diffraction_loss_model is not None

        screen = self.get_screen()
        return self._diffraction_loss_model(screen, ray_segment)

    def obstructs(self, r: Union[geom.Segment, Ray]) -> bool:
        if type(r) is geom.Segment:
            if abs(geom.distance(r.start, r.end))<1e-9:
                # Support corner case: check
                # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                # for more information
                return False
            else:
                return self.distance(r) <= self._distance_threshold

        elif type(r) is Ray:
            for p1, p2 in zip(r.vertices[:-1], r.vertices[1:]):
                if abs(geom.distance(p1, p2))<1e-9:
                    # Support corner case: check
                    # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                    # for more information
                    continue

                if self.obstructs(geom.Segment(p1, p2)):
                    return True

            return False

        else:
            raise TypeError(f"{type(r)}")

    def distance(self, s: geom.Segment) -> float:
        if not isinstance(s, geom.Segment):
            raise TypeError(f"{type(s)}")

        screen = self.get_screen()
        if screen.intersection(s) is not None:
            return 0

        # else: no intersection
        plane = geom.Plane(screen.p0, screen.adj1, screen.adj2)

        intersection = plane.intersection(s)
        if intersection is None:
            # segment is parallel to plane
            return math.inf  # TODO is this ok?

        # segment intersects plane outside the rectangle
        p0, p1, p2, p3 = screen.get_vertices()

        d01 = geom.distance(intersection, geom.Segment(p0, p1))
        d02 = geom.distance(intersection, geom.Segment(p0, p2))
        d31 = geom.distance(intersection, geom.Segment(p3, p1))
        d32 = geom.distance(intersection, geom.Segment(p3, p2))

        return min(d01, d02, d31, d32)


    def specular_reflection(self, pa: geom.Point, pb: geom.Point) -> Optional[geom.Point]:
        raise NotImplementedError("Specular reflection not yet implemented")  # pragma: no cover


if __name__ == '__main__':
    s = SphereObstacle(cpmm(geom.Point(0, 0, 0)), 0.2)

    pointa = geom.Point(0, 3, 0)
    pointb = geom.Point(3, 0, 0)

    print('s.obstructs(geom.Segment(pa, pb))=',
          s.obstructs(geom.Segment(pointa, pointb)))
    print('reflection: ', s.specular_reflection(pointa, pointb))
