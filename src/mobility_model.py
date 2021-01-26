# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from abc import ABC, abstractmethod
from src.geometry import Point, Vector, Rectangle
from typing import Optional, Callable, Sequence, Union
import collections
import logging

import numpy as np


class MobilityModel(ABC):
    @abstractmethod
    def location(self, t: float) -> Point:
        """
        Location Point at time t
        """


class PositionAllocation:
    def __init__(self, x: Callable[[], float], y: Callable[[], float], z: Callable[[], float]):
        self._x = x
        self._y = y
        self._z = z

    def __call__(self, *args, **kwargs) -> Point:
        return Point(self._x(), self._y(), self._z())


class ConstantPositionMobilityModelModel(MobilityModel):
    def __init__(self, pos: Point):
        self._pos = pos

    def location(self, t: float) -> Point:
        return self._pos


class ConstantVelocityMobilityModel(MobilityModel):
    def __init__(self, start_pos: Point, vel: Vector):
        self._start_pos = start_pos
        self._vel = vel

    def location(self, t: float) -> Point:
        return self._start_pos + t * self._vel


class ConstantAccelerationMobilityModel(MobilityModel):
    def __init__(self, start_pos: Point, vel: Vector, accel: Vector):
        self._start_pos = start_pos
        self._vel = vel
        self._accel = accel

    def location(self, t: float) -> Point:
        return self._start_pos + t * self._vel + 0.5 * t ** 2 * self._accel


class RandomWaypointMobilityModel(MobilityModel):
    def __init__(self, bounding_box: Rectangle, position_allocation: PositionAllocation,
                 speed_rv: Callable[[], float],
                 pause_rv: Callable[[], float], start_position: Optional[Point] = None):
        self._bounding_box = bounding_box
        self._position_allocator = position_allocation
        self._speed_rv = speed_rv
        self._pause_rv = pause_rv

        # Initialize waypoint
        self._update_waypoint(update_time=0,
                              start_position=start_position)

    def location(self, t: float) -> Point:
        dt = t - self._last_update_t
        t_to_next_waypoint = (self._pb - self._pa).length() / self._vel.length()

        while dt > t_to_next_waypoint + self._pause:
            logging.debug(f'updating waypoint for next location: t={t}, last_update_t={self._last_update_t}, dt={dt}, '
                          f't_to_next_waypoint={t_to_next_waypoint}, pause={self._pause}')
            self._update_waypoint(update_time=self._last_update_t + t_to_next_waypoint + self._pause,
                                  start_position=self._pb)
            dt = t - self._last_update_t
            t_to_next_waypoint = (self._pb - self._pa).length() / self._vel.length()

        if dt <= t_to_next_waypoint:
            # travelling from pa to pb
            return self._pa + dt * self._vel
        elif dt <= t_to_next_waypoint + self._pause:
            # pausing in pb
            return self._pb

        raise RuntimeError(f"RandomWaypointMobilityModel is in an unexpected state: dt={dt}, "
                           f"t_to_next_waypoint={t_to_next_waypoint}, pause={self._pause}")

    def _update_waypoint(self, update_time: float, start_position: Optional[Point] = None):
        if start_position is None:
            self._pa = self._get_random_position()
        else:
            self._pa = start_position
        self._pb = self._get_random_position()

        direction = self._pb - self._pa

        self._vel = direction.normalize() * self._speed_rv()
        self._pause = self._pause_rv()
        self._last_update_t = update_time

        logging.debug(f"update waypoint: pa={self._pa}, pb={self._pb}, velocity={self._vel}, "
                      f"pause={self._pause}, last_update={self._last_update_t}")

    def _get_random_position(self):
        # generate random point
        p = self._position_allocator()

        # if it is not inside the bounding box, generate one more
        while not self._bounding_box.is_inside(p):
            logging.debug(f'{p} is outside of {self._bounding_box}. Sampling new position')
            p = self._position_allocator()

        return p


class WaypointMobilityModel(MobilityModel):
    def __init__(self, positions: Sequence[Point], speeds: Union[Sequence[float], float],
                 pauses: Union[Sequence[float], float]):
        # setup object so that speeds and pauses are lists of length len(postitions)-1
        self._positions = positions

        if isinstance(speeds, collections.abc.Sequence):
            assert len(speeds) == len(self._positions) - 1
            self._speeds = speeds
        elif np.issubdtype(type(speeds), np.float) or np.issubdtype(type(speeds), np.integer):
            self._speeds = [float(speeds)] * (len(self._positions) - 1)

        if isinstance(pauses, collections.abc.Sequence):
            assert len(pauses) == len(self._positions) - 1
            self._pauses = pauses
        elif np.issubdtype(type(pauses), np.float) or np.issubdtype(type(pauses), np.integer):
            self._pauses = [float(pauses)] * (len(self._positions) - 1)

        # Look-up table for start time of a waypoint segment
        segment_duration = [(pb - pa).length() / speed + pause
                            for pa, pb, speed, pause in zip(self._positions[:-1],
                                                            self._positions[1:],
                                                            self._speeds,
                                                            self._pauses)]
        self._start_time_pos = [0.0] + np.cumsum(segment_duration).tolist()
        self._current_pos_idx = 0

    def location(self, t: float) -> Point:
        assert t <= self.max_mobility_duration(), \
            f"Time {t} exceeds the maximum mobility duration ({self._start_time_pos[-1]})"

        # advance to next waypoint segment if necessary
        while t > self._start_time_pos[self._current_pos_idx + 1]:
            self._current_pos_idx += 1

        pa = self._positions[self._current_pos_idx]
        pb = self._positions[self._current_pos_idx + 1]
        speed = self._speeds[self._current_pos_idx]
        pause = self._pauses[self._current_pos_idx]

        dt = t - self._start_time_pos[self._current_pos_idx]
        t_to_next_waypoint = (pb - pa).length() / speed

        if dt < t_to_next_waypoint:
            direction = (pb - pa).normalize()
            return pa + dt * speed * direction
        elif dt <= t_to_next_waypoint + pause:
            return pb

        raise RuntimeError(f"WaypointMobilityModel is in an unexpected state: dt={dt}, "
                           f"t_to_next_waypoint={t_to_next_waypoint}, pause={pause}")

    def max_mobility_duration(self):
        return self._start_time_pos[-1]


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    rwmm = RandomWaypointMobilityModel(bounding_box=Rectangle(0, 0, 10, 10),
                                       position_allocation=PositionAllocation(lambda: np.random.uniform(0, 11),
                                                                              lambda: np.random.uniform(0, 11),
                                                                              lambda: 0),
                                       speed_rv=lambda: np.random.uniform(2, 6),
                                       pause_rv=lambda: np.random.uniform(2, 6),
                                       start_position=Point(0, 0, 0))

    wmm = WaypointMobilityModel([Point(0, 0, 0), Point(10, 0, 0), Point(10, 10, 0), Point(0, 10, 0), Point(0, 0, 0)],
                                speeds=2,
                                pauses=2)

    for time in range(0, 100, 1):
        if time > wmm.max_mobility_duration():
            break
        print(time, wmm.location(time))
