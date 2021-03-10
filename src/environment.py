# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.scenario import Scenario
from src.obstacle import Obstacle
from src.ray import Ray
from typing import Sequence


class Environment:
    def __init__(self, scenario: Scenario, obstacles: Sequence[Obstacle]):
        self._scenario = scenario
        self._obstacles = obstacles

    def process(self):
        # For each time step
        for tx in range(self._scenario.get_num_nodes()):
            for rx in range(self._scenario.get_num_nodes()):
                if tx == rx:
                    # Ignore self-channel
                    continue

                time_steps = self._scenario.get_channel(tx, rx)
                # TODO need to update obstacle mobility at each time step
                for rays in time_steps:
                    for ray in rays:
                        self._process_ray(ray)  # TODO is this modified (deleted) in-place?
                        # TODO when creating diffracted/reflected rays from obstacles, need to check if other obstacles
                        #  obstruct them

    def _process_ray(self, ray: Ray):
        for obs in self._obstacles:
            if obs.obstructs(ray):
                ray.path_gain -= obs.transmission_loss()
