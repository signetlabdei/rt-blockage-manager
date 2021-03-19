# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.geometry import Point
import numpy as np
from src.scenario import QdRealizationScenario, Scenario
from src.obstacle import Obstacle, SphereObstacle
from src.ray import Ray
from typing import Sequence, List


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
                new_time_steps: List[List[Ray]] = []
                # TODO need to update obstacle mobility at each time step
                for tidx, rays in enumerate(time_steps):
                    t = tidx * self._scenario.get_time_step_duration()

                    new_rays: List[Ray] = [] # new list of rays
                    for ray in rays:
                        new_rays += self._process_ray(ray, t)
                        # TODO when creating diffracted/reflected rays from obstacles, need to check if other obstacles
                        #  obstruct them
                    new_time_steps.append(new_rays) # TODO check if correct
                
                self._scenario.set_channel(tx, rx, new_time_steps)

    def _process_ray(self, ray: Ray, t: float) -> List[Ray]:
        # TODO setup flexible ray processing: choose whether to consider obstructions, reflections, diffractions, diffusion, etc.
        for obs in self._obstacles:
            obs.update(t)
            if obs.obstructs(ray):
                ray.path_gain -= obs.transmission_loss()
        
        if ray.path_gain > -np.inf:
            return [ray]
        else:
            # Automatically remove ray if insignificant
            return []
