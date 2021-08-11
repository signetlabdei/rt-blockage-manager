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
from typing import Sequence, List
from enum import Flag, auto

class ObstacleInteraction(Flag):
    """An Enum class containing types of ray-obstacle interactions
    """
    NONE = auto()
    OBSTRUCTION = auto()
    REFLECTION = auto()
    DIFFRACTION = auto()

class Environment:
    def __init__(self, scenario: Scenario, obstacles: Sequence[Obstacle], obstacle_interactions: ObstacleInteraction=ObstacleInteraction.OBSTRUCTION):
        """Initialize an Environment object

        Args:
            scenario (Scenario): A Scenario object containing the ray-tracing information
            obstacles (Sequence[Obstacle]): The obstacles that will affect the existing scenario's rays
            obstacle_interactions (ObstacleInteraction, optional): Interactions between rays and obstacles required by the user. Interactions can be combined using `|`. Defaults to ObstacleInteraction.OBSTRUCTION.
        """
        self._scenario = scenario
        self._obstacles = obstacles
        self._obstacle_interactions = obstacle_interactions

        assert ObstacleInteraction.REFLECTION not in self._obstacle_interactions, f"Reflection is currently not supported"
        assert ObstacleInteraction.DIFFRACTION not in self._obstacle_interactions, f"Diffraction is currently not supported, instead, {self._obstacle_interactions=}"
        assert not (ObstacleInteraction.DIFFRACTION in self._obstacle_interactions and ObstacleInteraction.OBSTRUCTION in self._obstacle_interactions), f"Only one between diffraction and obstruction can be chosen at once, instead, {self._obstacle_interactions=}"
        if ObstacleInteraction.NONE in self._obstacle_interactions:
            assert self._obstacle_interactions == ObstacleInteraction.NONE, f"ObstacleInteraction.NONE cannot be combined with other interactions, instead, {self._obstacle_interactions=}"

    def process(self, absolute_power_threshold: float=None, relative_power_threshold: float=None):
        """Process the scenario with the given obstacles.
        Note 1: the original scenario object will be modified directly.
        Note 2: even if obstacle_interactions==ObstacleInteraction.NONE, power threshold will still be performed.

        Args:
            absolute_power_threshold (float, optional): All rays with path gain lower that this value will be discarded. Defaults to scenario's get value.
            relative_power_threshold (float, optional): Within a timestep, the most powerful ray is identified. All rays less powerful by this amount will be discarded. Defaults to scenario's get value.
        """
        if absolute_power_threshold is None:
            absolute_power_threshold = self._scenario.get_absolute_power_threshold()
        assert absolute_power_threshold <= 0, f"Power thresholds should be negative, instead, {absolute_power_threshold=}"

        if relative_power_threshold is None:
            relative_power_threshold = self._scenario.get_relative_power_threshold()
        assert relative_power_threshold <= 0, f"Power thresholds should be negative, instead, {relative_power_threshold=}"

        # For each node pair
        for tx in range(self._scenario.get_num_nodes()):
            for rx in range(self._scenario.get_num_nodes()):
                if tx == rx:
                    # Ignore self-channel
                    continue

                time_steps = self._scenario.get_channel(tx, rx)
                new_time_steps: List[List[Ray]] = []
                for tidx, rays in enumerate(time_steps):
                    t = tidx * self._scenario.get_time_step_duration()

                    new_rays: List[Ray] = [] # new list of rays
                    for ray in rays:
                        new_rays += self._process_ray(ray, t, absolute_power_threshold)
                    
                    # Apply relative threshold
                    if len(new_rays) > 0:
                        max_gain = max([r.path_gain for r in new_rays])
                        new_thr_rays = [r for r in new_rays if r.path_gain >= max_gain + relative_power_threshold]

                    else:
                        new_thr_rays = []

                    new_time_steps.append(new_thr_rays)
                
                self._scenario.set_channel(tx, rx, new_time_steps)

    def _process_ray(self, ray: Ray, t: float, absolute_power_threshold: float) -> List[Ray]:
        """Process a single ray from the scenario with the current obstacle configuration

        Args:
            ray (Ray): A ray from the scenario
            t (float): The time at which the ray is processed. This is needed to update the mobility models of the obstacles
            absolute_power_threshold (float): All rays with lower path gain will be discarded

        Returns:
            List[Ray]: The corresponding list of rays. Note that a single ray may be changed, removed (returning an empty list) or separated into multiple rays
        """
        # TODO setup flexible ray processing: choose whether to consider obstructions, reflections, diffractions, diffusion, etc.
        # TODO when creating diffracted/reflected rays from obstacles, need to check if other obstacles obstruct them
        for obs in self._obstacles:
            obs.update(t)

            if (ObstacleInteraction.OBSTRUCTION in self._obstacle_interactions) and obs.obstructs(ray):
                ray.path_gain -= obs.transmission_loss()
        
        if ray.path_gain > absolute_power_threshold:
            return [ray]
        else:
            # Automatically remove ray if below threshold
            return []
