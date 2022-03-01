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

from multiprocessing import Pool
import numpy as np
from src import geometry as geom
from src.scenario import Scenario
from src.obstacle import Obstacle
from src.ray import Ray
from typing import Callable, Sequence, List
from enum import Flag, auto
import functools


class ObstacleInteraction(Flag):
    """An Enum class containing types of ray-obstacle interactions
    """
    NONE = auto()
    OBSTRUCTION = auto()
    REFLECTION = auto()
    DIFFRACTION = auto()


class Environment:
    def __init__(self, scenario: Scenario, obstacles: Sequence[Obstacle], obstacle_interactions: ObstacleInteraction, aggregate_method: str="first", n_diffracted_rays: int = None, diffraction_threshold: float = None, n_reflections_diffraction: int = None, absolute_power_threshold: float = None, relative_power_threshold: float = None):
        """Initialize an Environment object
        NOTE: if both n_diffracted_rays, diffraction_threshold, and n_reflections_diffraction are None, the default user-defined obstacle interaction policy will be used
        NOTE: even if obstacle_interactions==ObstacleInteraction.NONE, power threshold will still be performed.

        Args:
            scenario (Scenario): A Scenario object containing the ray-tracing information.
            obstacles (Sequence[Obstacle]): The obstacles that will affect the existing scenario's rays.
            obstacle_interactions (ObstacleInteraction, optional): Interactions between rays and obstacles required by the user. Interactions can be combined using `|`. Defaults to ObstacleInteraction.OBSTRUCTION.
            aggregate_method (str, optional): Aggregation method for interaction between rays and the path segments. Defaults to 'first'.
            absolute_power_threshold (float, optional): All rays with path gain lower that this value will be discarded. Defaults to scenario's get value.
            relative_power_threshold (float, optional): Within a timestep, the most powerful ray is identified. All rays less powerful by this amount will be discarded. Defaults to scenario's get value.
            n_diffracted_rays (int): Number of most powerful rays to process with DIFFRACTION. Less powerful rays will be processed with OBSTRUCTION. Check _get_interactions_n_diffracted_rays for further information.
            diffraction_threshold (float): Threshold with respect to the most powerful ray of a given (tx, rx, timestep) tuple. The most powerful rays will be processed with DIFFRACTION, the less powerful ones will be processed with OBSTRUCTION, instead. Check _get_interactions_diffraction_threshold for further information.
            n_reflections_diffraction (int): If the ray reflection order is at most n_reflections_diffraction, DIFFRACTION is computed, otherwise, only OBSTRUCTION is computed.
        """
        self._scenario = scenario
        self._obstacles = obstacles
        self._obstacle_interactions = obstacle_interactions
        self._get_interactions: Callable[[Ray, List[Ray]], ObstacleInteraction]
        self._aggregate_method = aggregate_method

        assert ObstacleInteraction.REFLECTION not in self._obstacle_interactions, f"Reflection is currently not supported"
        assert not (
            ObstacleInteraction.DIFFRACTION in self._obstacle_interactions and ObstacleInteraction.OBSTRUCTION in self._obstacle_interactions), f"Only one between diffraction and obstruction can be chosen at once, instead, {self._obstacle_interactions=}"
        if ObstacleInteraction.NONE in self._obstacle_interactions:
            assert self._obstacle_interactions == ObstacleInteraction.NONE, f"ObstacleInteraction.NONE cannot be combined with other interactions, instead, {self._obstacle_interactions=}"
        
        # check power thresholds
        if absolute_power_threshold is None:
            absolute_power_threshold = self._scenario.get_absolute_power_threshold()
        assert absolute_power_threshold <= 0, f"Power thresholds should be negative, instead, {absolute_power_threshold=}"
        self.absolute_power_threshold = absolute_power_threshold

        if relative_power_threshold is None:
            relative_power_threshold = self._scenario.get_relative_power_threshold()
        assert relative_power_threshold <= 0, f"Power thresholds should be negative, instead, {relative_power_threshold=}"
        self.relative_power_threshold = relative_power_threshold
        
        # check diffraction interactions
        self._set_diffraction_interactions(n_diffracted_rays, diffraction_threshold, n_reflections_diffraction)


    def _set_diffraction_interactions(self, n_diffracted_rays, diffraction_threshold, n_reflections_diffraction):
        assert len([x
            for x in [n_diffracted_rays, diffraction_threshold, n_reflections_diffraction]
            if x is not None]) <= 1,\
        f"At most one of the following arguments should be not None, instead " \
        f"{n_diffracted_rays=}, {diffraction_threshold=}, {n_reflections_diffraction=}"

        if n_diffracted_rays is not None:
            assert n_diffracted_rays >= 0, f"{n_diffracted_rays=} should be non-negative"
            self._get_interactions = functools.partial(self._get_interactions_n_diffracted_rays,
                                                       n_diffracted_rays=n_diffracted_rays)

        elif diffraction_threshold is not None:
            assert diffraction_threshold >= 0, f"{diffraction_threshold=} should be non-negative"
            self._get_interactions = functools.partial(self._get_interactions_diffraction_threshold,
                                                       diffraction_threshold=diffraction_threshold)

        elif n_reflections_diffraction is not None:
            assert n_reflections_diffraction >= 0, f"{n_reflections_diffraction=} should be non-negative"
            self._get_interactions = functools.partial(self._get_interactions_n_reflections_diffraction,
                                                       n_reflections_diffraction=n_reflections_diffraction)

        else:
            self._get_interactions = self._get_default_interactions


    def process(self, n_workers: int = 1):
        """Process the scenario with the given obstacles.
        NOTE: the original scenario object will be modified directly.

        Args:
            n_workers (int): Number of CPU nodes to be used for parallel processing. Warning (1/2): using all CPU nodes may freeze other applications. Warning (2/2): using multiple CPU nodes may result in OOM error.
        """
        
        # check number of workers
        assert n_workers>0, f"The number of workers for parallel computing should be a positive integer, instead {n_workers=}"

        # For each node pair
        for tx in range(self._scenario.get_num_nodes()):
            for rx in range(self._scenario.get_num_nodes()):
                if tx == rx:
                    # Ignore self-channel
                    continue
                
                time_steps = self._scenario.get_channel(tx, rx)

                new_time_steps: List[List[Ray]] = []
                if n_workers==1:
                    for tidx, rays in enumerate(time_steps):
                        rays = self._process_rays(rays, tidx * self._scenario.get_time_step_duration())
                        new_time_steps.append(rays)
                else:
                    pool = Pool(processes=n_workers)
                    parallel_res = pool.starmap_async(self._process_rays,\
                            [(rays,tidx*self._scenario.get_time_step_duration()) for tidx, rays in enumerate(time_steps)],\
                                chunksize=len(time_steps)//n_workers) 
                    
                    pool.close()
                    pool.join()
                    new_list = parallel_res.get()
                    [new_time_steps.append(r[0]) for r in new_list]
                self._scenario.set_channel(tx, rx, new_time_steps)


    def _process_rays(self, rays: List[Ray], t: float) -> List[Ray]:
        new_rays: List[Ray] = []  # new list of rays
        for ray in rays:
            interactions = self._get_interactions(ray, rays)
            new_rays += self._process_ray(ray,
                                        t,
                                        interactions)

        # Apply relative threshold
        if len(new_rays) > 0:
            max_gain = max([r.path_gain for r in new_rays])
            new_thr_rays = [
                r for r in new_rays if r.path_gain >= max_gain + self.relative_power_threshold]

        else:
            new_thr_rays = []

        return new_thr_rays
      

    def _process_ray(self, ray: Ray, t: float, interactions: ObstacleInteraction) -> List[Ray]:
        """Process a single ray from the scenario with the current obstacle configuration

        Args:
            ray (Ray): A ray from the scenario
            t (float): The time at which the ray is processed. This is needed to update the mobility models of the obstacles absolute_power_threshold (float): All rays with lower path gain will be discarded
            interactions (ObstacleInteractions): Interactions to compute for the given ray

        Returns:
            List[Ray]: The corresponding list of rays. Note that a single ray may be changed, removed (returning an empty list) or separated into multiple rays
        """
        # TODO when creating diffracted/reflected rays from obstacles, need to check if other obstacles obstruct them
        for obs in self._obstacles:
            obs.update(t)
            losses: List[float] = []  # losses due to the interaction of ray path(s) with the obstacle
            for p1, p2 in zip(ray.vertices[:-1], ray.vertices[1:]):
                if abs(geom.distance(p1, p2))<1e-9:
                    # Support corner case: check
                    # https://github.com/signetlabdei/rt-blocakge-manager/issues/2
                    # for more information
                    continue
                ray_segment = geom.Segment(p1,p2)

                if obs.obstructs(ray_segment):
                    if (ObstacleInteraction.OBSTRUCTION in interactions):
                        losses.append(obs.transmission_loss())

                    elif (ObstacleInteraction.DIFFRACTION in interactions):
                        losses.append(obs.diffraction_loss(ray_segment))
                    
                    else:
                        raise NotImplementedError(f"{ObstacleInteraction} not implemented.")
                    
                    if self._aggregate_method=="first":
                        break
            
            if len(losses) == 0:
                obst_loss = 0.0
            else:
                if self._aggregate_method=="first":
                    assert len(losses)==1, "'first' aggregation method should compute the loss only for the first interaction of the obstacle with the path segment."
                    obst_loss = losses[0]
                elif self._aggregate_method=="sum":
                    obst_loss = sum(losses)
                elif self._aggregate_method=="max":
                    obst_loss = max(losses)
                else:
                    raise NotImplementedError(f"{self._aggregate_method} not implemented.")
        
            ray.path_gain -= obst_loss

        if ray.path_gain > self.absolute_power_threshold:
            return [ray]
        else:
            # Automatically remove ray if below threshold
            return []

    def _get_default_interactions(self, ray: Ray, current_rays: List[Ray]) -> ObstacleInteraction:
        """Return the ObstacleInteraction as set by the user.
        Arguments are ignored

        Args:
            ray (Ray): ignored
            current_rays (List[Ray]): ignored

        Returns:
            ObstacleInteraction: the ObstacleInteraction set by the user
        """
        return self._obstacle_interactions

    def _get_interactions_n_diffracted_rays(self, ray: Ray, current_rays: List[Ray], n_diffracted_rays: int) -> ObstacleInteraction:
        """If the given ray is among the n_diffracted_rays most powerful rays in current_rays, then keep the DIFFRACTION interaction, otherwise replace DIFFRACTION with OBSTRUCTION.

        NOTE: the function fails if DIFFRACTION is not part of the obstacle interactions.
        NOTE: if the processing modifies current_rays, the set of most powerful rays might change between successive calls of this function. For this reason, more rays than expected might be processed with DIFFRACTION.

        Args:
            ray (Ray): The ray to process
            current_rays (List[Ray]): The list of rays for the current timestep, tx, and rx
            n_diffracted_rays (int): How many most powerful rays should be processed with DIFFRACTION

        Returns:
            ObstacleInteraction: The updated interactions
        """
        assert ObstacleInteraction.DIFFRACTION in self._obstacle_interactions
        n_diffracted_rays = min(n_diffracted_rays, len(current_rays))

        current_rays_powers = [r.path_gain for r in current_rays]
        # find idx of most powerful rays
        # TODO current rays' powers are updated at each step, thus more than n_diffracted_rays could be processed
        idxs = np.argpartition(current_rays_powers, -
                               n_diffracted_rays)[-n_diffracted_rays:]
        n_current_rays = [current_rays[idx] for idx in idxs]

        if ray in n_current_rays:
            # keep diffraction interaction
            return self._obstacle_interactions
        else:
            # replace diffraction with obstruction
            return (self._obstacle_interactions ^ ObstacleInteraction.DIFFRACTION) | ObstacleInteraction.OBSTRUCTION

    def _get_interactions_diffraction_threshold(self, ray: Ray, current_rays: List[Ray], diffraction_threshold: float) -> ObstacleInteraction:
        """If the given ray's path gain is within diffraction_threshold of the most powerful ray in current_rays, then keep the DIFFRACTION interaction, otherwise replace DIFFRACTION with OBSTRUCTION.

        NOTE: the function fails if DIFFRACTION is not part of the obstacle interactions.
        NOTE: if the processing modifies current_rays, the set of most powerful rays might change between successive calls of this function. For this reason, more rays than expected might be processed with DIFFRACTION. This means that the thresholding operation is dynamic: if the most powerful ray is attenuated, the new most powerful ray might change.

        Args:
            ray (Ray): The ray to process
            current_rays (List[Ray]): The list of rays for the current timestep, tx, and rx
            diffraction_threshold (float): The threshold with respect to the most powerful ray in current_rays [dB]

        Returns:
            ObstacleInteraction: The updated interactions
        """
        assert ObstacleInteraction.DIFFRACTION in self._obstacle_interactions

        max_power = max([r.path_gain for r in current_rays])

        if ray.path_gain >= max_power - diffraction_threshold:
            # keep diffraction interaction
            return self._obstacle_interactions
        else:
            # replace diffraction with obstruction
            return (self._obstacle_interactions ^ ObstacleInteraction.DIFFRACTION) | ObstacleInteraction.OBSTRUCTION

    def _get_interactions_n_reflections_diffraction(self, ray: Ray, current_rays: List[Ray], n_reflections_diffraction: int) -> ObstacleInteraction:
        """If the ray reflection order is at most n_reflections_diffraction, DIFFRACTION is computed, otherwise, DIFFRACTION is replaced with OBSTRUCTION.

        NOTE: the function fails if DIFFRACTION is not part of the obstacle interactions.

        Args:
            ray (Ray): The ray to process
            current_rays (List[Ray]): unused
            n_reflections_diffraction (int): The maximum reflection order where diffraction is computed.

        Returns:
            ObstacleInteraction: The updated interactions
        """
        assert ObstacleInteraction.DIFFRACTION in self._obstacle_interactions

        if ray.refl_order() <= n_reflections_diffraction:
            # keep diffraction interaction
            return self._obstacle_interactions
        else:
            # replace diffraction with obstruction
            return (self._obstacle_interactions ^ ObstacleInteraction.DIFFRACTION) | ObstacleInteraction.OBSTRUCTION
