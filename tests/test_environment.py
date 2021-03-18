# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from src.geometry import Point
from src.obstacle import SphereObstacle
from src.scenario import QdRealizationScenario
from src.environment import Environment
import pytest
import numpy as np
from copy import deepcopy


def test_environment_no_blockage():
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(center=Point(5, 2, 0),
                                radius=1.0,
                                reflection_loss=np.inf,
                                transmission_loss=np.inf)]

    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])

    env = Environment(scenario=scenario, obstacles=obstacles)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10) # same number of timesteps
    assert len(ch01) == 1 # one timestep
    assert len(ch01[0]) == 1 # one ray
    assert ch01[0][0] == original_ray # the ray did not changed


def test_environment_perfect_blockage():
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(center=Point(5, 0, 0),
                                radius=1.0,
                                reflection_loss=np.inf,
                                transmission_loss=np.inf)]
    env = Environment(scenario=scenario, obstacles=obstacles)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 0  # one ray


def test_environment_imprefect_blockage():
    transmission_loss = 30

    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(center=Point(5, 0, 0),
                                radius=1.0,
                                reflection_loss=np.inf,
                                transmission_loss=transmission_loss)]

    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])

    env = Environment(scenario=scenario, obstacles=obstacles)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 1  # one ray
    assert ch01[0][0].path_gain == original_ray.path_gain - transmission_loss
