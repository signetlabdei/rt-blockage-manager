# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from src.ray import Ray
from src.geometry import Point
from src.obstacle import SphereObstacle
from src.scenario import QdRealizationScenario
from src.environment import Environment
from src.mobility_model import ConstantPositionMobilityModel as cpmm
import math
from copy import deepcopy


def test_environment_no_blockage():
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 2, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]

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
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 0, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]
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
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 0, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=transmission_loss)]

    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])
    assert(isinstance(original_ray, Ray))

    env = Environment(scenario=scenario, obstacles=obstacles)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 1  # one ray
    assert ch01[0][0].path_gain == original_ray.path_gain - transmission_loss


def test_environment_ws5():
    scenario = QdRealizationScenario('scenarios/WorkingScenario5')
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 0, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]

    original_rays01 = deepcopy(scenario.get_channel(0, 1))

    env = Environment(scenario=scenario, obstacles=obstacles)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10) == len(original_rays01)  # same number of timesteps
    assert len(ch01) == 11  # 11 timesteps
    for rays in ch01[:3] + ch01[-3:]:
        # one ray from (10,5,0) to (10,3,0) and from (10,-3,0) to (10,-5,0)
        assert len(rays) == 1
    for rays in ch01[3:-3]:
        # Obstruction from (10,2,0) to (10,-2,0)
        assert len(rays) == 0
