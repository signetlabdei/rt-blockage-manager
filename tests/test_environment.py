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

from src.ray import Ray
from src.geometry import Point
from src.obstacle import OrthoScreenObstacle, SphereObstacle
from src.scenario import QdRealizationScenario
from src.environment import Environment, ObstacleInteraction
from src.mobility_model import ConstantPositionMobilityModel as cpmm
import math
from copy import deepcopy
import pytest
import functools
from src.diffraction_models import dke_kunisch, atan_diffraction, empirical_itu


@pytest.mark.parametrize("obstacle_interactions,asserts", [(ObstacleInteraction.NONE, False),
                                                           (ObstacleInteraction.OBSTRUCTION, False),
                                                           (ObstacleInteraction.REFLECTION, True),
                                                           (ObstacleInteraction.DIFFRACTION, False),
                                                           (ObstacleInteraction.OBSTRUCTION |
                                                           ObstacleInteraction.DIFFRACTION, True),
                                                           (ObstacleInteraction.NONE | ObstacleInteraction.OBSTRUCTION, True)])
def test_environment_obstacle_interaction(obstacle_interactions, asserts):
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')

    if asserts:
        with pytest.raises(AssertionError):
            env = Environment(scenario, [], obstacle_interactions)
    else:
        env = Environment(scenario, [], obstacle_interactions)


def test_environment_no_blockage():
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 2, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]

    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 1  # one ray
    assert ch01[0][0] == original_ray  # the ray did not change


def test_environment_perfect_blockage():
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 0, 0)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]
    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)
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

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)
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

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10) == len(
        original_rays01)  # same number of timesteps
    assert len(ch01) == 11  # 11 timesteps
    for rays in ch01[:3] + ch01[-3:]:
        # one ray from (10,5,0) to (10,3,0) and from (10,-3,0) to (10,-5,0)
        assert len(rays) == 1
    for rays in ch01[3:-3]:
        # Obstruction from (10,2,0) to (10,-2,0)
        assert len(rays) == 0


def test_environment_ws6():
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')
    obstacles = [SphereObstacle(mm=cpmm(Point(5, 3, 1.6)),
                                radius=1.0,
                                reflection_loss=math.inf,
                                transmission_loss=math.inf)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)
    # Just check if corner reflection is properly supperted without throwing any error
    env.process()


def test_environment_diffraction():
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION)
    # Just check if corner reflection is properly supperted without throwing any error
    env.process()


@pytest.mark.parametrize("obstacle_interactions", [(ObstacleInteraction.NONE),
                                                   (ObstacleInteraction.OBSTRUCTION),
                                                   (ObstacleInteraction.DIFFRACTION)])
def test_environment_get_default_interactions(obstacle_interactions):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=obstacle_interactions)

    assert env._get_default_interactions(
        None, None) == obstacle_interactions  # type: ignore


test_rays = [Ray(0, -96, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -93, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -99, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -95, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -92, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -98, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -91, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -100, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -94, 0, [Point(0, 0, 0), Point(1, 1, 1)]),
             Ray(0, -97, 0, [Point(0, 0, 0), Point(1, 1, 1)])]


@pytest.mark.parametrize("ray,n_diffracted_rays,obstacle_interactions",
                         [(test_rays[2], 1, ObstacleInteraction.DIFFRACTION),
                          (test_rays[0], 1, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[0], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[1], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[2], 3, ObstacleInteraction.DIFFRACTION),
                          (test_rays[3], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[4], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[5], 3, ObstacleInteraction.DIFFRACTION),
                          (test_rays[6], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[7], 3, ObstacleInteraction.DIFFRACTION),
                          (test_rays[8], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[9], 3, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[10], 3, ObstacleInteraction.OBSTRUCTION)
                          ])
def test_environment_get_interactions_n_diffracted_rays(ray, n_diffracted_rays, obstacle_interactions):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION)

    assert env._get_interactions_n_diffracted_rays(
        ray, test_rays, n_diffracted_rays) == obstacle_interactions


@pytest.mark.parametrize("ray,diffraction_threshold,obstacle_interactions",
                         [(test_rays[2], 0, ObstacleInteraction.DIFFRACTION),
                          (test_rays[0], 0, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[0], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[1], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[2], 2, ObstacleInteraction.DIFFRACTION),
                          (test_rays[3], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[4], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[5], 2, ObstacleInteraction.DIFFRACTION),
                          (test_rays[6], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[7], 2, ObstacleInteraction.DIFFRACTION),
                          (test_rays[8], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[9], 2, ObstacleInteraction.OBSTRUCTION),
                          (test_rays[10], 2, ObstacleInteraction.OBSTRUCTION)
                          ])
def test_environment_get_interactions_diffraction_threshold(ray, diffraction_threshold, obstacle_interactions):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION)

    assert env._get_interactions_diffraction_threshold(
        ray, test_rays, diffraction_threshold) == obstacle_interactions


@pytest.mark.parametrize("ray,n_reflections_diffraction,expected",
                         [(Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1)]), 0, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1)]), 1, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1)]), 10, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1)]), 0, ObstacleInteraction.OBSTRUCTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1)]), 1, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1)]), 10, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1), Point(0, 1, 0)]), 0, ObstacleInteraction.OBSTRUCTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1), Point(0, 1, 0)]), 1, ObstacleInteraction.OBSTRUCTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1), Point(0, 1, 0)]), 2, ObstacleInteraction.DIFFRACTION),
                          (Ray(0, -90, 0, [Point(0, 0, 0), Point(1, 1, 1), Point(1, 0 ,1), Point(0, 1, 0)]), 10, ObstacleInteraction.DIFFRACTION)
                          ])
def test_environment_get_interactions_n_reflections_diffraction(ray, n_reflections_diffraction, expected):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION)

    assert env._get_interactions_n_reflections_diffraction(
        ray, [], n_reflections_diffraction) == expected


def test_environment_get_interactions_asserts():
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION)  # need DIFFRACTION

    with pytest.raises(AssertionError):
        env._get_interactions_diffraction_threshold(test_rays[0], test_rays, 5)
    with pytest.raises(AssertionError):
        env._get_interactions_n_diffracted_rays(test_rays[0], test_rays, 5)
    with pytest.raises(AssertionError):
        env._get_interactions_n_reflections_diffraction(test_rays[0], test_rays, 5)


@pytest.mark.parametrize("n_diffracted_rays",[0, 1, 1000000])
def test_environment_n_diffracted_rays(n_diffracted_rays):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    # should not raise errors:
    env = Environment(scenario=deepcopy(scenario),
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION,
                      n_diffracted_rays=n_diffracted_rays)  # need DIFFRACTION

    env.process()


@pytest.mark.parametrize("diffraction_threshold",[0, 10, math.inf])
def test_environment_diffraction_threshold(diffraction_threshold):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    # should not raise errors:
    env = Environment(scenario=deepcopy(scenario),
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION,
                      diffraction_threshold=diffraction_threshold)  # need DIFFRACTION

    env.process()


@pytest.mark.parametrize("n_reflections_diffraction",[0, 1, 1000000])
def test_environment_n_reflections_diffraction(n_reflections_diffraction):
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                     width=1,
                                     height=2,
                                     diffraction_loss_model=diffraction_loss_model,
                                     distance_threshold=0)]

    # should not raise errors:
    env = Environment(scenario=deepcopy(scenario),
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION,
                      n_reflections_diffraction=n_reflections_diffraction)  # need DIFFRACTION

    env.process()


def test_environment_interaction_diffraction_raise_error():
    scenario = QdRealizationScenario('scenarios/WorkingScenario6')

    diffraction_loss_model = functools.partial(dke_kunisch, wavelength=1e-9)
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(0, 0, 0)),
                                    width=1,
                                    height=2,
                                    diffraction_loss_model=diffraction_loss_model,
                                    distance_threshold=0)]

    # should raise errors:
    with pytest.raises(AssertionError):
        Environment(scenario=deepcopy(scenario),
                    obstacles=obstacles,
                    obstacle_interactions=ObstacleInteraction.DIFFRACTION,
                    n_diffracted_rays=-1)
    with pytest.raises(AssertionError):
        Environment(scenario=deepcopy(scenario),
            obstacles=obstacles,
            obstacle_interactions=ObstacleInteraction.DIFFRACTION,
            diffraction_threshold=-1)
    with pytest.raises(AssertionError):
        Environment(scenario=deepcopy(scenario),
            obstacles=obstacles,
            obstacle_interactions=ObstacleInteraction.DIFFRACTION,
            n_reflections_diffraction=-1)


@pytest.mark.parametrize("aggregation",
                         ["first",
                          "sum",
                          "max"])
@pytest.mark.parametrize("diffraction_model",
                         [functools.partial(atan_diffraction, wavelength=1e-9),
                         functools.partial(empirical_itu, wavelength=1e-9),
                         functools.partial(dke_kunisch, wavelength=1e-9)])                   
def test_aggregation_no_blockage_diffraction(aggregation, diffraction_model):
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(11, 0, 0)),
                                     width=10,
                                     height=10,
                                     distance_threshold=math.inf,
                                     diffraction_loss_model=diffraction_model)]
    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.DIFFRACTION,
                      aggregate_method=aggregation)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 1  # one ray
    assert ch01[0][0] == original_ray  # the ray did not change


@pytest.mark.parametrize("aggregation",
                         ["first",
                          "sum",
                          "max"])
def test_aggregation_no_blockage_obstruction(aggregation):
    scenario = QdRealizationScenario('scenarios/WorkingScenario1')
    obstacles = [OrthoScreenObstacle(mm=cpmm(Point(11, 0, 0)),
                                     width=10,
                                     height=10)]

    original_ray = deepcopy(scenario.get_channel(0, 1, 0)[0])

    env = Environment(scenario=scenario,
                      obstacles=obstacles,
                      obstacle_interactions=ObstacleInteraction.OBSTRUCTION,
                      aggregate_method=aggregation)
    env.process()

    # check whether the ray exists
    ch01 = env._scenario.get_channel(0, 1)
    ch10 = env._scenario.get_channel(1, 0)

    assert len(ch01) == len(ch10)  # same number of timesteps
    assert len(ch01) == 1  # one timestep
    assert len(ch01[0]) == 1  # one ray
    assert ch01[0][0] == original_ray  # the ray did not change