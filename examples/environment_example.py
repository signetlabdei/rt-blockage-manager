import functools
import math
from os import path

import src.utils as utils
from src.diffraction_models import empirical_itu
from src.environment import Environment, ObstacleInteraction
from src.geometry import Point, Vector
from src.mobility_model import ConstantVelocityMobilityModel as cvmm
from src.obstacle import OrthoScreenObstacle
from src.scenario import QdRealizationScenario


def main():
    ## Scenario params
    scenario_folder = "scenarios/HumanPresence"
    scenario_out = path.join(scenario_folder, "BlockageOut")
    obstacle_interactions = ObstacleInteraction.DIFFRACTION

    scenario = QdRealizationScenario(scenario_folder)

    ## Obstacle params
    start_pos = Point(5, 0, 0)
    speed = 1.2  # [m/s]
    vel = Vector(0, speed, 0)

    width = 0.4  # shoulder span [m]
    height = 1.7  # average human height [m]

    distance_threshold = math.inf  # [m]
    wavelength = utils.get_wavelength(scenario.get_frequency())

    # Setting up a diffraction loss model. The wavelength parameter is statically set to provide a function with the right signature to the obstacle
    diffraction_loss_model = functools.partial(
        empirical_itu, wavelength=wavelength)

    mm = cvmm(start_pos=start_pos, vel=vel)
    obs = OrthoScreenObstacle(mm=mm,
                              width=width,
                              height=height,
                              diffraction_loss_model=diffraction_loss_model,
                              distance_threshold=distance_threshold)

    ## Setup and process
    env = Environment(scenario=scenario,
                      obstacles=[obs],
                      obstacle_interactions=obstacle_interactions)
    env.process()

    out_scenario = env._scenario
    out_scenario.export(scenario_out, do_copy_unnecessary_files=True)

    print(f"Processed scenario exported to '{path.abspath(scenario_out)}'")


if __name__ == "__main__":
    main()
