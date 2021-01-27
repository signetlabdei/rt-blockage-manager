# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.qd_realization_io import import_qd_file, import_mpc_coordinates, import_rays
from src.qd_realization_io import import_parameter_configuration, import_scenario
from src.geometry import Point
import os


def test_import_qd_file():
    scenario_path = "scenarios/WorkingScenario1"
    qd_file_name = "Tx0Rx1.txt"

    qd_file = import_qd_file(os.path.join(scenario_path, "Output/Ns3/QdFiles", qd_file_name))

    # The chosen scenario should have 1 time step
    assert len(qd_file) == 1

    for k, v in qd_file[0].items():
        # The chosen scenario should only have one ray
        if k == 'n_rays':
            assert v == 1
        else:
            assert len(v) == 1


def test_import_mpc_coordinates():
    scenario_path = "scenarios/WorkingScenario1"
    mpc_coord_name = "MpcTx0Rx1Refl0Trc0.csv"

    mpc_coords = import_mpc_coordinates(os.path.join(scenario_path, "Output/Visualizer/MpcCoordinates", mpc_coord_name))

    # The chosen scenario should have 1 ray
    assert len(mpc_coords) == 1
    # The ray should be direct (i.e., only tx and rx coordinates)
    assert len(mpc_coords[0]) == 2
    # The coordinates should be Points
    assert all([type(x) == Point for x in mpc_coords[0]])


def test_import_rays():
    scenario_path = "scenarios/WorkingScenario1"
    max_refl = 0
    tx = 0
    rx = 1

    rays = import_rays(scenario_path, max_refl, tx, rx)

    # The scenario should have 1 time step
    assert len(rays) == 1
    # The scenario should have only 1 ray
    assert len(rays[0]) == 1
    # The scenario should have only the direct ray
    assert rays[0][0].is_direct()


def test_import_parameter_configuration():
    scenario_path = "scenarios/Indoor1"
    cfg = import_parameter_configuration(scenario_path)

    ground_truth = {'environmentFileName': 'Box.xml',
                    'generalizedScenario': 0,
                    'indoorSwitch': 1,
                    'mobilitySwitch': 0,
                    'mobilityType': 1,
                    'numberOfNodes': 3,
                    'numberOfTimeDivisions': 1,
                    'referencePoint': '[0,0,0]',
                    'selectPlanesByDist': 0,
                    'qdGeneratorType': 'off',
                    'switchRandomization': 1,
                    'totalNumberOfReflections': 2,
                    'totalTimeDuration': 0,
                    'switchSaveVisualizerFiles': 1,
                    'carrierFrequency': 60e9,
                    'qdFilesFloatPrecision': 6,
                    'useOptimizedOutputToFile': 1,
                    'minAbsolutePathGainThreshold': float('-inf'),
                    'minRelativePathGainThreshold': float('-inf'),
                    'materialLibraryPath': 'material_libraries/LectureRoomAllMaterialsMetalFloor.csv'}

    assert cfg == ground_truth


def test_import_working_scenario_1():
    scenario_path = "scenarios/WorkingScenario1"
    channels = import_scenario(scenario_path)

    # The scenario has 2 nodes
    assert len(channels) == 2
    assert len(channels[0]) == 2
    assert len(channels[1]) == 2

    # The channel has 1 time step
    assert len(channels[0][1]) == 1
    assert len(channels[1][0]) == 1

    # The channel has 1 ray
    assert len(channels[0][1][0]) == 1
    assert len(channels[1][0][0]) == 1

    # The ray is direct
    assert channels[0][1][0][0].is_direct()
    assert channels[1][0][0][0].is_direct()


def test_import_working_scenario_2():
    scenario_path = "scenarios/WorkingScenario2"
    channels = import_scenario(scenario_path)

    # The scenario has 2 nodes
    assert len(channels) == 2
    assert len(channels[0]) == 2
    assert len(channels[1]) == 2

    # The channel has 1 time step
    assert len(channels[0][1]) == 1
    assert len(channels[1][0]) == 1

    # The channel has 2 ray
    assert len(channels[0][1][0]) == 2
    assert len(channels[1][0][0]) == 2

    # The first ray is direct
    assert channels[0][1][0][0].is_direct()
    assert channels[1][0][0][0].is_direct()

    # The second ray is a first order reflection
    assert channels[0][1][0][1].refl_order() == 1
    assert channels[1][0][0][1].refl_order() == 1


def test_import_working_scenario_3():
    scenario_path = "scenarios/WorkingScenario3"
    channels = import_scenario(scenario_path)

    # The scenario has 2 nodes
    assert len(channels) == 2
    assert len(channels[0]) == 2
    assert len(channels[1]) == 2

    # The channel has 1 time step
    assert len(channels[0][1]) == 1
    assert len(channels[1][0]) == 1

    # The channel has 3 ray
    assert len(channels[0][1][0]) == 3
    assert len(channels[1][0][0]) == 3

    # The first ray is direct
    assert channels[0][1][0][0].is_direct()
    assert channels[1][0][0][0].is_direct()

    # The second ray is a first order reflection
    assert channels[0][1][0][1].refl_order() == 1
    assert channels[1][0][0][1].refl_order() == 1

    # The third ray is a first order reflection
    assert channels[0][1][0][2].refl_order() == 1
    assert channels[1][0][0][2].refl_order() == 1


def test_import_working_scenario_4():
    scenario_path = "scenarios/WorkingScenario4"
    channels = import_scenario(scenario_path)

    # The scenario has 2 nodes
    assert len(channels) == 2
    assert len(channels[0]) == 2
    assert len(channels[1]) == 2

    # The channel has 5 time step
    assert len(channels[0][1]) == 5
    assert len(channels[1][0]) == 5

    # The channel has 3 ray
    assert [len(channels[0][1][t]) for t in range(5)] == [2, 1, 0, 0, 1]
    assert [len(channels[1][0][t]) for t in range(5)] == [2, 1, 0, 0, 1]


def test_import_indoor1():
    scenario_path = "scenarios/Indoor1"

    channels = import_scenario(scenario_path)
    cfg = import_parameter_configuration(scenario_path)

    # Check nodes
    n_nodes = cfg['numberOfNodes']
    assert len(channels) == n_nodes
    assert len(channels[0]) == n_nodes
    assert len(channels[1]) == n_nodes

    # Check time steps
    time_steps = cfg['numberOfTimeDivisions']
    assert len(channels[0][1]) == time_steps
    assert len(channels[1][0]) == time_steps
