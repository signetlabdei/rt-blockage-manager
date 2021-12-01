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

from src.scenario import QdRealizationScenario
from src.ray import Ray
import pytest
import os
from datetime import datetime
import shutil


@pytest.mark.parametrize("scenario_path,nodes,duration,freq,time_steps,time_step_duration",
                         [('scenarios/WorkingScenario1', 2, 1, 60e9, 1, 1),
                          ('scenarios/WorkingScenario2', 2, 1, 60e9, 1, 1),
                          ('scenarios/WorkingScenario3', 2, 1, 60e9, 1, 1),
                          ('scenarios/WorkingScenario4', 2, 1, 60e9, 5, 0.2),
                          ('scenarios/HumanPresence', 2, 5.1, 60e9, 1500, 3.4*10**-3),
                          ('scenarios/Indoor1', 3, 0, 60e9, 1, 0)])
def test_qd_realization_scenario(scenario_path, nodes, duration, freq, time_steps, time_step_duration):
    s = QdRealizationScenario(scenario_path)

    assert s.get_num_nodes() == nodes
    assert s.get_scenario_duration() == duration
    assert s.get_frequency() == freq
    assert s.get_time_steps() == time_steps
    assert s.get_time_step_duration() == time_step_duration


def test_get_channel():
    s = QdRealizationScenario('scenarios/WorkingScenario4')

    # Get all channels
    ch = s.get_channel()
    assert isinstance(ch, list)  # txs
    assert isinstance(ch[0], list)  # rxs
    assert ch[0][0] is None  # self-channel
    assert isinstance(ch[0][1], list)  # time steps
    assert isinstance(ch[0][1][0], list)  # rays
    assert isinstance(ch[0][1][0][0], Ray)

    # Fail to request self-channel
    with pytest.raises(AssertionError):
        s.get_channel(0, 0)

    # Get all time steps for node pair
    ch = s.get_channel(0, 1)
    assert isinstance(ch, list)  # time steps
    assert isinstance(ch[0], list)  # rays
    assert isinstance(ch[0][0], Ray)

    # Get rays for node pair at given time
    ch = s.get_channel(0, 1, 0)
    assert isinstance(ch, list)  # rays
    assert isinstance(ch[0], Ray)


def test_scenario_eq():
    scenario_path = "scenarios/WorkingScenario1"
    s1 = QdRealizationScenario(scenario_path)
    s2 = QdRealizationScenario(scenario_path)
    assert s1 == s2

    ch = s2.get_channel(0, 1)
    ch[0][0].path_gain += 10
    s2.set_channel(0, 1, ch)

    assert s1 != s2

    s2 = QdRealizationScenario("scenarios/WorkingScenario2")
    assert s1 != s2

    # test same scenario, different _other_files
    out_folder = f"tmp_{datetime.now()}"
    try:
        shutil.copytree(scenario_path, out_folder)
        open(os.path.join(out_folder, 'tmpfile'), 'w').close()

        s2 = QdRealizationScenario(out_folder)
        assert s1 != s2
    finally:
        shutil.rmtree(out_folder)

    
    with pytest.raises(TypeError):
        s1 == 2
    

@pytest.mark.parametrize("scenario_path",
                         ["scenarios/WorkingScenario1",
                          "scenarios/WorkingScenario2",
                          "scenarios/WorkingScenario3",
                          "scenarios/WorkingScenario4",
                          "scenarios/WorkingScenario5",
                          "scenarios/WorkingScenario6",
                          "scenarios/HumanPresence",
                          "scenarios/Indoor1"])
def test_basic_export(scenario_path):
    s1 = QdRealizationScenario(scenario_path)

    out_folder = f"tmp_{datetime.now()}"
    try:
        s1.export(out_folder, precision=6, do_export_mpc_coords=True)
        s2 = QdRealizationScenario(out_folder)
        assert s1 == s2

        # Test re-write on same output folder multiple times
        s2.export(out_folder, precision=6, do_export_mpc_coords=True)
        s3 = QdRealizationScenario(out_folder)
        assert s1 == s3

    finally:
        shutil.rmtree(out_folder)


@pytest.mark.parametrize("scenario_path",
                         ["scenarios/WorkingScenario1",
                          "scenarios/WorkingScenario2",
                          "scenarios/WorkingScenario3",
                          "scenarios/WorkingScenario4",
                          "scenarios/WorkingScenario5",
                          "scenarios/WorkingScenario6",
                          "scenarios/HumanPresence",
                          "scenarios/Indoor1"])
def test_export_copy_other_files(scenario_path):
    s = QdRealizationScenario(scenario_path)

    out_folder = f"tmp_{datetime.now()}"
    try:
        s.export(out_folder, precision=6,
                 do_export_mpc_coords=True,
                 do_copy_unnecessary_files=True)
        os.path.exists(os.path.join(out_folder, "scenario"))

    finally:
        shutil.rmtree(out_folder)


def test_export_overwrite_other_files():
    s = QdRealizationScenario("scenarios/WorkingScenario1")

    out_folder = f"tmp_{datetime.now()}"

    # create ficticious colliding file
    os.makedirs(out_folder, exist_ok=False)
    with open(os.path.join(out_folder, "scenario"), 'wt') as f:
        f.write("ficticious!")

    try:
        s.export(out_folder, precision=6,
                 do_export_mpc_coords=True,
                 do_copy_unnecessary_files=True)

        assert os.path.exists(os.path.join(out_folder, "scenario"))

        with open(os.path.join(out_folder, "scenario"), 'rt') as f:
            line = f.readline()
            assert line != "ficticious!"

    finally:
        shutil.rmtree(out_folder)


@pytest.mark.parametrize("scenario_path",
                         ["scenarios/WorkingScenario1",
                          "scenarios/WorkingScenario2",
                          "scenarios/WorkingScenario3",
                          "scenarios/WorkingScenario4",
                          "scenarios/WorkingScenario5",
                          "scenarios/WorkingScenario6",
                          "scenarios/HumanPresence",
                          "scenarios/Indoor1"])
def test_export_identical_to_input(scenario_path):
    s1 = QdRealizationScenario(scenario_path)

    out_folder = f"tmp_{datetime.now()}"
    try:
        s1.export(out_folder, precision=6,
                  do_export_mpc_coords=True,
                  do_copy_unnecessary_files=True)

        s2 = QdRealizationScenario(out_folder)

        assert s1 == s2

    finally:
        shutil.rmtree(out_folder)
