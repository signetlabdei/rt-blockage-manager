# AUTHOR(S):
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
    

@pytest.mark.parametrize("scenario_path",
                         ["scenarios/WorkingScenario1",
                          "scenarios/WorkingScenario2",
                          "scenarios/WorkingScenario3",
                          "scenarios/WorkingScenario4",
                          "scenarios/WorkingScenario5",
                          "scenarios/Indoor1"])
def test_export(scenario_path):
    s1 = QdRealizationScenario(scenario_path)

    out_folder = f"tmp_{datetime.now()}"
    try:
        s1.export(out_folder, precision=6, do_export_mpc_coords=True)
        s2 = QdRealizationScenario(out_folder)
        _compare_scenarios_equal(s1, s2)

        # Test re-write on same output folder multiple times
        s2.export(out_folder, precision=6, do_export_mpc_coords=True)
        s3 = QdRealizationScenario(out_folder)
        _compare_scenarios_equal(s1, s3)
    
    finally:
        shutil.rmtree(out_folder)


def _compare_scenarios_equal(s1: QdRealizationScenario, s2: QdRealizationScenario) -> None:
    # The exported scenario should be the same as the original one
    assert s1._cfg == s2._cfg

    n_nodes1 = s1.get_num_nodes()
    n_nodes2 = s2.get_num_nodes()
    assert n_nodes1 == n_nodes2

    t1 = s1.get_time_steps()
    t2 = s2.get_time_steps()
    assert t1 == t2

    for tx in range(n_nodes1):
        for rx in range(n_nodes1):
            if tx == rx:
                continue

            for t in range(t1):
                ch1 = s1.get_channel(tx, rx, t)
                ch2 = s2.get_channel(tx, rx, t)

                for r1, r2 in zip(ch1, ch2):
                    assert r1 == r2
