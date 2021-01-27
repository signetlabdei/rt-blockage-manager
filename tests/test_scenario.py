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
    assert type(ch) == list  # txs
    assert type(ch[0]) == list  # rxs
    assert ch[0][0] is None  # self-channel
    assert type(ch[0][1]) == list  # time steps
    assert type(ch[0][1][0]) == list  # rays
    assert type(ch[0][1][0][0]) == Ray

    # Fail to request self-channel
    with pytest.raises(AssertionError):
        s.get_channel(0, 0)

    # Get all time steps for node pair
    ch = s.get_channel(0, 1)
    assert type(ch) == list  # time steps
    assert type(ch[0]) == list  # rays
    assert type(ch[0][0]) == Ray

    # Get rays for node pair at given time
    ch = s.get_channel(0, 1, 0)
    assert type(ch) == list  # rays
    assert type(ch[0]) == Ray
