# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from abc import ABC, abstractmethod
import math
from typing import Optional, Union, List, overload
import src.qd_realization_io as qdio
from src.ray import Ray


class Scenario(ABC):
    @abstractmethod
    def get_num_nodes(self) -> int:
        """
        Return the number of nodes in the scenario
        """

    @abstractmethod
    def get_time_steps(self) -> int:
        """
        Return the number of time steps of the scenario
        """

    @abstractmethod
    def get_scenario_duration(self) -> float:
        """
        Return the duration of the scenario [s]
        """

    @abstractmethod
    def get_time_step_duration(self) -> float:
        """
        Return the duration of the time steps of the scenario [s]
        """

    @abstractmethod
    def get_frequency(self) -> float:
        """
        Return the operating frequency of the scenario
        """

    @overload
    def get_channel(self) -> List[List[Optional[List[List[Ray]]]]]:
        ...

    @overload
    def get_channel(self, tx: int, rx: int) -> List[List[Ray]]:
        ...

    @overload
    def get_channel(self, tx: int, rx: int, t: int) -> List[Ray]:
        ...

    @abstractmethod
    def get_channel(self, tx: Optional[int] = None, rx: Optional[int] = None, t: Optional[int] = None) -> Union[
        List[List[Optional[List[List[Ray]]]]], List[List[Ray]], List[Ray]]:
        """
        Return a channel structure.
        Variants:
        - (self: Scenario) -> List[List[Optional[List[List[Ray]]]]]:
        Return the full channel structure.
        That is, a list matrix (list of list) of dimension (num_nodes x num_nodes).
        Each off-diagonal element is a list (time step) of lists of Ray.
        Elements on the main diagonal are not valid (no self-channel).
        - (self: Scenario, tx: int, rx: int) -> List[List[Ray]]:
        Return all time steps of the channel between the two valid nodes.
        That is, a list of lists of Ray.
        - (self: Scenario, tx: int, rx: int, t: int) -> List[Ray]:
        Return the list of Ray for the requested channel instance.
        """


class QdRealizationScenario(Scenario):
    def __init__(self, scenario_path: str):
        self._scenario_path = scenario_path

        # import qd-realization scenario
        self._channel = qdio.import_scenario(self._scenario_path)
        self._cfg = qdio.import_parameter_configuration(self._scenario_path)

        self._assess_scenario_validity()

    def _assess_scenario_validity(self):
        n_nodes = self._cfg['numberOfNodes']
        time_steps = self._cfg['numberOfTimeDivisions']

        # Check number of nodes (num rows)
        assert len(self._channel) == n_nodes, \
            f"Invalid number of nodes: declared {n_nodes}, found {len(self._channel)} tx (dim 0)"
        # Check number of nodes (num cols for each row)
        for rxs in self._channel:
            assert len(rxs) == n_nodes, \
                f"Invalid number of nodes: declared {n_nodes}, found {len(self._channel)} rx (dim 1)"

        # Check number of time steps for each node pair
        for tx, row in enumerate(self._channel):
            for rx, channel in enumerate(row):
                if tx == rx:
                    # Ignore elements on main diagonal
                    continue

                assert len(channel) == time_steps, \
                    f"Invalid number of time_steps for tx={tx}, rx={rx}: " \
                    f"declared {time_steps}, found {len(channel)}"

    def get_num_nodes(self) -> int:
        return self._cfg['numberOfNodes']

    def get_scenario_duration(self) -> float:
        return self._cfg['totalTimeDuration']

    def get_frequency(self) -> float:
        return self._cfg['carrierFrequency']

    def get_time_steps(self) -> int:
        return self._cfg['numberOfTimeDivisions']

    def get_time_step_duration(self) -> float:
        return self.get_scenario_duration() / self.get_time_steps()

    def get_channel(self, tx: Optional[int] = None, rx: Optional[int] = None, t: Optional[int] = None) -> Union[
        List[List[Optional[List[List[Ray]]]]], List[List[Ray]], List[Ray]]:

        if (tx is None) and (rx is None) and (t is None):
            # Return all channels
            return self._channel

        if tx is not None:
            assert 0 <= tx < self.get_num_nodes(), \
                f"Invalid tx index ({tx}) for scenario with {self.get_num_nodes()} nodes"
        if rx is not None:
            assert 0 <= rx < self.get_num_nodes(), \
                f"Invalid rx index ({rx}) for scenario with {self.get_num_nodes()} nodes"

        assert tx != rx, f"Invalid query for self-channel"

        if (tx is not None) and (rx is not None) and (t is None):
            # Return all time steps for given tx/rx pair
            return self._channel[tx][rx]

        if t is not None:
            assert 0 <= t < self.get_time_steps(), \
                f"Requesting invalid time step ({t}) for scenario with {self.get_time_steps()} time steps"

        if (tx is not None) and (rx is not None) and (t is not None):
            # Return list of Ray for given tx/rx pair at given time step
            return self._channel[tx][rx][t]

        raise ValueError(f"Invalid arguments: tx={tx}, rx={rx}, t={t}")
