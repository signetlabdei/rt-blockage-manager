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

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List, overload, Any
import src.qd_realization_io as qdio
from src.ray import Ray
import shutil
from os import path
import os
import math


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

    @abstractmethod
    def get_relative_power_threshold(self) -> float:
        """
        Return the relative power threshold of the scenario
        """

    @abstractmethod
    def get_absolute_power_threshold(self) -> float:
        """
        Return the absolute power threshold of the scenario
        """

    @abstractmethod
    def export(self, out_folder: str) -> None:
        """
        Export scenario in the appropriate folder structure
        """

    @abstractmethod
    @overload
    def get_channel(self) -> List[List[Optional[List[List[Ray]]]]]:
        ...

    @abstractmethod
    @overload
    def get_channel(self, tx: int, rx: int) -> List[List[Ray]]:
        ...

    @abstractmethod
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

    @abstractmethod
    def set_channel(self, tx: int, rx: int, t: List[List[Ray]]) -> None:
        """
        Set or update the existing channel
        """


class QdRealizationScenario(Scenario):
    def __init__(self, scenario_path: str):
        self._scenario_path = scenario_path

        # import qd-realization scenario
        self._channel = qdio.import_scenario(self._scenario_path)
        self._cfg: Dict[str, Any] = qdio.import_parameter_configuration(
            self._scenario_path)
        self._other_files = qdio.get_other_files(self._scenario_path)

        self._assess_scenario_validity()

    def _assess_scenario_validity(self):
        assert self._channel is not None

        n_nodes = self._cfg['numberOfNodes']
        # Check number of nodes (num rows)
        assert len(self._channel) == n_nodes, \
            f"Invalid number of nodes: declared {n_nodes}, found {len(self._channel)} tx (dim 0)"
        # Check number of nodes (num cols for each row)
        for rxs in self._channel:
            assert len(rxs) == n_nodes, \
                f"Invalid number of nodes: declared {n_nodes}, found {len(self._channel)} rx (dim 1)"

        # Check number of time steps for each node pair
        time_steps = self._cfg['numberOfTimeDivisions']
        for tx, row in enumerate(self._channel):
            for rx, channel in enumerate(row):
                if channel is None:
                    # Ignore None channels (e.g., elements on main diagonal)
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

    def get_relative_power_threshold(self) -> float:
        return self._cfg.get('minRelativePathGainThreshold', -math.inf)

    def get_absolute_power_threshold(self) -> float:
        return self._cfg.get('minAbsolutePathGainThreshold', -math.inf)

    def get_time_step_duration(self) -> float:
        return self.get_scenario_duration() / self.get_time_steps()

    def export(self, out_folder: str,
               precision: int = 6,
               do_export_mpc_coords: bool = True,
               do_copy_unnecessary_files: bool = True) -> None:
        qdio.export_parameter_configuration(out_folder, self._cfg)
        qdio.export_scenario(out_folder, self._channel,
                             precision=precision,
                             do_export_mpc_coords=do_export_mpc_coords)

        if do_copy_unnecessary_files and (not path.samefile(self._scenario_path, out_folder)):
            for file in self._other_files:
                src_file = path.join(self._scenario_path, file)
                dst_file = path.join(out_folder, file)

                if os.path.exists(dst_file):
                    os.remove(dst_file)

                dst_dir, _ = path.split(dst_file)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src_file, dst_file)

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
            ch = self._channel[tx][rx]
            assert ch is not None, f"channel({tx=}),({rx=}) should not be None"
            return ch

        if t is not None:
            assert 0 <= t < self.get_time_steps(), \
                f"Requesting invalid time step ({t}) for scenario with {self.get_time_steps()} time steps"

        if (tx is not None) and (rx is not None) and (t is not None):
            # Return list of Ray for given tx/rx pair at given time step
            ch = self._channel[tx][rx]
            assert ch is not None, f"channel({tx=}),({rx=}) should not be None"
            return ch[t]

        raise ValueError(
            f"Invalid arguments: tx={tx}, rx={rx}, t={t}")  # pragma: no cover

    def set_channel(self, tx: int, rx: int, t: List[List[Ray]]) -> None:
        assert 0 <= tx < self.get_num_nodes(
        ), f"Invalid tx index ({tx}) for scenario with {self.get_num_nodes()} nodes"
        assert 0 <= rx < self.get_num_nodes(
        ), f"Invalid rx index ({rx}) for scenario with {self.get_num_nodes()} nodes"
        ch = self._channel[tx][rx]
        assert ch is not None, f"None channel for {tx=}, {rx=}"
        assert len(t) == len(ch), f"Invalid {len(t)=}: should be {len(ch)}"

        self._channel[tx][rx] = t
        self._assess_scenario_validity()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, QdRealizationScenario):
            raise TypeError(f"{type(o)=}")

        if self._cfg != o._cfg:
            return False

        if sorted(self._other_files) != sorted(o._other_files):
            return False
        
        # compare channels
        for tx in range(self.get_num_nodes()):
            for rx in range(self.get_num_nodes()):
                if tx == rx:
                    continue

                for t in range(self.get_time_steps()):
                    self_ch = self.get_channel(tx, rx, t)
                    o_ch = o.get_channel(tx, rx, t)

                    if self_ch != o_ch:
                        return False

        return True
