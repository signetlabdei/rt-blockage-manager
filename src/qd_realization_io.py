# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from typing import TextIO, List, Dict, Optional, Union
from src.geometry import Point
from src.ray import Ray
from src import utils
import os


def _get_next_row_floats(file: TextIO, n_rays: int) -> List[float]:
    line = file.readline()
    assert line != '', f"Unexpected EOF for file {file.name}"

    floats = [float(x) for x in line.split(',')]
    assert len(
        floats) == n_rays, f"Expected {n_rays} entries, found {len(floats)}"

    return floats


def import_qd_file(path: str) -> List[Dict]:
    with open(path, 'rt') as f:
        line = f.readline()

        qd_file = []
        while line != '':  # EOF='' in python
            n_rays = int(line)
            rays: Dict[str, Union[int, List[float]]] = {'n_rays': n_rays}

            if n_rays > 0:
                rays = {**rays,  # add to dict
                        'delay': _get_next_row_floats(f, n_rays),
                        'path_gain': _get_next_row_floats(f, n_rays),
                        'phase_offset': _get_next_row_floats(f, n_rays),
                        'aod_el': _get_next_row_floats(f, n_rays),
                        'aod_az': _get_next_row_floats(f, n_rays),
                        'aoa_el': _get_next_row_floats(f, n_rays),
                        'aoa_az': _get_next_row_floats(f, n_rays)}

            qd_file.append(rays)

            # read next line
            line = f.readline()

    return qd_file


def _get_vertices_from_csv_string(line: str) -> List[Point]:
    coords = [float(x) for x in line.split(',')]
    assert len(
        coords) % 3 == 0, f"Line must have a multiple of 3 coordinates, found {len(coords)}, instead"

    # The string is expected to be formatted as "x1,y1,z1,x2,y2,z2,...,xn,yn,zn"
    vertices = [Point(x, y, z)
                for x, y, z in zip(coords[0::3], coords[1::3], coords[2::3])]
    return vertices


def import_mpc_coordinates(path: str) -> List[List[Point]]:
    with open(path, 'rt') as f:
        line = f.readline()

        rays_vxs = []
        while line != '':  # EOF='' in python
            vxs = _get_vertices_from_csv_string(line)
            rays_vxs.append(vxs)

            # read next line
            line = f.readline()

    return rays_vxs


def import_rays(scenario_path: str, max_refl: int, tx: int, rx: int) -> List[List[Ray]]:
    qd_file_folder = os.path.join(scenario_path, "Output/Ns3/QdFiles")
    mpc_coords_folder = os.path.join(
        scenario_path, "Output/Visualizer/MpcCoordinates")

    # Import QdFile
    qd_file_name = f"Tx{tx}Rx{rx}.txt"
    qd_file = import_qd_file(os.path.join(qd_file_folder, qd_file_name))

    # Import MpcCoordinates
    time_steps = []
    for t, qd_step in enumerate(qd_file):
        rays = []

        for refl in range(max_refl + 1):  # making range() inclusive
            # Only MPC Coordinates with (tx < rx) are surely printed
            mpc_coords_file_name = f"MpcTx{min(tx, rx)}Rx{max(tx, rx)}Refl{refl}Trc{t}.csv"

            try:
                mpc_coords = import_mpc_coordinates(
                    os.path.join(mpc_coords_folder, mpc_coords_file_name))

                for vertices in mpc_coords:
                    if tx > rx:
                        # The file with reversed tx/rx has been read: vertices need to be reversed
                        vertices = vertices[::-1]

                    rays.append(Ray(delay=qd_step['delay'].pop(0),
                                    path_gain=qd_step['path_gain'].pop(0),
                                    phase=qd_step['phase_offset'].pop(0),
                                    vertices=vertices))
            except FileNotFoundError:
                # if the mpc coordinate is not found, just ignore it
                pass

        assert len(
            rays) == qd_step['n_rays'], f"Expected {qd_step['n_rays']} rays, found {len(rays)} MPC coordinates"
        time_steps.append(rays)

    return time_steps


def import_parameter_configuration(scenario_path: str) -> Dict:
    filename = "Input/paraCfgCurrent.txt"

    with open(os.path.join(scenario_path, filename), 'rt') as f:
        line = f.readline().strip()
        assert line == "ParameterName\tParameterValue", f"First line should be a standard header. Found instead: {line}"

        # discard header
        line = f.readline().strip()
        cfg = {}
        while line != '':  # EOF='' in python
            param_name, param_value = line.split('\t')

            # value conversion
            if utils.isint(param_value):
                param_value = int(param_value)
            elif utils.isfloat(param_value):
                param_value = float(param_value)

            # add parameter to dictionary
            cfg[param_name] = param_value

            # read next line
            line = f.readline().strip()

    return cfg


def import_scenario(scenario_path: str) -> List[List[Optional[List[List[Ray]]]]]:
    assert os.path.isdir(
        scenario_path), f"scenario_path={scenario_path} not found"
    cfg = import_parameter_configuration(scenario_path)

    # prepare list-matrix (n_nodes x n_nodes) with pointers to channel lists
    n_nodes = cfg['numberOfNodes']
    channels: List[List[Optional[List[List[Ray]]]]] = [[[] for _ in range(n_nodes)]
                                                       for _ in range(n_nodes)]

    for tx in range(n_nodes):
        for rx in range(n_nodes):
            if tx == rx:
                # no channel to itself
                rays = None
            else:
                rays = import_rays(scenario_path=scenario_path,
                                   max_refl=cfg['totalNumberOfReflections'],
                                   tx=tx,
                                   rx=rx)
                assert len(rays) == cfg['numberOfTimeDivisions'], \
                    f"Expected {cfg['numberOfTimeDivisions']}, found {len(rays)}, instead"

            channels[tx][rx] = rays

    return channels
