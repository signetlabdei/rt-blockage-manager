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

from typing import TextIO, List, Dict, Optional, Union, Any
from src.geometry import Point
from src.ray import Ray
from src import utils
import os
import shutil
import re
import math

# Constants
_INPUT_PATH = "Input"

_OUTPUT_PATH = "Output"
# Output subdirs
_NS3_PATH = os.path.join(_OUTPUT_PATH, "Ns3")
_VISUALIZER_PATH = os.path.join(_OUTPUT_PATH, "Visualizer")
# Output/Ns3 subdirs
_QD_FILES_PATH = os.path.join(_NS3_PATH, "QdFiles")
# Output/Visualizer subdirs
_MPC_COORDS_PATH = os.path.join(_VISUALIZER_PATH, "MpcCoordinates")

_PARA_CFG_HEADER = "ParameterName\tParameterValue"
_QD_FILENAME_REGEX = r"Tx(\d+)Rx(\d+).txt"
_MPC_COORDS_FILENAME_REGEX = r"MpcTx(\d+)Rx(\d+)Refl(\d+)Trc(\d+).csv"

# Functions


def get_para_cfg_name() -> str:
    return "paraCfgCurrent.txt"


def get_qd_file_name(tx: int, rx: int) -> str:
    return f"Tx{tx}Rx{rx}.txt"


def get_mpc_coords_name(tx: int, rx: int, refl: int, t: int) -> str:
    assert tx < rx, f"It is assumed that tx<rx, instead, {tx=}, {rx=}"
    return f"MpcTx{tx}Rx{rx}Refl{refl}Trc{t}.csv"


def _get_next_row_floats(file: TextIO, n_rays: int) -> List[float]:
    line = file.readline()
    assert line != '', f"Unexpected EOF for file {file.name}"

    floats = [float(x) for x in line.split(',')]
    assert len(
        floats) == n_rays, f"Expected {n_rays} entries, found {len(floats)}"

    return floats


def _write_row_floats(file: TextIO, floats: List[float], precision: int = 6) -> None:
    float_format = f".{precision}g"
    ss = [f"{x:{float_format}}" for x in floats]
    file.write(",".join(ss))
    file.write("\n")


def import_qd_file(path: str) -> List[Dict[str, Any]]:
    with open(path, 'rt') as f:
        line = f.readline()

        qd_file = []
        while line != '':  # EOF='' in python
            n_rays = int(line)
            rays: Dict[str, Union[int, List[float]]] = {'n_rays': n_rays}

            if n_rays > 0:
                rays = {**rays,  # add to dict
                        'delay': _get_next_row_floats(f, n_rays),  # [s]
                        'path_gain': _get_next_row_floats(f, n_rays),  # [dB]
                        'phase_offset': _get_next_row_floats(f, n_rays),  # [rad]
                        'aod_el': _get_next_row_floats(f, n_rays),  # [deg]
                        'aod_az': _get_next_row_floats(f, n_rays),  # [deg]
                        'aoa_el': _get_next_row_floats(f, n_rays),  # [deg]
                        'aoa_az': _get_next_row_floats(f, n_rays)}  # [deg]

            qd_file.append(rays)

            # read next line
            line = f.readline()

    return qd_file


def export_qd_file(out_folder: str, tx: int, rx: int,
                   ch_matrix: List[List[Optional[List[List[Ray]]]]],
                   precision: int = 6) -> None:
    ch = ch_matrix[tx][rx]
    assert ch is not None, f"Invalid choice for {tx=}, {rx=}: channel is None"

    qd_folder = os.path.join(out_folder, _QD_FILES_PATH)
    filename = get_qd_file_name(tx, rx)
    os.makedirs(qd_folder, exist_ok=True)

    with open(os.path.join(qd_folder, filename), 'wt') as f:
        for rays in ch:
            # for each timestep
            n_rays = len(rays)
            f.write(f"{n_rays}\n")

            if n_rays > 0:
                _write_row_floats(f, [ray.delay  # [s]
                                      for ray in rays], precision)
                _write_row_floats(f, [ray.path_gain  # [dB]
                                      for ray in rays], precision)
                _write_row_floats(f, [ray.phase  # [rad]
                                      for ray in rays], precision)
                _write_row_floats(f, [math.degrees(ray.aod_inclination())  # [deg]
                                      for ray in rays], precision)
                _write_row_floats(f, [math.degrees(ray.aod_azimuth()) % 360  # [deg] wrap to [0, 360)
                                      for ray in rays], precision)
                _write_row_floats(f, [math.degrees(ray.aoa_inclination())  # [deg]
                                      for ray in rays], precision)
                _write_row_floats(f, [math.degrees(ray.aoa_azimuth()) % 360  # [deg] wrap to [0, 360)
                                      for ray in rays], precision)


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


def export_mpc_coordinates(out_folder: str, tx: int, rx: int,
                           ch_matrix: List[List[Optional[List[List[Ray]]]]]) -> None:
    ch = ch_matrix[tx][rx]
    assert ch is not None, f"Invalid choice for {tx=}, {rx=}: channel is None"

    mpc_coords_folder = os.path.join(out_folder, _MPC_COORDS_PATH)
    os.makedirs(mpc_coords_folder, exist_ok=True)

    for t, rays in enumerate(ch):
        for ray in rays:
            refl = ray.refl_order()
            # Only MPC Coordinates with (tx < rx) are surely printed
            mpc_coords_file_name = get_mpc_coords_name(tx, rx, refl, t)

            # Appen one ray at a time
            with open(os.path.join(mpc_coords_folder, mpc_coords_file_name), 'at') as f:
                coords = [f"{p.x},{p.y},{p.z}" for p in ray.vertices]
                f.write(','.join(coords))
                f.write('\n')


def import_rays(scenario_path: str, max_refl: int, tx: int, rx: int) -> List[List[Ray]]:
    qd_file_folder = os.path.join(scenario_path, _QD_FILES_PATH)
    mpc_coords_folder = os.path.join(
        scenario_path, _MPC_COORDS_PATH)

    if tx > rx:
        reverse_points = True
    else:
        reverse_points = False

    # Import QdFile
    qd_file_name = get_qd_file_name(tx, rx)
    qd_file = import_qd_file(os.path.join(qd_file_folder, qd_file_name))

    # Import MpcCoordinates
    time_steps = []
    for t, qd_step in enumerate(qd_file):
        rays = []

        for refl in range(max_refl + 1):  # making range() inclusive
            # Only MPC Coordinates with (tx < rx) are surely printed
            if not reverse_points:
                mpc_coords_file_name = get_mpc_coords_name(tx, rx, refl, t)
            else:
                mpc_coords_file_name = get_mpc_coords_name(rx, tx, refl, t)

            try:
                mpc_coords = import_mpc_coordinates(
                    os.path.join(mpc_coords_folder, mpc_coords_file_name))

                for vertices in mpc_coords:
                    if reverse_points:
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


def import_parameter_configuration(scenario_path: str) -> Dict[str, Any]:
    filename = get_para_cfg_name()

    with open(os.path.join(scenario_path, _INPUT_PATH, filename), 'rt') as f:
        line = f.readline().strip()
        assert line == _PARA_CFG_HEADER, f"First line should be a standard header. Found instead: {line}"

        # discard header
        line = f.readline().strip()
        cfg: Dict[str, Any] = {}
        while line != '':  # EOF='' in python
            param_name, param_value = line.split('\t')

            # value conversion
            converted_value: Any
            if utils.isint(param_value):
                converted_value = int(param_value)
            elif utils.isfloat(param_value):
                converted_value = float(param_value)
            else:
                converted_value = param_value

            # add parameter to dictionary
            cfg[param_name] = converted_value

            # read next line
            line = f.readline().strip()

    return cfg


def export_parameter_configuration(out_folder: str, cfg: Dict[str, Any]) -> None:
    input_folder = os.path.join(out_folder, _INPUT_PATH)
    os.makedirs(input_folder, exist_ok=True)
    filename = get_para_cfg_name()

    with open(os.path.join(input_folder, filename), 'wt') as f:
        f.write(f"{_PARA_CFG_HEADER}\n")
        for k, v in cfg.items():
            f.write(f"{k}\t{v}\n")


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


def export_scenario(out_folder: str,
                    ch_matrix: List[List[Optional[List[List[Ray]]]]],
                    precision: int = 6,
                    do_export_mpc_coords: bool = True) -> None:

    os.makedirs(out_folder, exist_ok=True)

    # If the MPC coords folder already exists: remove it
    # MPC coords are appended to files, the folder must be empty
    mpc_coords_folder = os.path.join(
        out_folder, _MPC_COORDS_PATH)
    if os.path.exists(mpc_coords_folder):
        shutil.rmtree(mpc_coords_folder)
    os.makedirs(mpc_coords_folder, exist_ok=False)

    n_nodes = len(ch_matrix)
    for tx in range(n_nodes):
        for rx in range(n_nodes):
            if rx == tx:
                # Avoid exporting self-channel
                continue

            export_qd_file(out_folder, tx, rx, ch_matrix, precision)
            if do_export_mpc_coords and (tx < rx):
                export_mpc_coordinates(out_folder, tx, rx, ch_matrix)


def get_other_files(scenario_path: str) -> List[str]:
    all_files = _list_files(scenario_path, recursive=True)

    other_files = []
    for file in all_files:
        if is_qd_file(file):
            continue

        if is_mpc_coords_file(file):
            continue

        if is_para_cfg_file(file):
            continue

        other_files.append(file)

    return other_files


def _list_files(path: str, recursive: bool = True) -> List[str]:
    listdir = os.listdir(path)

    # separate files and folders
    files: List[str] = []
    folders: List[str] = []
    for f in listdir:
        full_path = os.path.join(path, f)
        if os.path.isfile(full_path):
            files.append(f)
        elif os.path.isdir(full_path):
            folders.append(f)
        else:
            raise TypeError(
                f"{full_path=} is neither a file nor a folder")  # pragma: no cover

    if not recursive:
        return files

    if len(folders) == 0:
        # no subfolders: return
        return files

    for folder in folders:
        subfiles = _list_files(os.path.join(path, folder), recursive=recursive)
        # append sub-files
        files += [os.path.join(folder, file) for file in subfiles]

    return files


def is_qd_file(filepath: str) -> bool:
    dir, filename = os.path.split(filepath)
    if not re.match(_QD_FILENAME_REGEX, filename):
        return False

    pardir, qd_dir = os.path.split(dir)
    pardir, ns3_dir = os.path.split(pardir)
    _, out_dir = os.path.split(pardir)
    if os.path.join(out_dir, ns3_dir, qd_dir) != _QD_FILES_PATH:
        return False

    return True


def is_mpc_coords_file(filepath: str) -> bool:
    dir, filename = os.path.split(filepath)
    if not re.match(_MPC_COORDS_FILENAME_REGEX, filename):
        return False

    pardir, mpc_coords_dir = os.path.split(dir)
    pardir, vis_dir = os.path.split(pardir)
    _, out_dir = os.path.split(pardir)
    if os.path.join(out_dir, vis_dir, mpc_coords_dir) != _MPC_COORDS_PATH:
        return False

    return True


def is_para_cfg_file(filepath: str) -> bool:
    dir, filename = os.path.split(filepath)
    if filename != get_para_cfg_name():
        return False

    _, input_dir = os.path.split(dir)
    if input_dir != _INPUT_PATH:
        return False

    return True
