import pathlib
import os
import shutil
from os.path import join, isfile
import pandas as pd
import numpy as np


class Helper():
    """Utility class for various functionalities.
    """
    def __init__(self):
        pass

    def create_dir(self,
                   path_dir: str,
                   overwrite: bool = False,
                   add_missing_parent_dirs: bool = True,
                   raise_exception: bool = False,
                   verbose: bool = False,
                   logger=None) -> tuple[str, bool]:
        """Creates directory.

        :param path_dir: Directory to create
        :type path_dir: str
        :param overwrite: Whether or not to overwrite if "path_dir" directory already exists, defaults to False
        :type overwrite: bool, optional
        :param add_missing_parent_dirs: If "path_dir" is multiple levels deep, whether to create parten directories.
          defaults to True
        :type add_missing_parent_dirs: bool, optional
        :param raise_exception: Whether to raise an exception if the directory already exists (if overwrite=False)
         or there are missing parent directories (if add_missing_parent_dirs=False), defaults to False
        :type raise_exception: bool, optional
        :param verbose: Whether to print the action that was performed to the console, defaults to False
        :type verbose: bool, optional
        :param logger: Instance of a logger, defaults to None
        :type logger: Some logger instance, optional
        :return: _description_
        :rtype: Tuple[str, bool]
        """
        if not isinstance(overwrite, bool):
            raise ValueError(f"'overwrite' must be of type bool, but its value is '{overwrite}' of type "
                             f"'{type(overwrite)}'.")
        msg, keep_going = self._create_dir(path_dir, overwrite, add_missing_parent_dirs, raise_exception, verbose)
        if logger is not None:
            logger.info(msg)
        return path_dir, msg, keep_going

    @staticmethod
    def open_dirs(directories: str|list[str]) -> None:
        """Open directories in the file explorer.

        :param directories: Directory or directories to open.
        :type directories: str | list[str]
        :return: None
        :rtype: None
        """
        directories = directories if not isinstance(directories, str) else [directories]
        for directory in directories:
            try:
                os.startfile(directory)
            except FileNotFoundError:
                print(f"Directory {directory} could not be opened because it doesn't exist")
        return None
    
    @staticmethod
    def _create_dir(target: str,
                    overwrite: bool,
                    add_missing_parent_dirs: bool,
                    raise_exception: bool,
                    print_message: bool) -> tuple[str, bool]:
        msg, keep_going = str(), bool()
        try:
            if overwrite:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    msg = f"Existing directory {target} was overwritten."
                else:
                    msg = f"Could not overwrite {target} as it did not exist. Created it instead."
                keep_going = True
            else:
                msg, keep_going = f"Directory {target} created successfully.", True
            pathlib.Path(target).mkdir(parents=add_missing_parent_dirs, exist_ok=False)
        except Exception as exc:
            if exc.args[0] == 2:  # FileNotFoundError
                if raise_exception:
                    raise FileNotFoundError(f"Not all parent directories exist for directory {target}.")
                else:
                    msg, keep_going = f"Not all parent directories exist for directory {target}.", False
            elif exc.args[0] == 17:  # FileExistsError
                if raise_exception:
                    raise FileExistsError(f"Directory {target} already exists and was not changed.")
                else:
                    msg, keep_going = f"Directory {target} already exists and was not changed.", False
        if print_message:
            print(msg)
        return msg, keep_going


def convert_plot_digitizer_dfs(root: str, exclude: str):
    def rep(value):
        return float(value.replace(",", "."))
    map_coeff = {"Cl": "CL", "Cd": "CD", "Cm": "CM"}
    dir_save = Helper().create_dir(join(root, "plot_digitizer_converted"))[0]
    for file_name in os.listdir(root):
        file = join(root, file_name)
        if exclude in file_name or not isfile(file):
            continue
        df = pd.read_csv(file, delimiter=";")
        df_converted = pd.DataFrame()
        coeff = map_coeff[file_name[:2]]
        for col, new_col in zip(df.columns, ["AOA", coeff]):
            data = df[col].apply(rep)
            df_converted[new_col] = data
        df_converted.to_csv(join(dir_save, file_name), index=False)


def save_OSU_static_polar(ff_measurement: str, ff_save: str):
    alpha = []
    C_l = []
    C_d = []
    C_m = []
    with open(ff_measurement) as f:
        lines = f.readlines()
        for line in lines:
            split = line.split()
            if len(split) > 0:
                if split[0] == "Corrected":
                    alpha.append(float(split[6].split("=")[1]))
                    C_l.append(float(split[7].split("=")[1]))
                    C_d.append(float(split[8].split("=")[1]))
                    C_m.append(float(split[9].split("=")[1]))
                elif split[0] ==  "Drag":
                    C_d[-1] = float(split[-1])
    alpha = np.asarray(alpha)
    C_l = np.asarray(C_l)
    C_d = np.asarray(C_d)
    C_m = np.asarray(C_m)
            
    sort_ids = alpha.argsort()
    alpha = alpha[sort_ids]
    C_l = C_l[sort_ids]
    C_d = C_d[sort_ids]
    C_m = C_m[sort_ids]
    pd.DataFrame({"alpha": alpha, "C_l": C_l, "C_d": C_d, "C_m": C_m}).to_csv(ff_save, index=None)


def save_OSU_unsteady_polar(ff_measurement: str, ff_save: str):
    data = []
    with open(ff_measurement) as f:
        lines = f.readlines()
        found_beginning = False
        for line in lines:
            split = line.split()
            if len(split) != 0:
                if split[0] == "Sample":
                    found_beginning = True
                    continue
                if not found_beginning:
                    continue
            elif found_beginning:
                break
            else:
                continue
            current_data = [float(val.split(",")[0]) for val in split[2:]]
            data.append(current_data)
    pd.DataFrame(data, columns=["alpha", "C_l", "C_d", "C_m"]).to_csv(ff_save, index=None)
    
    


