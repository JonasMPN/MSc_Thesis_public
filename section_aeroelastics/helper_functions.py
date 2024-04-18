import pathlib
import os
import shutil
import matplotlib.pyplot
import numpy as np
from typing import Tuple, List
import json


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
                   logger=None) -> Tuple[str, bool]:
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
                    print_message: bool) -> Tuple[str, bool]:
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
