from abc import ABC, abstractmethod
from defaults import DefaultsSimulation
import numpy as np
import pandas as pd
from os.path import join
from copy import copy
from typing import Callable


class SimulationResults(DefaultsSimulation):
    """Instances of this class are used as data storage variables during the airfoil simulation. Parameters are stored
    as attributes of the instance. Attributes can be dynamically added using "add_param()". Each instance comes with a
     set of default parameters; these are defined in the parent class DefaultsSimulation. All parameter attributes are initialised as numpy arrays with as many values as there are timesteps in the simulation. 

    :param DefaultsSimulation: Class defining default settings for the simulation parameters.
    """
    def __init__(self, time: np.ndarray) -> None:
        """Initialise an instance. The instance has a set of default parameters as defined in the parent class 
        DefaultsSimulation.

        :param time: Time array that is used for the simulation.
        :type time: np.ndarray
        """
        DefaultsSimulation.__init__(self)
        self._n_time_steps = time.size
        self.time = time
        self.add_param(*self._dfl_params)  # _dfl_params comes from DefaultsSimulation
    
    def set_param(self, **kwargs):
        """Adds an instance attribute with the name of the kwarg and the value of the kwarg.
        """
        for name, value in kwargs.items():
            vars(self)[name] = value

    def add_param(self, *args: str | tuple[str, int], init_as=np.zeros):
        """Add instance attributes. Each attribute is initialised to hold a value for each time step. 

        :param init_as: _description_, defaults to np.zeros
        :type init_as: _type_, optional
        """
        args = [(arg, 1) if isinstance(arg, str) else arg for arg in args]
        for name, n_params in args:
            vars(self)[name] = init_as((self._n_time_steps, n_params)).squeeze()
    
    def _save(
        self, 
        root: str, 
        files: dict[str, list[str]]=None, 
        split: dict[str, list[str]]=None, 
        use_default: bool=True):
        """Saves instance attributes to CSV .dat files.

        :param root: Root directory in relation to the current working directory into which the files are saved.
        :type root: str
        :param files: Sets the name of the file (key) and the instance attributes saved into it (values), defaults to 
        None. If "use_default=True", the default parameters from DefaultsSimulation._dfl_files are always saved.
        :type files: dict[str: list[str]], optional
        :param split: Instance attributes that are multidimensional arrays should be split into each component. 
        "split" defines the appendices the names of the components get. Example: split={"vel": ["x", "y", "rot"]}
        will cause the attribute "vel" to be saved as three columns with names "vel_x", "vel_y", and "vel_z". If
        "use_default=True", the default splot from DefaultsSimulation._dfl_splot is always used. defaults to None
        :type split: dict[str: str], optional
        :param use_default: Whether or not to use the defaults for "files" and "split". If "True", the default 
        settings are added to the user-specified "files" and "split". To only use the user-specified "files" and 
        "split", "use_default=False" has to be set. defaults to True
        :type use_default: bool, optional
        """
        files = files if files is not None else {}
        split = split if split is not None else {}
        user_defined_split = split.keys()
        user_defined_files = files.keys()

        if use_default:
            for param, split_into in self._dfl_split.items():
                if param not in user_defined_split:
                    split[param] = split_into
                    
            for filename, params in self._dfl_files.items():
                if filename in user_defined_files:
                    if files[filename] is not None:
                        params = list(set(params+files[filename]))
                    files[filename] = params
                else:
                    files[filename] = self._dfl_files[filename]
                    
        for file, params in files.items():
            df = pd.DataFrame()
            for param in params:
                if param in split.keys():
                    for i, split_name in enumerate(split[param]):
                        try:
                            df[param+"_"+split_name] = vars(self)[param][:, i]
                        except ValueError:
                            raise ValueError(f"The above ValueError was caused for parameter {param}.")
                else:
                    df[param] = vars(self)[param]
            df.to_csv(join(root, file), index=False)


class SimulationSubRoutine(ABC):
    def __init__(self, verbose: bool=True) -> None:
        self.verbose = verbose
        defined = {
            "_scheme_settings_must": self._scheme_settings_must.keys(),
            "_scheme_settings_can": self._scheme_settings_can.keys(),
            "_sim_params_required": self._sim_params_required.keys()
        }
        if "_copy_scheme" in vars(self).keys():
            copy_params = {new_scheme: from_scheme for from_scheme, copy_to in self._copy_schemes.items() 
                           for new_scheme in copy_to }
        else:
            copy_params = {}
        copied_params = copy_params.keys()
        for scheme in self._implemented_schemes:
            for attribute, has_keys in copy(defined).items():
                if scheme in copied_params:
                    assert scheme not in has_keys, (f"Settings for scheme '{scheme}' is tried to be set directly and "
                                                    f"by copying from scheme '{copy_params[scheme]}' simultaneously. "
                                                    "Only one at a time is allowed.")
                    defined[scheme] = vars(self, attribute)[copy_params[scheme]]
                    continue
                assert scheme in has_keys, (f"Attribute '{attribute}' of class '{type(self).__name__}' is/are missing "
                                            f"a key for scheme '{scheme}'. Implemented keys are {has_keys}.")
            for fnc in [self.get_scheme_method, self.get_scheme_init_method]:
                try:
                    fnc(scheme)
                except KeyError:
                    raise AssertionError(f"Method '{fnc.__name__}()' of class '{type(self).__name__}' is/are missing a "
                                         f"return for scheme '{scheme}'.")        

    def prepare_and_get_scheme(self, scheme: str, simulation_results: SimulationResults, call_from: str, **kwargs):
        """Returns a function object that receives a ThreeDOFsAirfoil instance and returns a numpy array with
        [aero_force_x, aero_force_y, aero_force_moment]. "scheme" sets which kind of calculation for the aerodynamic
        forces is used. Implemented schemes are defined in _implemented_schemes.

        :param scheme: Scheme for calculating the aerodynamic forces. Can be "steady", "quasi_steady", and "BL". "BL"
         uses a Beddoes-Leishman model as given in #todo add paper.
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the aerodynamic forces.
        :rtype: Callable
        """
        self._check_scheme(scheme)
        self._pre_calculations(scheme)

        must_have = self._scheme_settings_must[scheme]
        can_have = self._scheme_settings_can[scheme]
        self._check_scheme_settings(scheme, call_from, must_have, can_have, kwargs.keys())
        for param in self._sim_params_required[scheme]:
            simulation_results.add_param(param)
        return self.get_scheme_method(scheme)
    
    def get_additional_params(self, scheme: str):
        return self._sim_params_required[scheme]

    def _check_scheme(self, scheme: str):
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
    
    def _check_scheme_settings(self, scheme: str, fnc_call: str, must_have: list, can_have: list, was_given: list):
        """Utility function checking whether all necessary scheme settings have been set.

        :param scheme: _description_
        :type scheme: str
        :param fnc_call: _description_
        :type fnc_call: str
        :param must_have: _description_
        :type must_have: list
        :param was_given: _description_
        :type was_given: list
        :raises ValueError: _description_
        """
        is_missing = []
        for must in must_have:
            if must not in was_given:
                is_missing.append(must)
                
        not_supported = []
        for given in was_given:
            if given not in can_have+must_have:
                not_supported.append(given)
                
        if len(is_missing) != 0:
            raise ValueError(f"Using scheme '{scheme}' requires that {must_have} are given in function call "
                             f"'{fnc_call}'()', but {is_missing} is/are missing.")
        
        if len(not_supported) != 0 and self.verbose:
            print(f"Using scheme '{scheme}' in '{fnc_call}()' accepts kwrags '{can_have}', but {not_supported} "
                  f"was/were additionally given. They have no influence on the execution.")
  
    def _skip(*args, **kwargs):
        pass

    def _pre_calculations(self, scheme: str):
        pass

    @abstractmethod
    def get_scheme_method(self, scheme: str) -> Callable:
        pass

    @abstractmethod
    def get_scheme_init_method(self, scheme: str) -> Callable:
        pass

    @property
    @abstractmethod
    def _implemented_schemes() -> list[str]:
        pass

    @property
    @abstractmethod
    def _scheme_settings_must() -> dict[str, list[str]]:
        pass

    @property
    @abstractmethod
    def _scheme_settings_can() -> dict[str, list[str]]:
        pass

    @property
    @abstractmethod
    def _sim_params_required() -> dict[str, list[str]]:
        pass


class Rotations:
    """Class providing methods for anything related to performing rotations.
    """
    def project(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Projecst the points in "array" into a coordinate system rotated by the angles given in "angles". The 
        rotation is around the third component of "array", meaning it doesn't change that value. If "angles" is a 
        single value, all points are rotated by that angle. If "angles" has as many values as "array" has points, each 
        point is rotated by the angle in "angles" of the same index.

        :param array: (n, 3) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 3, 1))
        rot_matrices = self.passive_3D_planar(angles)
        return (rot_matrices@array).reshape(n_rot, 3)
    
    def project_separate(self, array: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Same behaviour as "project()", except that the first two components of "array" are separately
        projected into the rotated coordinate system (that is given by "angles"). Example:
        array[i, :] = [drag, lift, moment] in an aerodynamic coordinate system that is rotated by "alpha"
        from the x-y-z coordinate system. then the output of "project_separate(array, alpha) is 
        [drag_x, drag_y, lift_x, lift_y, moment], i.e., the drag and lift projected onto the x and y axes.

        :param array: (n, 3) array
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: Projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 3, 1))
        rot_matrices = self.passive_3D_planar_separate(angles)
        combined = (rot_matrices@array).reshape(n_rot, 5)
        return combined
    
    def project_2D(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Projecst the points in "array" into a coordinate system rotated by the angles given in "angles". The 
        rotation is around the third component of "array", meaning it doesn't change that value. If "angles" is a 
        single value, all points are rotated by that angle. If "angles" has as many values as "array" has points, each 
        point is rotated by the angle in "angles" of the same index.

        :param array: (n, 3) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 2, 1))
        rot_matrices = self.passive_2D(angles)
        return (rot_matrices@array).reshape(n_rot, 2)
    
    def rotate_2D(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Rotates the points in "array" in the current coordinate system rotated by the angles given in "angles". 
        Assuming the coordinates are [x, y], the rotation is around a hypothetical z-axis following the right-hand
        convention. If "angles" is a single value, all points are rotated by that angle. If "angles" has as many values 
        as "array" has points, each point is rotated by the angle in "angles" of the same index.

        :param array: (n, 2) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: rotated points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 2, 1))
        rot_matrices = self.active_2D(angles)
        return (rot_matrices@array).reshape(n_rot, 2)
        
    @staticmethod
    def _process_rotation(func):
        """Wrapper for the rotations with two functionalities:
        - checks the validity of the angle argument
        - adjusts the rotation matrix if multiple angles are supplied
        """
        def wrapper(angle):
            is_numpy = False
            if isinstance(angle, np.ndarray):
                is_numpy = True
                angle = angle.squeeze()
                if len(angle.shape) > 1:
                    raise ValueError(f"If using '{func.__name__}' with a numpy.ndarray input, the array must "
                                    "be a one dimensional.")
            rot_mat = func(angle)
            if isinstance(angle, (float, int)):
                return rot_mat
            elif is_numpy:
                return rot_mat.transpose(2, 0, 1)
            else:
                raise ValueError(f"The angle(s) for '{func.__name__}()' have to be given as int, float, or "
                                 f"np.ndarray but they were {type(angle)}.")
        return wrapper
    
    @staticmethod
    @_process_rotation
    def passive_2D(angle: float) -> np.ndarray:
        """Passive 2D rotation.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (2, 2) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

    @staticmethod
    @_process_rotation
    def active_2D(angle: float) -> np.ndarray:
        """Active 2D rotation.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (2, 2) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
                
    @staticmethod
    @_process_rotation
    def passive_3D_planar(angle: float) -> np.ndarray:
        """Passive 3D rotation around the axis of the third array index axis.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (3, 3) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.sin(angle), np.zeros_like(angle)],
                           [-np.sin(angle), np.cos(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)]])

    @staticmethod
    @_process_rotation
    def passive_3D_planar_separate(angle: float) -> np.ndarray:
        """Passive 3D rotation (projection) around the axis of the third array index axis. The first coordinate of the
        array will be projected onto the rotated first and second axis. The same holds for the second coordinate of the
        array that will be projected. See project_separate() for an example.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (5, 3) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.zeros_like(angle), np.zeros_like(angle)],
                           [-np.sin(angle), np.zeros_like(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.sin(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.cos(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)]])
    