from abc import ABC, abstractmethod
from defaults import DefaultsSimulation
import numpy as np
import pandas as pd
from os.path import join
from copy import copy
from typing import Callable
import json
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks


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
                            #todo add original error message
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
                assert scheme in has_keys, (f"Attribute '{attribute}' of class '{type(self).__name__}' is missing "
                                            f"a key-value pair for scheme '{scheme}'.")
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
        self._check_scheme(scheme)
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
    

def get_inflow(
    t: np.ndarray, 
    ramps: list[tuple[float, float, float]], 
    init_velocity: float=0, 
    init_angle: float=0) -> np.ndarray:
    """Creates a 2D np.ndarray array with the inflow conditions. Assumes constant time step size

    :param t: time array for the simulation with equidistant time steps
    :type t: np.ndarray
    :param ramps: list of tuples of (time begin ramp, time end ramp, velocity, angle). 'time begin ramp' defines when
     the ramp starts, 'time for ramp' how long the ramp takes, and 'velocity' and 'angle' what the velocity and angle 
     are after the ramp.
    :type ramps: list[tuple[float, float, float, float]]
    :return: _description_
    :rtype: np.ndarray
    """
    n_ramps = len(ramps)
    inflow = np.zeros((t.size, 2))
    init_inflow = init_velocity*np.c_[np.cos(np.deg2rad(init_angle)), np.sin(np.deg2rad(init_angle))]
    inflow[0:(t<=ramps[0][0]).argmax(), :] = init_inflow
    last_velocity = init_velocity
    last_angle = init_angle
    for i_ramp, (t_begin, t_end, velocity, angle) in enumerate(ramps):
        ids = np.logical_and(t>=t_begin, t<=t_end)
        n_timesteps = ids.sum()
        velocities = np.linspace(last_velocity, velocity, n_timesteps)
        angles = np.linspace(np.deg2rad(last_angle), np.deg2rad(angle), n_timesteps)
        inflow[ids] = velocities[:, np.newaxis]*np.c_[np.cos(angles), np.sin(angles)]
        last_velocity = velocity
        last_angle = angle

        i_end_current_ramp = ids.argmax()+n_timesteps
        if i_ramp<n_ramps-1:
            i_next_ramp = (t>=ramps[i_ramp+1][0]).argmax()
            inflow[i_end_current_ramp:i_next_ramp, :] = inflow[i_end_current_ramp-1, :]
        else:
            inflow[i_end_current_ramp:, :] = inflow[i_end_current_ramp-1, :]
    return inflow
        

def compress_oscillation(file_motion: str, 
                         file_compressed: str,
                         cols: str|list[str],
                         col_time: str="time", 
                         period_res: int=100,
                         max_rmse: float=1e-4):
    cols = cols if isinstance(cols, list) else [cols]
    df = pd.read_csv(file_motion)
    time = df[col_time].to_numpy()
    compressed = {param: {"t_begin": None,
                          "dt": None,
                          "ampl": None,
                          "freq": None, 
                          "shift": None, 
                          "overall_period": None,
                          "overall_amplitude": None, 
                          "peaks_mean": None,
                          "fft_coeffs_ids": None,
                          "fft_coeffs_re": None,
                          "fft_coeffs_imag": None,
                          "N": None,
                          "rmse": None} 
                          for param in cols}
    for col in cols:
        val = df[col].to_numpy()
        peaks = find_peaks(val)[0]
        n_peaks = find_peaks(-val)[0]

        ids_period = np.arange(peaks[-2], peaks[-1])
        n_interp = ids_period.size
        t_period = time[ids_period]
        val_period = val[ids_period]

        compressed[col]["t_begin"] = float(t_period[0])
        compressed[col]["overall_period"] = float(t_period[-1]-t_period[0])
        compressed[col]["overall_amplitude"] = float((val[peaks[-1]]-val[n_peaks[-1]])/2)
        compressed[col]["peaks_mean"] = float((val[peaks[-1]]+val[n_peaks[-1]])/2)

        t_compression = np.linspace(t_period[0], t_period[-1], period_res)
        v_compression = interp1d(t_period, val_period)(t_compression)

        compressed[col]["dt"] = t_compression[1]-t_compression[0]

        fourier_coeffs = rfft(v_compression)
        amplitude = 2/period_res*np.abs(fourier_coeffs)
        ids_sorted_ampl = np.argsort(amplitude)[::-1]
        for use_up_to in range(1, period_res+1):
            yf_subset = np.zeros_like(fourier_coeffs, dtype=complex)
            ids_subset = ids_sorted_ampl[:use_up_to]
            yf_subset[ids_subset] = fourier_coeffs[ids_subset] 
            
            base_reconstructed = interp1d(t_compression, irfft(yf_subset, n=period_res))(t_period)
            rmse = np.sqrt(((val_period-base_reconstructed)**2).sum()/n_interp)
            if rmse < max_rmse:
                break
            
        if use_up_to == period_res:
            print(f"Max error of '{max_rmse}' could not be reached for oscillation of '{col}'; "
                  f"continuing with all ({period_res}) coefficients and an rmse='{rmse}'.")

        round_at = 5
        freq = np.round(rfftfreq(period_res, compressed[col]["dt"])[ids_subset], round_at)
        amplitudes = amplitude[ids_subset]
        try:
            freq_0 = freq == 0
            if freq_0.sum() > 1:
                raise ValueError(f"More than one zero-frequency fft coefficient found. This is likely due to "
                                 f"rounding the frequencys at the {round_at}th decimal point.")
            idx_mean = freq_0.argmax()
            amplitudes[idx_mean] /= 2
        except ValueError:
            pass
        
        compressed[col]["fft_coeffs_ids"] = ids_subset.tolist()
        compressed[col]["fft_coeffs_re"] = [tmp_yf.real for tmp_yf in fourier_coeffs[ids_subset].tolist()]
        compressed[col]["fft_coeffs_imag"] = [tmp_yf.imag for tmp_yf in fourier_coeffs[ids_subset].tolist()]
        compressed[col]["N"] = period_res
        compressed[col]["ampl"] = amplitudes.tolist()
        compressed[col]["freq"] = freq.tolist()
        compressed[col]["shift"] = np.angle(fourier_coeffs)[ids_subset].tolist()
        compressed[col]["rmse"] = float(rmse)

    with open(file_compressed, "w") as f:
        json.dump(compressed, f, indent=4)
    return compressed


def reconstruct_from_coefficients(N, FFT_coeffs: list[tuple[float, float]]):
    yf = np.zeros(N, dtype=complex)
    for idx, coeff_value in FFT_coeffs:
        yf[idx] = coeff_value
    return irfft(yf, n=N)


def reconstruct_from_file(file: str, separate_mean: bool=False):
    with open(file, "r") as f:
        compressed = json.load(f)
    
    reconstructed = {}
    mean = {}
    for param, info in compressed.items():
        ids = info["fft_coeffs_ids"]
        real_parts = info["fft_coeffs_re"]
        imag_parts = info["fft_coeffs_imag"]
        N = info["N"]
        dt = info["dt"]
        t_begin = info["t_begin"]
        time = t_begin+dt*np.arange(N)
        coeffs = [(idx, re+1j*imag) for idx, re, imag in zip(ids, real_parts, imag_parts)]
        vals = reconstruct_from_coefficients(N, coeffs)
        
        if separate_mean:
            ids_zero = np.asarray(info["freq"])==0
            if ids_zero.sum() > 1:
                raise ValueError(f"More than one frequency=0 found for {param} from {file}.")
            elif ids_zero.sum() == 0:
                mean[param] = 0
            else:
                idx = ids_zero.argmax()
                mean[param] = info["ampl"][idx]*np.cos(info["shift"][idx])
            vals -= mean[param]
        reconstructed[param] = {"time": time, "vals": vals}
    if not separate_mean:
        return reconstructed
    else:
        return reconstructed, mean


def zero_oscillations(time: np.ndarray, periods: int, **kwargs):
    """Takes one-period oscillations that happen at different times and extends (not just shift!) and cuts them such
      that all oscillations start at the same time. If x is an array containing the values of the period and 
      x[0]==x[-1], then x[:-1] has to be given. In other words, the period is not fully closed but the last (the same
      as the first value) must be missing.

    :param time: _description_
    :type time: np.ndarray
    :param periods: _description_
    :type periods: int
    :return: _description_
    :rtype: _type_
    """
    # each kwarg is dict with keys "time" and "vals"
    base_param = [*kwargs.keys()][0]
    base_time = kwargs[base_param]["time"]
    dt = np.round(base_time[1]-base_time[0], 5)  # if dt<e-5 then this fails!
    oscillations = {}
    oscillations[base_param] = kwargs[base_param]["vals"]
    time = time-time[0]
    dt = time[1]
    full_time = time
    for i in range(1, periods):
        full_time = np.r_[full_time, dt+time+full_time[-1]]
    for i, (param, osci_data) in enumerate(kwargs.items()):
        osc_val = osci_data["vals"]
        if i==0:
            oscillations[param] = np.tile(osc_val, periods)
            continue
        osc_time = osci_data["time"]
        dt = osc_time[1]-osc_time[0]
        osc_val = np.r_[osc_val, osc_val]
        if osc_time[0] < base_time[0]:
            osc_time = np.r_[osc_time, osc_time[-1]+dt*np.arange(1, osc_time.size+1)]
        else:
            osc_time = np.r_[osc_time[0]-dt*np.arange(1, osc_time.size+1)[::-1], osc_time]
        oscillations[param] = np.tile(interp1d(osc_time, osc_val)(base_time), periods)
    return full_time, oscillations