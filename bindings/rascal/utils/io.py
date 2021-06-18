import os
import importlib
from collections import Iterable
import numpy as np
import json
import datetime
from copy import deepcopy
from abc import ABC, abstractmethod

BETA_VERSION = "0.1"

CURRENT_VERSION = BETA_VERSION

MAX_RECURSION_DEPTH = 20


def dump_obj(fn, instance, version=CURRENT_VERSION):
    """Save a python object that inherits from the BaseIO class

    Parameters
    ----------
    fn : string
        path to save instance
    instance : class
        python object that inherits from the BaseIO class
    version : string, optional
        serialization version to use, by default CURRENT_VERSION

    Raises
    ------
    RuntimeError
        When instance does not inherit from BaseIO
    """
    if isinstance(instance, BaseIO):
        to_file(fn, instance, version)
    else:
        raise RuntimeError(
            "The instance does not inherit from BaseIO: {}".format(
                instance.__class__.__mro__
            )
        )


def load_obj(fn):
    """Load a python object from a file

    Parameters
    ----------
    fn : string
        path to the file describing the saved object

    Returns
    -------
    python class that inherits from BaseIO

    """
    return from_file(fn)


def dump_json(fn, data):
    """Utility to save a python object to a file.

    Parameters
    ----------
    fn : string
        filename to save data
    data :
        a json serializable python object
    """
    with open(fn, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2)


def load_json(fn):
    """Utility to load a python object saved in the json format

    Parameters
    ----------
    fn : string
        filename

    Returns
    -------
        loaded python object from fn
    """

    def _decode(o):
        # JSON does not have integer keys so they are converted to string
        # to load the object as it was in python this hook converts to 'int' all
        # dictionary keys that can be converted
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {_decode(k): v for k, v in o.items()}
        else:
            return o

    with open(fn, "r") as f:
        data = json.load(f, object_hook=_decode)
    return data


def is_npy(data):
    """is data a numpy array ?"""
    return isinstance(data, np.ndarray)


def is_large_array(data):
    """is data a numpy array larger than 50MB ?"""
    if is_npy(data):
        if data.nbytes > 50e6:
            return True
        else:
            return False
    else:
        return False


def is_npy_filename(fn):
    """does fn string corresponds to a saved numpy array ?"""
    if isinstance(fn, str):
        filename, file_extension = os.path.splitext(fn)
        if file_extension == ".npy":
            return True
        else:
            return False
    else:
        return False


def get_class(module_name, class_name):
    """Use module_name and class_name to make an instantiable class."""
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def obj2dict_beta(cls, state):
    """Take a python object cls with its state and return a dictionary that
    can be used to create a copy of this object.

    Parameters
    ----------
    cls : object

    state : dictionary
        Contains the state of cls, i.e. the parameters used to initialize cls
        in a 'init_params' field and the rest of the data needed to recover
        the current state in a 'data' field.

    Returns
    -------
    dictionary
        fully serialized version of cls to a dictionary
    """
    VERSION = BETA_VERSION
    module_name = cls.__module__
    class_name = cls.__name__
    frozen = dict(
        version=VERSION,
        class_name=class_name,
        module_name=module_name,
        init_params=state["init_params"],
        data=state["data"],
    )
    return frozen


def dict2obj_beta(data):
    """Take data, a dictionary created by the obj2dict function, and creates
    a python object as described.

    Parameters
    ----------
    data : dictionary
        [description]

    Returns
    -------
    deserialized python object described by data
    """
    cls = get_class(data["module_name"], data["class_name"])
    obj = cls(**data["init_params"])
    obj._set_data(data["data"])
    return obj


def is_valid_object_dict_beta(data):
    """check compatibility of data to be used in dict2obj_beta"""
    valid_keys = [
        "version",
        "class_name",
        "module_name",
        "init_params",
        "data",
    ]
    aa = []
    if isinstance(data, dict):
        for k in data:
            if k in valid_keys:
                aa.append(True)
        if len(aa) == len(valid_keys):
            return True
        else:
            return False
    else:
        return False


obj2dict = {BETA_VERSION: obj2dict_beta}
dict2obj = {BETA_VERSION: dict2obj_beta}
is_valid_object_dict = {BETA_VERSION: is_valid_object_dict_beta}


def get_current_io_version():
    return CURRENT_VERSION


def get_supported_io_versions():
    return list(dict2obj.keys())


class BaseIO(ABC):
    """Interface of a Python class serializable by to_dict()

    It corresponds to 3 methods:

    + _get_init_params is expected to return a dictionary containing all the
    parameters used by the __init__() methods.

    + _get_data is expected to return a dictionary containing all the data
    that is not set by the initialization of the class.

    + _set_data is expected to set the data that has been extracted by _get_data

    The underlying c++ objects are not pickle-able so deepcopy does not work out
    of the box. This class provides an override of the __deepcopy__() function
    so that classes that inherit from this base class can be deepcopied.
    """

    @abstractmethod
    def _get_data(self):
        return dict()

    @abstractmethod
    def _set_data(self, data):
        pass

    @abstractmethod
    def _get_init_params(self):
        return dict()

    def __deepcopy__(self, memo=None):
        """Overrides deepcopy default behaviour with custom serialization
        instead of using pickle."""
        return from_dict(to_dict(self))

    def __setstate__(self, state):
        """Overrides default pickling behaviour passing through the dict representation."""
        obj = from_dict(state)
        self.__dict__.update(obj.__dict__)

    def __getstate__(self):
        """Overrides default pickling behaviour passing through the dict representation."""
        return to_dict(self)


def _get_state(obj):
    if isinstance(obj, BaseIO):
        state = dict(data=obj._get_data(), init_params=obj._get_init_params())
    else:
        raise ValueError(
            'input object: "{}" does not inherit from "BaseIO"'.format(obj)
        )
    return state


def to_dict(obj, version=CURRENT_VERSION, recursion_depth=0):
    """Recursively serialize to dict via the BaseIO interface.

    obj has to inherit from BaseIO."""
    if recursion_depth >= MAX_RECURSION_DEPTH:
        raise ValueError(
            "The object to be serialized to dict contains more than {}".format(
                MAX_RECURSION_DEPTH
            )
            + " levels of nested objects suggesting there is a circular reference."
            + " Objects containing a reference to themselves are not supported."
        )
    else:
        recursion_depth += 1

    state = _get_state(obj)

    # loop over the 2 fields of state
    for name, entry in state.items():
        if isinstance(entry, dict):
            # case of potentially nested objects
            for k, v in entry.items():
                if isinstance(v, BaseIO):
                    state[name][k] = to_dict(v, version, recursion_depth)
                elif isinstance(v, list):
                    # make sure list of objects are properly serialized
                    ll = []
                    for val in v:
                        if isinstance(val, BaseIO):
                            ll.append(to_dict(val, version, recursion_depth))
                        else:
                            ll.append(val)
                    state[name][k] = ll
    data = obj2dict[version](obj.__class__, state)
    return data


def from_dict(data):
    """Recursirvely deserialize from dict via the BaseIO interface."""
    # temporary dictionary to hold the object being recovered
    data_obj = dict()
    version = data["version"]
    for name, entry in data.items():
        if isinstance(entry, dict):
            data_obj[name] = dict()
            for k, v in entry.items():
                if is_valid_object_dict[version](v):
                    # in case of nested objects
                    data_obj[name][k] = from_dict(v)
                elif isinstance(v, list):
                    # in case of list make sure to handle list of serialized
                    # objects
                    ll = []
                    for val in v:
                        if is_valid_object_dict[version](val):
                            ll.append(from_dict(val))
                        else:
                            ll.append(val)
                    data_obj[name][k] = ll
                else:
                    # just transfer the data
                    data_obj[name][k] = v
        else:
            # just transfer the data
            data_obj[name] = entry

    obj = dict2obj[version](data_obj)
    return obj


def to_file(fn, obj, version=CURRENT_VERSION):
    """Saves the object 'obj' to a file named 'fn'.

    It uses the to_dict() serialization procedure."""
    fn = os.path.abspath(fn)
    filename, file_extension = os.path.splitext(fn)
    data = to_dict(obj, version=version)
    class_name = data["class_name"].lower()

    if file_extension == ".json":
        _dump_npy(fn, data, class_name)
        dump_json(fn, data)
    else:
        raise NotImplementedError("Unknown file extention: {}".format(file_extension))


def from_file(fn):
    """Loads an object that was saved using to_file() from a file"""
    fn = os.path.abspath(fn)
    path = os.path.dirname(fn)
    filename, file_extension = os.path.splitext(fn)
    if file_extension == ".json":
        data = load_json(fn)
        version = data["version"]
        if is_valid_object_dict[version](data):
            _load_npy(data, path)
            return from_dict(data)
        else:
            raise RuntimeError(
                "The file: {}; does not contain a valid dictionary".format(fn)
                + " representation of an object."
            )

    else:
        raise NotImplementedError("Unknown file extention: {}".format(file_extension))


def _dump_npy(fn, data, class_name):
    """Saves numpy array to the object file.

    If the array is large (>50MB) main file contains a relative path to the *.npy
    file so that it can be loaded properly.
    Small numpy array are converted to lists and saved in the main file."""
    filename, file_extension = os.path.splitext(fn)
    for k, v in data.items():
        if isinstance(v, dict):
            if "class_name" in data:
                class_name = data["class_name"].lower()
            _dump_npy(fn, v, class_name)
        elif is_large_array(v):
            if "tag" in data:
                class_name += "-" + data["tag"]
            v_fn = filename + "-{}-{}".format(class_name, k) + ".npy"
            v_bfn = os.path.basename(v_fn)
            data[k] = v_bfn
            np.save(v_fn, v)

        elif is_npy(v):
            data[k] = ["npy", v.tolist()]


def _load_npy(data, path):
    """Loads a numpy array saved using _dump_npy().

    A large array stored in a different file is mmaped so it is physically
    loaded only when needed."""
    for k, v in data.items():
        if isinstance(v, dict):
            _load_npy(v, path)
        elif is_npy_filename(v):
            data[k] = np.load(os.path.join(path, v), mmap_mode="r")
        elif isinstance(v, list):
            if len(v) == 2:
                if "npy" == v[0]:
                    data[k] = np.array(v[1])


class RascalEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "todict"):
            d = obj.todict()

            if not isinstance(d, dict):
                raise RuntimeError(
                    "todict() of {} returned object of type {} "
                    "but should have returned dict".format(obj, type(d))
                )
            if hasattr(obj, "ase_objtype"):
                d["__ase_objtype__"] = obj.ase_objtype

            return d
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime.datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, complex):
            return {"__complex__": (obj.real, obj.imag)}
        return json.JSONEncoder.default(self, obj)


def json_dumps_frame(frames, **json_dumps_kwargs):
    """Serialize frames to a JSON formatted string.

    Parameters
    ----------
    frames : list(ase.Atoms) or ase.Atoms
        List of atomic structures (or single one) to be dumped to a json

    json_dumps_kwargs : dict
        List of arguments forwarded to json.dumps

    Return
    ------
    T
    """
    if type(frames) is not list:
        frames = [frames]

    json_frames = {}
    for i, frame in enumerate(frames):
        json_frames[str(i)] = json.loads(json.dumps(frame, cls=RascalEncoder))

    json_frames["ids"] = list(range(len(frames)))
    json_frames["nextid"] = len(frames)

    return json.dumps(json_frames, **json_dumps_kwargs)
