"""
    General utility methods.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import io
from . import pickle, LEGACY_MODULES
from typing import Tuple, Union, List

########################################################################################################################


class RenameUnpickler(pickle.Unpickler):
    """
    Custom Unpickler to load pkl object with old and legacy modules.
    """

    def find_class(self, module, name):
        """
        Override the find_class method of pickle, replacing legacy module names with new ones.
        :param module: str; module name.
        :param name: str; class name.
        :return:
        """
        renamed_module = module

        if module in LEGACY_MODULES.keys():
            renamed_module = LEGACY_MODULES[module]

        return super(RenameUnpickler, self).find_class(renamed_module, name)

########################################################################################################################


def renamed_load(file_obj):
    """
    This method is the same as pickle.load().
    :param file_obj: binary file.
    :return: de-serialized object.
    """
    return RenameUnpickler(file_obj).load()

########################################################################################################################


def renamed_loads(pickled_bytes):
    """
    This method is the same as pickle.loads().
    :param pickled_bytes: bytes to unpickle.
    :return: de-serialized object.
    """
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

########################################################################################################################


def min_max_scaler(starting_range: Tuple[Union[float, int]],
                   new_range: Tuple[Union[float, int]],
                   value: float) -> float:
    """
    Scale the input value from a starting range to a new one.
    :param starting_range: tuple of float; the starting range.
    :param new_range: tuple of float; the new range.
    :param value: float; value to be rescaled.
    :return: float; rescaled value.
    """

    assert isinstance(starting_range, tuple) and len(starting_range) == 2, \
        "feature_range must be a tuple as (min, max)"
    assert isinstance(new_range, tuple) and len(new_range) == 2, \
        "feature_range must be a tuple as (min, max)"

    min_start_value = starting_range[0]
    max_start_value = starting_range[1]
    min_new_value = new_range[0]
    max_new_value = new_range[1]

    value_std = (value - min_start_value) / (max_start_value - min_start_value)
    scaled_value = value_std * (max_new_value - min_new_value) + min_new_value

    return scaled_value

########################################################################################################################


def timestamps_headers(num_timeunits: int) -> List[str]:
    """
    Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
    For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
    :param num_timeunits: int; the number of timeunits in a day.
    :return: list of string; list of timeunits.
    """

    start_time = datetime.strptime('00:00', '%H:%M')
    timeunit = 24 * 60 / num_timeunits
    timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
    timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

    return timestamps

########################################################################################################################


def instances_preprocessing(instances: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PV and Load values from string to float.
    :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
    :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
    """

    assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
    assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

    # Instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    # Instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

    return instances


