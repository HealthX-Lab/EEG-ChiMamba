import os
import json
from copy import deepcopy
import pyedflib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset


class CauEegDataset(Dataset):
    """PyTorch Dataset Class for CAUEEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        data_list (list of dict): List of dictionary for the data.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(
            self, 
            root_dir: str, 
            data_list: list, 
            load_event: bool, 
            file_format: str = "edf", 
            transform=None, 
            alt_signal_root=None
        ):
        if file_format not in ["edf", "feather", "memmap", "np"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )

        self.data = np.random.rand(105, 1400000)

        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.file_format = file_format
        self.transform = transform

        if alt_signal_root:
            self.signal_root = alt_signal_root
        else:
            self.signal_root = self.root_dir
        print(f"Reading signals from {self.signal_root}")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx]) # age is contained in here

        sample["signal"] = self._read_signal(sample, bands=False)


        # event
        if self.load_event:
            sample["event"] = self._read_event(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_signal(self, anno, bands=False):
        if self.file_format == "edf":
            return self._read_edf(anno)
        elif self.file_format == "feather":
            return self._read_feather(anno, bands=bands)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        edf_file = os.path.join(self.signal_root, f"signal/edf/{anno['serial']}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    def _read_feather(self, anno, bands=False):
        if bands:
            fname = os.path.join(self.signal_root, f"signal/bands/{anno['serial']}.feather")
        else:
            fname = os.path.join(self.signal_root, f"signal/feather/{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_memmap(self, anno):
        fname = os.path.join(self.signal_root, f"signal/memmap/{anno['serial']}.dat")
        signal = np.memmap(fname, dtype="int32", mode="r").reshape(21, -1)
        return signal

    def _read_np(self, anno):
        fname = os.path.join(self.signal_root, f"signal/{anno['serial']}.npy")
        return np.load(fname)

    def _read_event(self, m):
        fname = os.path.join(self.root_dir, "event", m["serial"] + ".json")
        with open(fname, "r") as json_file:
            event = json.load(json_file)
        return event


class CauEegWindowDataset(Dataset):
    """PyTorch Dataset Class for CAUEEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        data_list (list of dict): List of dictionary for the data.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(
            self, 
            root_dir: str, 
            data_list: list, 
            timestamps:list, 
            file_format: str = "edf", 
            load_event=False, 
            transform=None, 
            use_freq_bands=False,
            alt_signal_root=None
        ):
        if file_format not in ["edf", "feather", "memmap", "np"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )

        self.root_dir = root_dir
        self.data_list = data_list
        self.timestamps = timestamps
        self.file_format = file_format
        self.load_event = load_event
        self.transform = transform
        self.use_freq_bands = use_freq_bands

        
        if alt_signal_root:
            self.signal_root = alt_signal_root
        else:
            self.signal_root = self.root_dir
        print(f"Reading signals from {self.signal_root}")


        self.data_dict = {}
        for d in self.data_list:
            self.data_dict[d["serial"]] = d


    def get_num_subjects(self):
        return len(self.data_list)

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):

        timestamp = self.timestamps[idx]
        serial = timestamp["serial"]

        # Get the annotations
        sample = deepcopy(self.data_dict[serial])

        # Attach the timestamp
        sample["timestamp"] = timestamp["times"]

        # Get the signal
        sample["signal"] = self._read_signal(sample, bands=False)

        # event
        if self.load_event:
            sample["event"] = self._read_event(serial)

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def _read_signal(self, anno, bands=False):
        if self.file_format == "edf":
            return self._read_edf(anno)
        elif self.file_format == "feather":
            return self._read_feather(anno, bands=bands)

    def _read_edf(self, serial):
        edf_file = os.path.join(self.signal_root, f"signal/edf/{serial}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    
    def _read_feather(self, anno, bands=False):
        if bands:
            fname = os.path.join(self.signal_root, f"signal/bands/{anno['serial']}.feather")
        else:
            fname = os.path.join(self.signal_root, f"signal/feather/{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_event(self, serial):
        fname = os.path.join(self.root_dir, "event", serial + ".json")
        with open(fname, "r") as json_file:
            event = json.load(json_file)
        return event

