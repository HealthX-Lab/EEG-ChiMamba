import pprint
import os
import sys
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from paths import CAUEEG_RAW
import json
import mne
from pathlib import Path
import pyedflib
import numpy as np
import pyarrow.feather as feather
from tqdm.auto import tqdm
import logging
from omegaconf import OmegaConf as OC
from external.caueeg.datasets.caueeg_script import load_caueeg_task_datasets, make_dataloader, calculate_signal_statistics, load_caueeg_config
from external.caueeg.datasets.caueeg_script import load_caueeg_task_split, calculate_age_statistics
from external.caueeg.datasets.pipeline import EegRandomCrop
from external.caueeg.datasets.pipeline import EegNormalizeMeanStd, EegNormalizePerSignal, EegNormalizeAge
from external.caueeg.datasets.pipeline import EegDropChannels
from external.caueeg.datasets.pipeline import EegAdditiveGaussianNoise, EegMultiplicativeGaussianNoise, EegAddGaussianNoiseAge
from external.caueeg.datasets.pipeline import EegToTensor, EegToDevice
from external.caueeg.datasets.pipeline import eeg_collate_fn
from external.caueeg.datasets.caueeg_dataset import CauEegWindowDataset, CauEegDataset

mne.set_log_level('WARNING')
from mne.utils._logging import verbose
from mne.filter import _check_filterable, _check_method, create_filter

log = logging.getLogger(__file__)

def standardize_array(arr, ax, set_mean=None, set_std=None, return_mean_std=False):
    """
    Taken from: https://github.com/mackelab/neural_timeseries_diffusion
    """

    if set_mean is None:
        arr_mean = np.mean(arr, axis=ax, keepdims=True)
    else:
        arr_mean = set_mean
    if set_std is None:
        arr_std = np.std(arr, axis=ax, keepdims=True)
    else:
        arr_std = set_std

    assert np.min(arr_std) > 0.0
    if return_mean_std:
        return (arr - arr_mean) / arr_std, arr_mean, arr_std
    else:
        return (arr - arr_mean) / arr_std


class EegDeterministicCrop(object):
    """Crop the signal based on a start and stop timestamp (timestamp in samples).

    Args:

    """
    def __init__(self, bands=False):
        self.bands = bands
        return

    def __call__(self, sample):

        start_sample = sample["timestamp"][0]
        stop_sample = sample["timestamp"][1]

        signal = sample['signal']

        if self.bands:
            sample['signal'] = signal[:, :, start_sample:stop_sample] # [Channels, Bands, Timesteps]
        else:
            sample['signal'] = signal[:, start_sample:stop_sample]

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"



def compose_transforms(cfg, dataset_type="caueeg"):
    """
    Transforms to use for the single epoch dataset (no sliding window). These are applied to the data in 
    the dataset __getitem__ method.
    """

    assert dataset_type in ["caueeg", "window"]

    transform = []
    transform_eval = []
    transform_multicrop = []

    if dataset_type == "caueeg":
        transform += [
            EegRandomCrop(
                crop_length=cfg["seq_length"],
                length_limit=cfg.get("signal_length_limit", 10**7),
                multiple=cfg.get("crop_multiple", 1),
                latency=cfg.get("latency", 0),
                segment_simulation=cfg.get("segment_simulation", False),
                return_timing=cfg.get("crop_timing_analysis", False)
            )
        ]
        transform_multicrop += [
            EegRandomCrop(
                crop_length=cfg["seq_length"],
                length_limit=cfg.get("signal_length_limit", 10**7),
                multiple=cfg.get("test_crop_multiple", 8),
                latency=cfg.get("latency", 0),
                segment_simulation=cfg.get("segment_simulation", False),
                return_timing=cfg.get("crop_timing_analysis", False)
            )
        ]

    elif dataset_type == "window" and not cfg.train.train_random_crop:
        transform += [
            EegDeterministicCrop(bands=False)
        ]
        transform_eval += [
            EegDeterministicCrop(bands=False)
        ]

    elif dataset_type == "window" and cfg.train.train_random_crop :
        transform += [
            EegRandomCrop(
                crop_length=int(cfg.data.sampling_freq * cfg.data.window_len_s),
                length_limit=cfg.train.signal_length_limit,
                multiple=1,
                latency=cfg.data.latency,
                segment_simulation=False,
                return_timing=cfg.train.return_timing,
                bands=False
            )
        ]
        transform_eval += [
            EegDeterministicCrop(bands=False)
        ]



    # Drop or keep EKG and Photic channels
    if dataset_type == "caueeg":
        channel_ekg = cfg["signal_header"].index("EKG")
        channel_photic = cfg["signal_header"].index("Photic")

        use_ekg = cfg["EKG"]
        use_photic = cfg["photic"]

    elif dataset_type == "window":
        channel_ekg = cfg.data.signal_header.index("EKG")
        channel_photic = cfg.data.signal_header.index("Photic")

        use_ekg = cfg.data.EKG
        use_photic = cfg.data.photic

    if use_ekg == "O" and use_photic == "O":
        pass
    elif use_ekg == "O" and use_photic == "X":
        transform += [EegDropChannels([channel_photic])]
        transform_multicrop += [EegDropChannels([channel_photic])]
        transform_eval += [EegDropChannels([channel_photic])]

    elif use_ekg == "X" and use_photic == "O":
        transform += [EegDropChannels([channel_ekg])]
        transform_multicrop += [EegDropChannels([channel_ekg])]
        transform_eval += [EegDropChannels([channel_ekg])]

    elif use_ekg == "X" and use_photic == "X":
        transform += [EegDropChannels([channel_ekg, channel_photic])]
        transform_multicrop += [EegDropChannels([channel_ekg, channel_photic])]
        transform_eval += [EegDropChannels([channel_ekg, channel_photic])]
    else:
        raise ValueError(f"Both config['EKG'] and config['photic'] have to be set to one of ['O', 'X']")


    # Convert from numpy to torch tensor
    transform += [EegToTensor()]
    transform_multicrop += [EegToTensor()]
    transform_eval += [EegToTensor()]

    # Compose into torch transform
    transform = transforms.Compose(transform)
    transform_multicrop = transforms.Compose(transform_multicrop)
    transform_eval = transforms.Compose(transform_eval)


    return transform, transform_multicrop, transform_eval

def compose_preprocess(cfg, train_loader=None, verbose=False, dataset_type="caueeg", device=None):

    assert dataset_type in ["caueeg", "window"]

    preprocess_train = []
    preprocess_test = []

    # dataset specific formatting of the cfg
    get_mean = False
    if dataset_type == "caueeg":
        device=cfg["device"]
        input_norm = cfg["input_norm"]
        if "signal_mean" not in cfg or "signal_std" not in cfg:
            get_mean = True

        run_mode = cfg.get("run_mode", None)
        awgn = cfg.get("awgn", None)
        mgn = cfg.get("mgn", None) 

    elif dataset_type == "window":
        input_norm = cfg.data.input_norm
        run_mode = cfg.train.run_mode
        awgn = cfg.data.awgn
        mgn = cfg.data.mgn

    # to device
    preprocess_train += [EegToDevice(device=device)]
    preprocess_test += [EegToDevice(device=device)]

    # data normalization (age) ######
    if cfg.data.use_age and cfg.train.run_mode == "train":
        if (cfg.data.age_mean is None) and (cfg.data.age_std is None):
            age_mean, age_std = calculate_age_statistics(cfg, dataset_path=CAUEEG_RAW, task=cfg.data.task, verbose=False)
        preprocess_train += [EegNormalizeAge(mean=age_mean, std=age_std)]
        preprocess_test += [EegNormalizeAge(mean=age_mean, std=age_std)]
        cfg.data.age_mean = age_mean.item()
        cfg.data.age_std = age_std.item()

    # additive Gaussian noise for augmentation (age) 
    if cfg.data.use_age and cfg.train.run_mode == "train":
        if cfg.data.awgn_age > 0.0:
            preprocess_train += [EegAddGaussianNoiseAge(mean=0.0, std=cfg.data.awgn_age)]
        else:
            raise ValueError(f"config['awgn_age'] have to be None or a positive floating point number")
    #######

    # data normalization (1D signal) 
    if input_norm == "dataset":
        if get_mean and dataset_type == "caueeg":
            cfg["signal_mean"], cfg["signal_std"] = calculate_signal_statistics(
                train_loader, repeats=5, verbose=False
            )
        if dataset_type == "caueeg":
            preprocess_train += [EegNormalizeMeanStd(mean=cfg["signal_mean"], std=cfg["signal_std"])]
            preprocess_test += [EegNormalizeMeanStd(mean=cfg["signal_mean"], std=cfg["signal_std"])]
        elif dataset_type == "window":
            np_mean = np.array(OC.to_container(cfg.data.signal_mean))
            np_std = np.array(OC.to_container(cfg.data.signal_std))
            preprocess_train += [EegNormalizeMeanStd(mean=np_mean, std=np_std)]
            preprocess_test += [EegNormalizeMeanStd(mean=np_mean, std=np_std)]
    
    elif input_norm == "datapoint":
        log.info("Using the DATAPOINT normalization strategy.")
        preprocess_train += [EegNormalizePerSignal()]
        preprocess_test += [EegNormalizePerSignal()]
    elif input_norm == "minmax":
        raise ValueError("MinMax Norm not yet implemented.")
    elif input_norm == "no":
        pass
    else:
        raise ValueError(f"cfg['input_norm'] have to be set to one of ['dataset', 'datapoint', 'no']")


    # multiplicative Gaussian noise for augmentation (1D signal) 
    if run_mode == "eval":
        pass
    elif mgn is None:
        pass
    elif mgn == "None":
        pass
    elif mgn > 0.0:
        preprocess_train += [EegMultiplicativeGaussianNoise(mean=0.0, std=mgn)]
    else:
        raise ValueError(f"config['mgn'] have to be None or a positive floating point number")

    # additive Gaussian noise for augmentation (1D signal) 
    if run_mode == "eval":
        pass
    elif awgn is None:
        pass
    elif awgn == "None":
        pass
    elif awgn > 0.0:
        preprocess_train += [EegAdditiveGaussianNoise(mean=0.0, std=awgn)]
    else:
        raise ValueError(f"cfg['awgn'] have to be None or a positive floating point number")
    
    # Compose All at Once
    preprocess_train = transforms.Compose(preprocess_train)
    preprocess_train = torch.nn.Sequential(*preprocess_train.transforms)

    preprocess_test = transforms.Compose(preprocess_test)
    preprocess_test = torch.nn.Sequential(*preprocess_test.transforms)

    if verbose:
        print("preprocess_train:", preprocess_train)
        print("\n" + "-" * 100 + "\n")

        print("preprocess_test:", preprocess_test)
        print("\n" + "-" * 100 + "\n")

    return preprocess_train, preprocess_test


def get_caueeg_dataloaders(cfg, verbose=False):
    """
    Heavily adapted from the CAUEEG-CEEDNET repo.
    """

    # Update the cfg with info from the dataset jsons
    config_dataset = load_caueeg_config(CAUEEG_RAW)
    pprint.pprint(config_dataset)
    cfg.update(**config_dataset)

    # Prepare the transforms for the datasets
    transform, transform_multicrop = compose_transforms(cfg)

    # Get the datasets
    config_task, train_dataset, val_dataset, test_dataset = load_caueeg_task_datasets( 
        dataset_path=CAUEEG_RAW,
        task=cfg["task"],
        load_event=cfg["load_event"],
        file_format=cfg["file_format"],
        transform=transform
    )

    print(f"Train dset length: {len(train_dataset)}")
    print(f"Val dset length: {len(val_dataset)}")
    print(f"Test dset length: {len(test_dataset)}")

  
    _, multicrop_test_dataset = load_caueeg_task_split( # Returns config, dset
        dataset_path=CAUEEG_RAW,
        task=cfg["task"],
        split="test",
        load_event=cfg["load_event"],
        file_format=cfg["file_format"],
        transform=transform_multicrop,
        verbose=verbose,
    )

    cfg.update(**config_task)

    train_loader, val_loader, test_loader, multicrop_test_loader = make_dataloader(
        cfg, train_dataset, val_dataset, test_dataset, multicrop_test_dataset, verbose=False
    )


    preprocess_train, preprocess_test = compose_preprocess(cfg, train_loader, verbose=verbose)

    # Apply the preprocessing transforms to the data
    in_channels = cfg["num_electrodes"]
    if cfg["EKG"] == 'X':
        in_channels -= 1
    if cfg["photic"] == 'X':
        in_channels -= 1
    cfg["preprocess_train"] = preprocess_train
    cfg["preprocess_test"] = preprocess_test
    cfg["in_channels"] = in_channels
    cfg["out_dims"] = len(cfg["class_label_to_name"])

    return train_loader, val_loader, test_loader, multicrop_test_loader


def load_caueeg_windows_dataset(
        dataset_path: str, 
        task: str, 
        file_markers, 
        load_event: bool, 
        file_format: str = "feather",
        random_crop=False,
        pred_len=None,
        transform_train=None,
        transform_eval=None,
        use_freq_bands=False,
        alt_signal_root=None 
    ):
    

    task = task.lower()
    if task not in ["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"]:
        raise ValueError(
            f"load_caueeg_task_datasets(task) receives the invalid task name: {task}. "
            f"Make sure the task name is correct."
        )

    try:
        with open(os.path.join(dataset_path, task + ".json"), "r") as json_file:
            task_dict = json.load(json_file)

        datasets = {}


        if random_crop:
            datasets["train"] = CauEegDataset( 
                root_dir=dataset_path,
                data_list=task_dict["train_split"],
                load_event=load_event,
                file_format=file_format,
                transform=transform_train,
                use_freq_bands=use_freq_bands,
                alt_signal_root=alt_signal_root
            )
        else:
            datasets["train"] = CauEegWindowDataset(
                root_dir=dataset_path,
                data_list=task_dict["train_split"],
                timestamps=file_markers["train_split"],
                file_format=file_format,
                load_event=load_event,
                transform=transform_train,
                use_freq_bands=use_freq_bands,
                alt_signal_root=alt_signal_root
            )

            

        datasets["validation"] = CauEegWindowDataset(
            root_dir=dataset_path,
            data_list=task_dict["validation_split"],
            timestamps=file_markers["validation_split"],
            file_format=file_format,
            load_event=load_event,
            transform=transform_eval,
            use_freq_bands=use_freq_bands,
            alt_signal_root=alt_signal_root
        )
        datasets["test"] = CauEegWindowDataset(
            root_dir=dataset_path,
            data_list=task_dict["test_split"],
            timestamps=file_markers["test_split"],
            file_format=file_format,
            load_event=load_event,
            transform=transform_eval,
            use_freq_bands=use_freq_bands,
            alt_signal_root=alt_signal_root
        )
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caueeg_task_datasets(dataset_path={dataset_path}) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise
    except ValueError as e:
        print(f"ERROR: load_caueeg_task_datasets(file_format={file_format}) encounters an error of {e}.")
        raise


    return task_dict, datasets


def get_window_dataloaders(cfg, datasets: dict):
    
    num_workers = cfg.train.num_workers
    if cfg.train.use_cuda:
        pin_memory = True
    else:
        pin_memory = False

    batch_sz = cfg.model.minibatch


    dataloaders = {}
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_size=batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn
    )

    dataloaders["validation"] = DataLoader(
        datasets["validation"],
        batch_size=batch_sz,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn
    )

    dataloaders["test"] = DataLoader(
        datasets["test"],
        batch_size=batch_sz,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn
    )

    return dataloaders

def get_create_filemarkers(cfg, file_markers_path: str, window_len, window_percent_overlap, latency, signal_root=CAUEEG_RAW):
    """
    Looks for the window filemarkers. If the file markers do not exist, it creates them.
    """

    # If the file marker path is given and exists, load it then return the file
    pth = Path(file_markers_path)
    file_markers = None
    try:
        pth.resolve()
        with open(pth) as f:
            file_markers = json.load(f)
    except FileNotFoundError: # Else, create the filemarkers
        print("File marker with the following path: ")
        print(pth)
        print("not found. Creating file markers.")
        file_markers = create_window_filemarkers(cfg, pth, window_len, window_percent_overlap, latency, signal_root=signal_root)

    return file_markers
    


def create_window_filemarkers(cfg, file_markers_path, window_len, window_percent_overlap, latency, signal_root=CAUEEG_RAW):
    split_file_markers = {
        "train_split" : None,
        "validation_split" : None,
        "test_split" : None
    }

    # Since there is no file at the file_markers_path location, we will create it, then save it at that location

    # Open the task dict
    task = cfg.data.task
    with open(os.path.join(CAUEEG_RAW, task + ".json"), "r") as json_file:
        task_dict = json.load(json_file)
    
    print("Generating the filemarkers.")
    for split in ["train_split", "validation_split", "test_split"]:
        split_windows = []
        signals = [] # Used in the training set to calculate mean and std
        for subject_info in task_dict[split]:

            # Load the subjects data
            if cfg.data.file_format == "feather":
                signal = read_feather(signal_root, subject_info["serial"]) # Signal in shape [Num_channels, timepoints]
            elif cfg.data.file_format == "edf":
                signal = read_edf(signal_root, subject_info["serial"])

            # Split into window
            subject_windows = get_window_timestamps( # [{serial: 123 , times: (start, stop)}, ...]
                serial=subject_info["serial"],
                signal=signal, 
                latency=latency,
                window_len=window_len,
                window_percent_overlap=window_percent_overlap
            )

            split_windows.extend(subject_windows)

        
        split_file_markers[split] = split_windows

    # Get the mean and std of the trainset filemarkers to save along with the filemarkers
    print("Calculating mean and std of training data.")
    train_mean, train_std = get_train_mean_std(
        split_file_markers["train_split"], 
        file_format=cfg.data.file_format, 
        num_channels=cfg.data.num_electrodes,
        signal_root=signal_root
    )
    split_file_markers["train_mean"] = train_mean.tolist()
    split_file_markers["train_std"] = train_std.tolist()


    # Save the filemarkers
    with open(file_markers_path, "w") as out:
        json.dump(split_file_markers, out)

    return split_file_markers


def get_window_timestamps(serial, signal, latency, window_len, window_percent_overlap):

    timestamps = []

    stride = int(window_len * (1 - window_percent_overlap))
    last_start_index = signal.shape[1] - window_len - 1 # This is the last possible index a window could be extracted from
    
    for start in range(latency, last_start_index, stride):
        timestamps.append({
            "serial": serial,
            "times": (start, start + window_len)
        })

    return timestamps

def get_train_mean_std(train_filemarkers, file_format="feather", num_channels=21, signal_root=CAUEEG_RAW):

    # idx_count = 0
    count = 0
    signal_sum = np.zeros((num_channels))
    signal_sum_sqrt = np.zeros((num_channels))
    for clip in tqdm(train_filemarkers):
        if file_format == "feather":
            signal = read_feather(signal_root, clip["serial"])
        elif file_format == "edf":
            signal = read_edf(signal_root, clip["serial"])

        #all_cropped.append(signal[:, clip["times"][0]:clip["times"][1]])
        clipped_signal = signal[:, clip["times"][0]:clip["times"][1]]
        signal_sum = signal_sum + clipped_signal.sum(axis=-1)
        signal_sum_sqrt += (clipped_signal**2).sum(axis=-1)
        count += clipped_signal.shape[-1]

        
        # if idx_count == 0:
        #     idx_means = np.zeros_like
        # idx_count += 1

    total_mean = signal_sum / count
    total_var = (signal_sum_sqrt / count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    total_mean = np.expand_dims(np.expand_dims(total_mean, 0), -1)
    total_std = np.expand_dims(np.expand_dims(total_std, 0), -1)

    return total_mean, total_std

def read_edf(data_dir, serial):
    edf_file = os.path.join(data_dir, f"signal/edf/{serial}.edf")
    signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
    return signal

def read_feather(data_dir, serial):
    fname = os.path.join(data_dir, f"signal/feather/{serial}.feather")
    df = feather.read_feather(fname)
    return df.values.T