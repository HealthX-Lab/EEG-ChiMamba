import os
import json
import pprint
import math
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .caueeg_dataset import CauEegDataset
from .pipeline import eeg_collate_fn


# __all__ = []


def load_caueeg_config(dataset_path: str):
    """Load the configuration of the CAUEEG dataset.

    Args:
        dataset_path (str): The file path where the dataset files are located.
    """
    try:
        with open(os.path.join(dataset_path, "annotation.json"), "r") as json_file:
            annotation = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caueeg_config(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    config = {k: v for k, v in annotation.items() if k != "data"}
    return config


def load_caueeg_full_dataset(dataset_path: str, load_event: bool = True, file_format: str = "edf", transform=None):
    """Load the whole CAUEEG dataset as a PyTorch dataset instance without considering the target task.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Preprocessing process to apply during loading signals.

    Returns:
        The PyTorch dataset instance for the entire CAUEEG dataset.
    """
    try:
        with open(os.path.join(dataset_path, "annotation.json"), "r") as json_file:
            annotation = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caueeg_full(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    eeg_dataset = CauEegDataset(
        dataset_path, annotation["data"], load_event=load_event, file_format=file_format, transform=transform
    )

    config = {k: v for k, v in annotation.items() if k != "data"}

    return config, eeg_dataset


def load_caueeg_task_datasets(
    dataset_path: str, task: str, load_event: bool = True, file_format: str = "edf", transform=None, verbose=False
):
    """Load the CAUEEG datasets for the target benchmark task as PyTorch dataset instances.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        task (str): The target task to load among 'dementia' or 'abnormal'.
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        file_format (str): Determines which file format will be used (default: 'edf').
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the datasets.

    Returns:
        The PyTorch dataset instances for the train, validation, and test sets for the task and their configurations.
    """
    task = task.lower()
    if task not in ["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"]:
        raise ValueError(
            f"load_caueeg_task_datasets(task) receives the invalid task name: {task}. "
            f"Make sure the task name is correct."
        )

    try:
        with open(os.path.join(dataset_path, task + ".json"), "r") as json_file:
            task_dict = json.load(json_file)

        train_dataset = CauEegDataset(
            dataset_path, task_dict["train_split"], load_event=load_event, file_format=file_format, transform=transform
        )
        val_dataset = CauEegDataset(
            dataset_path,
            task_dict["validation_split"],
            load_event=load_event,
            file_format=file_format,
            transform=transform,
        )
        test_dataset = CauEegDataset(
            dataset_path, task_dict["test_split"], load_event=load_event, file_format=file_format, transform=transform
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

    config = {k: v for k, v in task_dict.items() if k not in ["train_split", "validation_split", "test_split"]}

    if verbose:
        print("task config:")
        pprint.pprint(config, compact=True)
        print("\n", "-" * 100, "\n")

        print("train_dataset[0].keys():")
        pprint.pprint(train_dataset[0].keys(), compact=True)

        if torch.is_tensor(train_dataset[0]):
            print("train signal shape:", train_dataset[0]["signal"].shape)
        else:
            print("train signal shape:", train_dataset[0]["signal"][0].shape)

        print()
        print("\n" + "-" * 100 + "\n")

        print("val_dataset[0].keys():")
        pprint.pprint(val_dataset[0].keys(), compact=True)
        print("\n" + "-" * 100 + "\n")

        print("test_dataset[0].keys():")
        pprint.pprint(test_dataset[0].keys(), compact=True)
        print("\n" + "-" * 100 + "\n")

    return config, train_dataset, val_dataset, test_dataset


def load_caueeg_task_split(
    dataset_path: str,
    task: str,
    split: str,
    load_event: bool = True,
    file_format: str = "edf",
    transform=None,
    verbose=False,
):
    """Load the CAUEEG dataset for the specified split of the target benchmark task as a PyTorch dataset instance.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        task (str): The target task to load among 'dementia' or 'abnormal'.
        split (str): The desired dataset split to get among "train", "validation", and "test".
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the dataset.

    Returns:
        A PyTorch dataset instance for the specified split for the task and their configurations.
    """
    task = task.lower()
    if task not in ["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"]:
        raise ValueError(
            f"load_caueeg_task_split(task) receives the invalid task name: {task}. "
            f"Make sure the task name is correct."
        )

    try:
        with open(os.path.join(dataset_path, task + ".json"), "r") as json_file:
            task_dict = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caueeg_task_split(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    if split in ["train", "training", "train_split", "training_split"]:
        dataset = CauEegDataset(
            dataset_path, task_dict["train_split"], load_event=load_event, file_format=file_format, transform=transform
        )
    elif split in ["val", "validation", "val_split", "validation_split"]:
        dataset = CauEegDataset(
            dataset_path,
            task_dict["validation_split"],
            load_event=load_event,
            file_format=file_format,
            transform=transform,
        )
    elif split in ["test", "test_split"]:
        dataset = CauEegDataset(
            dataset_path, task_dict["test_split"], load_event=load_event, file_format=file_format, transform=transform
        )
    else:
        raise ValueError(
            f"ERROR: load_caueeg_task_split(split) needs string among of " f"'train', 'validation', and 'test'"
        )

    config = {k: v for k, v in task_dict.items() if k not in ["train_split", "validation_split", "test_split"]}

    if verbose:
        print(f"{split}_dataset[0].keys():")
        pprint.pprint(dataset[0].keys(), compact=True)

        if torch.is_tensor(dataset[0]):
            print(f"{split} signal shape:", dataset[0]["signal"].shape)
        else:
            print(f"{split} signal shape:", dataset[0]["signal"][0].shape)

        print("\n" + "-" * 100 + "\n")

    return config, dataset


def calculate_signal_statistics(train_loader, preprocess_train=None, repeats=5, verbose=False):
    """
    Repeats are performed here because random cropping is used.
    """
    signal_means = torch.zeros((1,))
    signal_stds = torch.zeros((1,))
    n_count = 0

    for r in range(repeats):
        for i, sample in enumerate(train_loader): 
            if preprocess_train is not None:
                preprocess_train(sample)

            signal = sample["signal"]
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)  # [N, C, L] or [N, (2)C, F, T]

            if r == 0 and i == 0:
                signal_means = torch.zeros_like(mean)
                signal_stds = torch.zeros_like(std)

            signal_means += mean
            signal_stds += std
            n_count += 1

    signal_mean = torch.mean(signal_means / n_count, dim=0, keepdim=True)  # [N, C, L] or [N, (2)C, F, T]
    signal_std = torch.mean(signal_stds / n_count, dim=0, keepdim=True) 
                                                                        
    if verbose:
        print("Mean and standard deviation for signal:")
        pprint.pprint(signal_mean, width=250)
        print("-")
        pprint.pprint(signal_std, width=250)
        print("\n" + "-" * 100 + "\n")

    return signal_mean, signal_std


def calculate_age_statistics(cfg, dataset_path, task, verbose=False):
    """
    Modified from the original CAUEEG code.
    """

    ages = []
    
    # Load the task dict
    with open(os.path.join(dataset_path, task + ".json"), "r") as json_file:
        task_dict = json.load(json_file)


    # Iterate through training dset and collect ages
    for s in task_dict["train_split"]:
        ages.append(s["age"])

    # Find mean and std of ages
    age_t = torch.tensor(ages, dtype=torch.float32)

    std, mean = torch.std_mean(age_t, dim=-1)

    return mean, std



def make_dataloader(config, train_dataset, val_dataset, test_dataset, multicrop_test_dataset, verbose=False):
    if config["device"] == "cpu":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = 0  
        pin_memory = True

    batch_size = config["minibatch"] / config.get("crop_multiple", 1)
    if batch_size < 1 or batch_size % 1 > 1e-12:
        raise ValueError(
            f"ERROR: config['minibatch']={config['minibatch']} "
            f"is not multiple of config['crop_multiple']={config['crop_multiple']}."
        )
    batch_size = round(batch_size)

    multi_batch_size = config["minibatch"] / config.get("test_crop_multiple", 1)
    if multi_batch_size < 1 or multi_batch_size % 1 > 1e-12:
        raise ValueError(
            f"ERROR: config['minibatch']={config['minibatch']} "
            f"is not multiple of config['test_crop_multiple']={config['test_crop_multiple']}."
        )
    config["multi_batch_size"] = round(multi_batch_size)

    if config.get("ddp", False):
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    if config.get("run_mode", None) == "train":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn,
        )
    else: # What is this for?
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
    )

    multicrop_test_loader = None
    if multicrop_test_dataset is not None:
        multicrop_test_loader = DataLoader(
            multicrop_test_dataset,
            batch_size=config["multi_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn,
        )

    if verbose:
        print("train_loader:")
        print(train_loader)
        print("\n" + "-" * 100 + "\n")

        print("val_loader:")
        print(val_loader)
        print("\n" + "-" * 100 + "\n")

        print("test_loader:")
        print(test_loader)
        print("\n" + "-" * 100 + "\n")

        print("multicrop_test_loader:")
        print(multicrop_test_loader)
        print("\n" + "-" * 100 + "\n")

    return train_loader, val_loader, test_loader, multicrop_test_loader