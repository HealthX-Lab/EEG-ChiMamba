from pathlib import Path
import random
import os
from datetime import datetime
import logging
import mne
from mne.io import RawArray
import torch
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def check_create_directory(path, log=False):
    """
    Checks if a directory exists. If it doesnt, makes it.
    """

    pth = Path(path)
    if not pth.exists():
        pth.mkdir(parents=True)
        if log:
            print(f"Directory created at {str(pth.resolve())}")

    return str(pth.resolve())


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    np.random.seed(s)
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


def get_num_class_labels(cfg, filemarkers, task_dict, random_crop=False):

    class_samples = {}
    class_samples["train_split"] = {k:0 for k, _ in cfg.train.class_name_to_label.items()}
    class_samples["validation_split"] = {k:0 for k, _ in cfg.train.class_name_to_label.items()}
    class_samples["test_split"] = {k:0 for k, _ in cfg.train.class_name_to_label.items()}

    for split in filemarkers:

        if "split" not in split:
            continue

        # Build the serial to class dict
        data_dict = {}

        
        if random_crop and split == "train_split":
            for d in task_dict[split]:
                class_samples["train_split"][d["class_name"]] += 1
            continue # Go to the next split

        for d in task_dict[split]:
            data_dict[d["serial"]] = d

        # Go through the samples in the current split
        for f in filemarkers[split]:

            label = data_dict[f['serial']]['class_label'] # 0, 1 or 2
            class_name = cfg.train.class_label_to_name[label]

            class_samples[split][class_name] += 1

    return class_samples

class CheckpointManager:
    """
    Inspired by https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch and heavily edited
    """
    def __init__(self, base_epochs, patience, model_name, max_delta=0.00, min_delta=0.0001, save_path=None):
        self.base_epochs = base_epochs # Number of epochs to run before running the checkpointer
        self.patience = patience
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.save_path = save_path # The path the model is saved to
        self.model_name = model_name

        self.early_epoch_counter = base_epochs
        self.patience_counter = 0
        self.best_score = np.Inf
        self.stopped = False
        self.latest_save_path = None
        self.first_save = False

        self.old_models = []
        print("Max Delta:", self.max_delta)

    def _save_model(self, model, optimizer):

        now = datetime.now()
        save_name = f"{now.strftime('%m_%d_%Y_%H_%M_%S')}_{self.model_name}.tar"
        save_path = os.path.join(self.save_path, "saved_models")
        check_create_directory(save_path)
        filename = os.path.join(save_path, save_name)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename)

        log.info(f"[CHECKPOINT] Model saved to:  {filename}")

        if self.latest_save_path is not None:
            self.old_models.append(self.latest_save_path)

        self.latest_save_path = filename

    def save_overtrained_model(self, model, optimizer):

        now = datetime.now()
        save_name = f"{now.strftime('%m_%d_%Y_%H_%M_%S')}_{self.model_name}_overtrained.tar"
        save_path = os.path.join(self.save_path, "saved_models")
        check_create_directory(save_path)
        filename = os.path.join(save_path, save_name)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename)

        log.info(f"[CHECKPOINT] Model saved to:  {filename}")

    def early_stop(self, val_loss, model, optimizer):

        
        if self.early_epoch_counter == 0:

            
            if val_loss < self.best_score - self.min_delta:
                self.best_score = val_loss
                log.info("[CHECKPOINT] Validation loss decrease. Saving model.")
                self._save_model(model, optimizer)

                # Reset the counter since we progressed on the loss
                self.patience_counter = 0

            elif val_loss > self.best_score + self.max_delta:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info("[CHECKPOINT] Validation loss unchanged for PATIENCE epochs. Stopping training.")
                    self.stopped = True
                    return True
        else:
            self.early_epoch_counter -= 1

    def delete_old_models(self):
        for state_dict in self.old_models:
            log.info(f"Removing: {state_dict}")
            os.remove(state_dict)

def plot_eeg(signal_batch, channel_names):

    rand = random.randint(0, signal_batch.shape[0] - 1)

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=200,
        ch_types='eeg'
    )
    raw = RawArray(signal_batch[rand].numpy(), info)
    raw.plot(show=True)
    plt.show()

def plot_eeg_np(signal, channel_names=None, title="None"):

    if not channel_names:
        channel_names = [str(x) for x in range(signal.shape[0])]

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=200,
        ch_types='eeg'
    )
    raw = RawArray(signal, info)
    raw.plot(show=True, title=title)
    #plt.show()

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.

    From: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L1128

    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

def count_parameters(model):
    """
    From CAUEEG repo.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list_param_counts(model, to_log=False):

    for name, p in model.named_parameters():
        if p.requires_grad:
            if not to_log:
                print(f"{name} {p.numel()}")
            else:
                log.info(f"{name} {p.numel()}")

def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
