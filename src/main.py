import os
import json
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['HYDRA_FULL_ERROR'] = '1'
import hydra
import wandb
from omegaconf import OmegaConf as OC
from omegaconf import DictConfig
import torch
import logging
from utils import set_seed, get_num_class_labels, count_parameters, list_param_counts
from utils import set_seed
from train_window import train_script_window
from paths import CAUEEG_FM, CAUEEG_RAW
from data import (
    get_create_filemarkers, 
    load_caueeg_windows_dataset,
    get_window_dataloaders, 
    compose_preprocess,
    compose_transforms
)

from external.caueeg.datasets.caueeg_script import load_caueeg_config

log = logging.getLogger('root')


def run_training_window(cfg):
    """
    Runs the training paradigm that uses the deterministic sliding window dataset.
    """

    # Set the random seed
    seed = cfg.train.seed
    set_seed(seed)
    
    # Check if the filemarkers exist. If not, create them. --------------------
    window_samples = cfg.data.sampling_freq * cfg.data.window_len_s
    window_percent_overlap_float = cfg.data.window_percent_overlap / 100


    file_markers_path = CAUEEG_FM / cfg.data.file_marker_name
    file_markers = get_create_filemarkers(
        cfg=cfg,
        file_markers_path=file_markers_path,
        window_len=window_samples, # Window length is in number of samples
        window_percent_overlap=window_percent_overlap_float,
        latency=cfg.data.latency
    )

    # Get the config for the dataset, mostly for the signal_headers -----------
    config_dataset = load_caueeg_config(CAUEEG_RAW) # Gets dataset name and signal header
    cfg.data.signal_header = config_dataset["signal_header"]

    # Drop Photic and EKG mean and std (if not using)
    drop_indices = []
    if cfg.data.EKG == 'X':
        drop_indices.append(cfg.data.signal_header.index("EKG"))
    if cfg.data.photic == 'X':
        drop_indices.append(cfg.data.signal_header.index("Photic"))

    cfg.data.signal_mean = [i for j, i in enumerate(file_markers["train_mean"][0]) if j not in drop_indices]
    cfg.data.signal_std = [i for j, i in enumerate(file_markers["train_std"][0]) if j not in drop_indices]

    # In channels for the model
    in_channels = cfg.data.num_electrodes
    if cfg.data.EKG == 'X':
        in_channels -= 1
    if cfg.data.photic == 'X':
        in_channels -= 1
    cfg.model.in_channels = in_channels

    # Check that if model.use_age is not 'no', data.use_age is also set to True.
    if cfg.model.use_age != 'no' and cfg.data.use_age == False:
        raise ValueError("Model set to use age, but data.use_age is False.")
    if cfg.model.use_age == 'no' and cfg.data.use_age == True:
        raise ValueError("Model not using age, but data.use_age is True.") 

    # Set the device to GPU (if available) -----------
    if cfg.get("use_cuda", True) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create the transforms needed for the dataset ----------------------------
    transforms_train, _, transform_eval = compose_transforms(cfg, dataset_type="window")
    preprocess_train, preprocess_test = compose_preprocess(cfg, dataset_type="window", device=device)

    # Get the datasets --------------------------------------------------------
    task_dict, datasets = load_caueeg_windows_dataset( 
        CAUEEG_RAW, 
        cfg.data.task,
        file_markers,
        cfg.data.load_event,
        cfg.data.file_format,
        cfg.train.train_random_crop,
        transform_train=transforms_train,
        transform_eval=transform_eval,
        use_freq_bands=False
    )

    # Update the cfg with some info from the task dict
    cfg.train.task_name = task_dict["task_name"]
    cfg.train.task_description = task_dict["task_description"]
    cfg.train.class_label_to_name = task_dict["class_label_to_name"]
    cfg.train.class_name_to_label = task_dict["class_name_to_label"]

    # Print the number of samples in each class
    class_labels = get_num_class_labels(cfg, file_markers, task_dict, random_crop=cfg.train.train_random_crop)
    if cfg.train.train_random_crop:
        log.info(f"The label balance is: {class_labels}")
    else:
        log.info(f"The label (number of patients for train) balance is: {class_labels}")

    # Use the datasets to create the dataloaders
    dataloaders = get_window_dataloaders(cfg, datasets) 

    # Out dims and seq_length for the model
    out_dims = len(cfg.train.class_label_to_name)    
    cfg.model.out_dims = out_dims
    cfg.model.seq_length = cfg.data.sampling_freq * cfg.data.window_len_s

    # Log the updated cfg
    log.info(OC.to_yaml(cfg))

    # Start the wandb job
    if cfg.train.use_wandb:
        wandb.init(
            project="MCIAD",
            config=OC.to_container(cfg, resolve=True)
        )
        wandb.define_metric("train/batch_step")
        wandb.define_metric("train/batch_loss", step_metric="train/batch_step")

    # Init the model
    model = hydra.utils.instantiate(cfg.model)
    log.info(f"Model contains {count_parameters(model)} trainable parameters.")
    list_param_counts(model, to_log=True)

    model.to(device)

    if cfg.train.use_wandb:
        wandb.watch(model, log_freq=10, log="all")

    # Fire off the training and testing script
    metrics = train_script_window(
        cfg,
        model,
        dataloaders,
        transform_train=preprocess_train,
        transform_test=preprocess_test,
        do_testing=cfg.train.do_testing
    )

    # Save the metrics to the output folder
    metrics_fname = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_fname, "w") as fout:
        fout.write(json.dumps(metrics))

    torch.cuda.empty_cache()

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    print(OC.to_yaml(cfg))
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Hydra output directory: {cfg.output_dir}")

    run_mode = cfg.train.run_mode
    run_type = cfg.train.run_type
    data_type = cfg.data.dataset_type

    # Verify that the window dataset is being used with the window train routine.
    if run_type != data_type:
        print("The dataset must be in a format that the training and testing protocol can accept.")
        return

    if run_mode == "train" and run_type == "window":
        trained_model_path = run_training_window(cfg)
        print(trained_model_path)

if __name__ == "__main__":
    main()