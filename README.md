# Dimentia Mamba


## Dataset

The dataset must be downloaded from: https://github.com/ipis-mjkim/caueeg-dataset


## Environment Setup

Requirements: Ubuntu 20.04 (project uses WSL) and CUDA 11.8.

1. Create a new Conda environment

```
conda create -n ceednetmamba python=3.10 -y 
```

2. Activate the Conda environment

```
conda activate ceednetmamba
```

3. Install PyTorch

```
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda==11.8 -c pytorch -c nvidia
```

4. Install Causal-conv 1D and Mamba

```
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 --no-cache-dir
```

5. Install the [caueeg-ceednet requirements](https://github.com/ipis-mjkim/caueeg-ceednet) (using their provided requirements.txt file)

```
pip install -r requirements.txt
```

6. If you numpy is at version 2, it must be downgraded

```
pip install --upgrade numpy==1.26.4
```

7. Clone the Dimentia Mamba repository

```
git clone https://github.com/HealthX-Lab/EEG-DimentiaMamba-official.git
```

8. cd into the repo

```
cd EEG-DimentiaMamba
```

9. Extract the dataset using 7z into the data/caueeg/caueeg-dataset subdirectory

```
7z x caueeg-dataset.zip
```

10. Convert the edf files to feather format

```
python ./src/external/caueeg/datasets/convert_file_format.py
```

## How to run

To run the full train/test pipeline, run the train_test.sh script in the scripts/ directory.

## Generate figures

The figures are generated using the occlusions_final.ipynb notebook in the src/notebooks/ directory. Simply run the cells in the notebook to generate the figures in the report. 

## Acknowledgements

This codebase builds upon the [caueeg-ceednet](https://github.com/ipis-mjkim/caueeg-ceednet) and [SiMBA](https://github.com/badripatro/simba) repositories. We thank the authors for making their code publicly available, and we kindly request that you consider citing their works if relevant.