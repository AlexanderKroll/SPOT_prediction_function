# Description
This repository contains an easy-to-use Python function for the SPOT prediction model from our paper [SPOT: A machine learning model that predicts specific substrates for transport proteins](https://doi.org/10.1371/journal.pbio.3002807). 


## Downloading data folder
Before you can run the ESP prediction function, you need to [download and unzip a data folder from Zenodo](https://doi.org/10.5281/zenodo.8046233). Afterwards, this repository should have the following strcuture:

    ├── code                   
    ├── data                    
    └── README.md

## How to use the ESP prediction function
There is a Jupyter notebook "using_SPOT.ipynb" in the folder "code" that contains an example on how to use the SPOT prediction function. If a GPU is available, code is executed on GPU.

## Requirements

The following packages were installed

```bash
conda create -n SPOT python=3.11
conda activate SPOT

pip install rdkit-pypi==2022.9.5
pandas==2.2.0
xgboost==2.0.3
transformers==4.37.2
fair-esm==0.4.2
pip install torch==2.5.1
```

## Problems/Questions
If you face any issues or problems, please open an issue.

