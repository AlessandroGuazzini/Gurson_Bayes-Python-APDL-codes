# GursonBayes

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

*A multitask Bayesian optimization framework for the identification of Gurson–Tvergaard–Needleman (GTN) constitutive model parameters using a limited number of finite element simulations.*

## Overview

This repository provides the Python and APDL codes, together with the datasets, associated with the paper:

**“A multitask Bayesian optimization framework for the identification of a Gurson–Tvergaard–Needleman constitutive model for MS1500 steel in a hydrogen environment”**  
T. Grossi, A. Guazzini, L. Romanelli, C. M. Belardini, B. D. Monelli

*(currently under peer review)*

The proposed framework combines multitask Gaussian Process regression and Bayesian optimization to efficiently identify GTN model parameters from a reduced set of finite element (FE) simulations.

## Installation & Requirements

### For `uv` users (recommended)
If you have [uv](https://github.com/astral-sh/uv) installed, you can replicate the environment and sync all dependencies directly from `pyproject.toml`:
```powershell
uv sync
.\.venv\Scripts\activate
```
This will create a `.venv` with all necessary packages (Python 3.10+) and activate it.

### For `pip` users
Alternatively, you can use `pip` with the `requirements.txt` file by creating and activating a virtual environment manually:
```powershell
# Create the environment
python -m venv .venv

# Activate the environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.10
- numpy ~= 1.26.4
- scipy ~= 1.14.1
- scikit-learn ~= 1.5.2
- pandas ~= 2.2.3
- matplotlib ~= 3.9.2
- gpytorch ~= 1.13
- pillow ~= 10.4.0
- pyansys ~= 2024.2.0 (Note: should match your local Ansys version as per PyAnsys instructions)

## Usage

1. Open the `Gurson_Bayes` folder and run `Database_Random_Importer.py`.  
   The variable **`num_pairs`** controls the number of FE simulations randomly selected from the full database (100 simulations) to construct the initial training dataset.

2. Run `Multitask_GPR_Training.py`.  
   The parameter **`tol`** corresponds to the variable E_0 defined in the paper.

3. Run `TEST_Bay_Opt_Tot.py`.  
   The main user-defined parameters are:
   - **`mean_order`**: order of the error metric (default = 1, i.e. mean absolute error).
   - **`loss_thresh`**: stopping threshold for the optimization loop.  
     The default value is −10 (negative sign due to the maximization formulation).  
     For higher accuracy, use values with smaller absolute magnitude (e.g. −5).
   - **`num_iterations`**: maximum number of optimization iterations if the loss threshold is not reached.

4. To reproduce the **brute-force solution**, set:
   - **`num_pairs` = 100** (entire FE database),
   - **`loss_thresh` = −0.1** (ensures 10 simulations are executed for each Cd value),
   - **`num_iterations` = 10**.

5. To reproduce tests **S1, S2, …** presented in the paper, set:
   - **`num_pairs` = 5**,  
   - **`loss_thresh` = −10**,  
   - **`num_iterations` = 10**.

## Repository Structure


```
Gurson_Bayes-Python-APDL-codes/
│
├── Gurson_Bayes/
│   ├── Experimental_Curves/              # Un-notched specimen curves used by the scripts
│   │   └── *.txt
│   ├── Database_Random_Importer.py       # Random selection of FE simulations for initial database
│   ├── Multitask_GPR_Training.py         # Multitask Gaussian Process training
│   ├── TEST_Bay_Opt_Tot.py               # Bayesian optimization loop
│   └── utils/                            # Utility functions
│
├── Experimental Data/
│   └── *.txt                             # Experimental force–displacement curves
│
├── Database_MeshFact_1/
│   └── *.txt                             # FE simulation database
│
├── ANSYS_Folder/
│   └── ...                               # Working directory for MAPDL runs
│
├── APDL code/ANSYS_APDL_models           # APDL scripts for un-notched and V-notched specimens
│
├── Supplementary Material                # Supplementary material for the paper
│
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies


```


## Contact

For questions, suggestions, or collaboration opportunities, please contact:

- **T. Grossi** — tommaso.grossi@santannapisa.it  
- **A. Guazzini** — alessandro.guazzini@phd.unipi.it  

Alternatively, open an issue in this repository.



