# GursonBayes
*A Multitask Bayesian Optimization framework for the identification of GTN constitutive model parameters based on a relatively small number of FE simulations.*

## Overview
This repository contains the code (Python and APDL) and the data for the paper: **"A Multitask Bayesian Optimization framework to identify a Gurson-Tvergaard-Needleman constitutive model for a MS1500 steel in hydrogen environment"** by.  T. Grossi, A. Guazzini, L. Romanelli, C. M. Belardini and B. D. Monelli.

*(under peer review)*

## Requirements
- Python = 3.10
- numpy~=1.26.4
- scipy~=1.14.1
- scikit-learn~=1.5.2
- pandas~=2.2.3
- matplotlib~=3.9.2
- gpytorch~=1.14
- pillow~=10.4.0
- pyansys==2024.2.5


## Usage
1. First open Gurson_Bayes, run *Database_Random_Importer.py* where the variable **num_pairs** select the amount of simulations, taken from the databse with 100 FE simulations, that you want to use for the initial database.
2. Run *Multitask_GPR_Training.py*, where **tol** is the variable related to the variable E0 of the paper
3. Run *TEST_Bay_Opt_Tot.py* where the variable that can be arbitrarily changed are:
   - **mean_order**: change the degree of the error evaluated (default 1, i.e. mean absolute error)
   - **loss_thresh**: the threshold value for stop iterating. Default is -10 ("-" sign because it is resolved a maximization problem). If more accuracy is wanted, use values grater in absolute value, e.g. -5.
   - **num_iterations**: the maximum number of iterations that can be esecuted if **loss_threshold** is not reached.
4. For the reproduction of the **brute-force solution**, set:
     - **num_pairs** = 100 (employ all FE database)
     - **loss_thresh** = -0.1 (because we want that 10 simulations are executed for all the Cd)
     - **num_iterations** = 10.
5. For the reproduction of the tests S1,S2... of the paper set:
     - **num_pairs** = 5 
     - **loss_thresh** = -10 
     - **num_iterations** = 10.
  

## Repository Structure
```
Gurson_Bayes-Python-APDL-codes/
│
├── Gurson_Bayes/
│   ├── Experimental_Curves/              # curves of the un-notched specimen needed by the script
│       ├── *.txt   
│   ├── Database_Random_Importer.py       # Random selection of FE simulations for initial database
│   ├── Multitask_GPR_Training.py         # Multitask Gaussian Process training
│   ├── TEST_Bay_Opt_Tot.py               # Bayesian optimization loop
│   ├── utils/                            # Utility functions and helpers   
│
├── Experimental Data/
│   ├── *.txt                             # Experimental force-displacement curves for all the tests
│
├── Database_MeshFact_1/
│   ├── *.txt                             # FE simulation database files
│
├── ANSYS_Folder/
│   └── ...                               # Working directory for MAPDL runs
|
├── APDL code/ANSYS_APDL_models           # Script APDL for reproducing un-notched and V-notched specimen simulation
|
├── Supplementary Material                # supplementary material for the paper                            
│
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies

```

## Contact
For suggestions, questions, or collaboration opportunities, please contact:
- **T. Grossi**: tommaso.grossi@santannapisa.it
- **A. Guazzini**: alessandro.guazzini@phd.unipi.it
- Open an issue on this repository.


