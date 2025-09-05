# Balancing-Access-and-Utilisation-in-HWRC-Closure-Sequencing-The-Hampshire-Case
Thesis Repository includes data, results and codes, contains code for facility assignment optimization and analysis of closure strategies. It uses mixed-integer programming to model assignments, evaluate accessibility, and compute financial savings under different closure scenarios.

---

## Repository structure

### Folders
- **data/**  
  Contains input data such as:  
  - `Sector dataset.xlsx` (user and facility attributes)  
  - GeoJSON files for mapping  

- **own_results/**  
  Stores all generated results including:  
  - Assignment Excel files (`Result_O_x.xlsx`, `Result_NO_x.xlsx`)  
  - Figures (`figure1.png`, `figure2.png`, etc.)  
  - Tables (`table5.xlsx`)  

---

### Modules

- **utils.py**  
  Provides helper functions:  
  - Create and save distance/travel dictionaries  
  - Urban/rural travel probability functions  
  - Tier-based saving dictionary  
  - Load and prepare input data  

- **model.py**  
  Core optimization model:  
  - Pyomo MIP formulation with Gurobi solver  
  - Objective functions and constraints  
  - Capacity, budget, and assignment logic  

- **results.py**  
  Functions for experiment batches:  
  - `optimal_results`: sequential closure strategy  
  - `non_optimal_results`: Tier-based closure benchmark  
  - Saves outputs into structured Excel files  

- **plotting.py**  
  Helper functions to visualize assignments and statistics.  

- **main.py**  
  Entry point:  
  - Loads `Sector dataset.xlsx`  
  - Creates and saves distance/travel dictionaries  
  - Runs both optimal and non-optimal results  

- **tables_and_figures.py**  
  Script to reproduce the study’s outputs:  
  - Generates Figures 1–4  
  - Builds Table 5  
  - Stores results under `own_results/tables and figures/`  

---

## Running the project

Run baseline experiments:

    python main.py

This produces assignment results in:

    own_results/All Assignment/

Generate tables and figures:

    python tables_and_figures.py

Outputs appear in:

    own_results/tables and figures/

---

## Requirements

The following Python packages are used across the modules:

    pyomo
    gurobipy   # requires valid license
    pandas
    numpy
    geopy
    matplotlib

Standard library modules used:

    json
    bz2
    pickle
    os
    shutil
    random
