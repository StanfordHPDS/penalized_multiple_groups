# Penalized Fair Regression for Multiple Groups in Chronic Kidney Disease

This repository contains the code for penalized fair regression with respect to multiple groups and for implementing it in simulated data and also for end stage renal disease prediction among chronic kidney disease patients in the [American Family Cohort](https://www.jabfm.org/content/30/4/559). 

The corresponding publication is available on [arxiv](https://arxiv.org/abs/2512.17340).

Data citation and location: Stanford Center for Population Health Sciences (2024). AFC OMOP RIF. Redivis. DOI:10.71778/V2DW-7A53. 

## Setting up this repository

Running `esrd_analytic.qmd` and rendering `penalty_writeup.pdf` require access to the Nero Google Cloud Platform developed by PHS, Stanford University School of Medicine, and Stanford Research Computing Center. Follow these [steps](https://github.com/StanfordHPDS/nero_setup) to set up a Nero instance on which files in this repository can be run. 

## Navigating this repository

```bash
├── README.md
├── _extensions
├── code
│   ├── comps.py
│   ├── pen_regression.py
│   ├── simpop.py
├── sherlock
│   ├── comp_batch.sbatch
│   ├── comps_caller.sbatch
│   ├── grid_of_grids.sbatch
│   ├── indiv_batch.sbatch
│   ├── penalty_iteration.py
│   ├── result_combiner.py
│   ├── result_combiner.sbatch
│   ├── vary_draws.sbatch
├── tables
├── appendix.pdf
├── appendix.qmd
├── esrd_analytic.html
├── esrd_analytic.qmd
├── ESRD_fullpipe.sql
├── fairpenalties.bib
├── Makefile
├── penalty_writeup.pdf
├── penalty_writeup.qmd
├── pyproject.toml
└── uv.lock
```

The `code` directory contains scripts for performing the analysis described in `penalty_writeup.pdf`. The primary script of interest is `code/pen_regression.py`, which contains the functions to perform the proposed novel penalized regression (including code to calculate the score function and perform the corresponding random search for penalty weights). Additionally, `code/simpop.py` contains functions that generate the simulated data and `code/comps.py` contains functions to run the comparison methods. The code should be run with `uv` commands corresponding to this repository. 

The analyses in this project can be broken down into the synthetic analyses, which were conducted on Stanford's research computing platform Sherlock, and the CKD progression analysis in the AFC data, which were conducted through the Stanford Nero Google Cloud Platform. 

### Simulations on Sherlock

First, the package `fairlearn` must be installed to run `comps_caller.sbatch`. Instructions for package installation on the platform are [here](https://www.sherlock.stanford.edu/docs/software/using/python/#installing-packages), although it may be preferable to use the command `“python3.12 -m pip install --user fairlearn”`. 
To conduct the Sherlock analyses, move the `code` folder and the contents of the `sherlock` folder onto the research computing platform then run the following commands: 

```bash
sbatch comps_caller.sbatch
sbatch grid_of_grids.sbatch
sbatch vary_draws.sbatch
```

Each command will run a batch of jobs, each of which will generate a single result CSV. These CSV files were combined using `result_combiner.sbatch` after each batch of jobs and renamed to create files `comp_res.csv` (corresponding to `comps_caller.sbatch`), `drawcounts_{i}.csv` (corresponding to `vary_draws.sbatch`), and `penalty_res_{i}.csv` (corresponding to `grid_of_grids.sbatch`), which are used in the manuscript. `vary_draws.sbatch` must be called 5 times to achieve the 100 replications used in the published analysis. 

### CKD Analysis

Conducting the CKD analysis requires access to the AFC data through the Stanford Nero Google Cloud Platform, set up as described above. We also draw on eGFR data from Foryciarz et al (2025), medRxiv:10.1101/2025.04.03.25325206 (code [here](https://github.com/StanfordHPDS/data_transformation/tree/8d065724e64392f2560adb6726779081884ce4c5/egfr_microsim/cohort/sql_scripts)). Given correct file access, the first step in the CKD analysis is running the SQL script `ESRD_fullpipe.sql` to generate a cohort. 

SQL code is run on BigQuery, a web platform for interacting with the data that is part of the Nero Cloud Platform. As with Nero, VPN connection is required to access BigQuery. The SQL file included references data files that are part of a particular AFC project. 

Once the SQL code has been run, the CKD analyses can be performed. The command 

```bash
make all
```

will perform the CKD analysis (`esrd_analytic.html`), generate the scientific manuscript (`penalty_writeup.pdf`) based on the CKD analyses and simulations, and generate the corresponding appendix (`appendix.pdf`). This command uses the Makefile to regenerate  and all necessary files (except Sherlock-generated tables). Running this command requires access to the AFC data through the Stanford Nero Google Cloud Platform. 

It is possible to regenerate the manuscript with summary data tables without AFC data access (and therefore without redoing the CKD analyses) with the following command: 

```bash
uv run quarto render penalty_writeup.qmd
```

Similarly, it is possible to generate the appendix with summary data tables without AFC data access (and therefore without redoing the CKD analyses) with the following command: 

```bash
uv run quarto render penalty_writeup.qmd
```

The CKD analyses can be performed alone in the Nero Cloud Platform with AFC data access to generate the summary data tables using the command

```bash
uv run quarto render esrd_analytic.qmd
```

Package dependencies can by found in `pyproject.toml`, with additional code version information in `uv.lock`.
