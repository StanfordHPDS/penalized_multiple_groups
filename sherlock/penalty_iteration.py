### copy over code
# bring in relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime
import os

warnings.filterwarnings("ignore")

## pull
import sys

sys.path.append("code")
from pen_regression import *
from simpop import *



# make a population
n = 1000000
ps = [0.2, 0.4, 0.6, 0.8, 0.95]
alpha_options = [0.2, 0.4, 0.6, 0.8, 0.99]
misspec_options = [0, 1]
k_options = [
    [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
    [0.005, 0.3, 0.3, 0.002, 0.02, 0.02],
    [0.02, 0.02, 0.02, 0.002, 0.002, 0.002],
]
a = 3
n_iter = int(sys.argv[1])
p_cut = 0.5
misspec = misspec_options[int(sys.argv[2])]
kset = int(sys.argv[3])
k = k_options[kset]

# repeat enough times to generate empirical 2 sided 95% CIs

results = []
res = run_sim_pipe(n, k, kset, p_cut, ps, misspec, n_iter)
os.makedirs("results", exist_ok=True)
res.to_csv(f"results/penalty_res_{n_iter}_{misspec}_{kset}_{sys.argv[4]}.csv", index=False)