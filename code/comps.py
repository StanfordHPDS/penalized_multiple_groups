import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import datetime
import sys
import os

sys.path.append("code")
from simpop import *

### comparison method packages
## reductions from fairlearn
from fairlearn.reductions import (
    TruePositiveRateParity,
    ExponentiatedGradient,
    GridSearch,
)


## function to check comparison methods
def comp_evaluator(
    fitted_model,
    X_test: np.array,
    y_test: np.array,
    Z_test: np.array,
    kset: int = 0,
    a: int = 3,
    method: str = "ExpGrad",
    time: datetime.timedelta = 0,
):
    """
    Evaluate a given comparison method.

    Args:
        fitted_model: Fitted estimator from the comparison method.
        X_test: Covariates for prediction, test population.
        y_test: Target values, test population.
        Z_test: Group membership indicators, test population.
        kset: Indicator of simulated population distribution.
        a: Number of groups.
        method: Name of comparison method.
        time: runtime for fitting the comparison method.

    Returns:
        model_outputs: Dataframe of characteristics of comparison method and performance.
    """
    # generate predictions
    y_pred = fitted_model.predict(X_test)

    # get accuracy for actual model
    acc = np.mean(y_test == y_pred)

    y_true_ref = y_test[Z_test[:, a] == 1]
    y_pred_ref = y_pred[Z_test[:, a] == 1]
    ref_frac1 = np.sum(np.multiply(y_pred_ref, y_true_ref)) / np.sum(y_true_ref)
    # get opportunity differences
    ng = []
    tpr_dif = []
    for i in range(a):
        # subset to group data
        y_true_z = y_test[Z_test[:, i] == 1]
        y_pred_z = y_pred[Z_test[:, i] == 1]
        # calculate TPRs, take differences from reference
        tpr_dif += [
            min(
                np.sum(np.multiply(y_pred_z, y_true_z)) / np.sum(y_true_z) - ref_frac1,
                0,
            )
        ]
        ng += [np.sum(Z_test[:, i] == 1)]
    mean_unfair = np.sum(np.multiply(np.array(tpr_dif), np.array(ng))) / sum(ng)
    max_unfair = min(tpr_dif)
    print(tpr_dif)
    model_outputs = pd.DataFrame(
        {
            "method": [method],
            "k_set": [kset],
            "accuracy": [acc],
            "mean_unfairness": [mean_unfair],
            "max_unfairness": [max_unfair],
            "A_unfairness": [tpr_dif[0]],
            "B_unfairness": [tpr_dif[1]],
            "C_unfairness": [tpr_dif[2]],
            "runtime": [time],
        }
    )
    return model_outputs


def run_comp_pipe(
    n: int = 1000000,
    k: list = [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
    kset: int = 0,
    ps: list = [0.2, 0.4, 0.6, 0.8, 0.95],
):
    """
    Pipeline to run all the comparison methods and evaluate them.

    Args:
        n: Simulated population size
        k: Simulated population generation parameters.
        kset: Index for simulated population distribution.
        ps: Simulated population generation parameters.

    Returns:
        model_outputs: Dataframe of characteristics of comparison method and performance.
    """
    # make simulated population
    X, Y, Z, _ = simpop(n, ps, k)
    Y = np.squeeze(Y)
    runtimes = []

    ### fit the comparison methods
    ## reduction: exponentiated gradient
    redfes = []
    for ep in [0.1, 0.01]:
        startime = datetime.datetime.now()
        exp_grad_est = ExponentiatedGradient(
            estimator=LogisticRegression(solver="liblinear", fit_intercept=True),
            constraints=TruePositiveRateParity(difference_bound=ep),
        )
        redfes += [exp_grad_est.fit(X, Y, sensitive_features=Z)]
        runtimes += [(datetime.datetime.now() - startime)]

    ## reduction: gridsearches
    gsfes = []
    for cw in [0.25, 0.5, 0.75]:
        startime = datetime.datetime.now()
        grid = GridSearch(
            estimator=LogisticRegression(solver="liblinear", fit_intercept=True),
            constraints=TruePositiveRateParity(difference_bound=.01),
            constraint_weight=cw,
            grid_size=128,
        )
        grid.fit(X, Y, sensitive_features=Z)
        gsfes += [grid]
        runtimes += [(datetime.datetime.now() - startime)]

    fitted_estimators = redfes + gsfes

    # fresh sim pull and evaluate
    X_test, Y_test, Z_test, Y_p = simpop(n, ps, k)
    refgrp = 1 - np.max(Z_test, axis=1)
    Z_test = np.hstack((Z_test, refgrp[:, np.newaxis]))
    Y_test = np.squeeze(Y_test)

    method_names = ["ExpGrad.1", "ExpGrad.01", "RedGrid25", "RedGrid5", "RedGrid75"]
    results_list = []
    for i, est in enumerate(fitted_estimators):
        results_list += [
            comp_evaluator(
                est, X_test, Y_test, Z_test, kset, 3, method_names[i], runtimes[i]
            )
        ]

    return pd.concat(results_list)


## Code below is for command line calls!
# preliminaries
n = 1000
k_options = [
    [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
    [0.005, 0.3, 0.3, 0.002, 0.02, 0.02],
    [0.02, 0.02, 0.02, 0.002, 0.002, 0.002],
]
kset = int(sys.argv[1])

# call
res = run_comp_pipe(n, k_options[kset], kset, [0.2, 0.4, 0.6, 0.8, 0.95])
os.makedirs("results", exist_ok=True)
res.to_csv(f"results/comp_res_{kset}_{sys.argv[2]}.csv", index=False)
