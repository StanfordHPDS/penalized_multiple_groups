# bring in relevant libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

warnings.filterwarnings("ignore")


## generate the simulated data
def simpop(n: int, ps: list, k: list):
    """
    Generate simulated population

    Args:
        n: Population size
        ps: Distribution parameters, static
        k: Distribution parameters, variable

    Returns:
        X: Coefficients / predictors. n x m.
        Y: Targets / outcomes. n x 1.
        Z: Indicators of group memberships. n x 3.
        probs: Outcome probabilities (used to geneerate outcomes). n x 1.
    """
    ## inputs: number of observations, list of probabilities for binary covariates, list of coefficients for calculating group identities (see text explanation above)
    # generate predictors
    X1 = np.random.normal(30, 15, n)
    X2 = np.random.normal(30, 15, n)
    X3 = np.random.poisson(15, n)
    X4 = np.random.poisson(15, n)
    X5 = np.random.binomial(1, ps[0], n)
    X6 = np.random.binomial(1, ps[1], n)
    X7 = np.random.binomial(1, ps[2], n)
    X8 = np.random.binomial(1, ps[3], n)
    X9 = np.random.binomial(1, ps[4], n)
    # generate group identifiers
    A = np.random.binomial(
        1,
        np.clip(k[0] * (X9 + 0.1) + X6 * k[3] + np.random.normal(0, 0.02, n), 0, 1),
        n,
    )
    B = np.random.binomial(
        1,
        np.clip(k[1] * (X7 + 0.05) + X6 * k[4] + np.random.normal(0, 0.02, n), 0, 1),
        n,
    )
    C = np.random.binomial(
        1,
        np.clip(k[2] * (X8 + 0.4) + X6 * k[5] + np.random.normal(0, 0.02, n), 0, 1),
        n,
    )
    # generate outcome
    ycat = (
        (X1 - 2 * X2) / 30
        + (3 * X3 - 2 * X4) / 50
        + (
            np.power(2, X5 + X6)
            + 6 * X7
            + 10 * X8 * X9
            + 8 * X3 * A
            + 6 * X4 * B
            + C * np.random.normal(60, 2, n)
        )
        / 100
        + np.random.normal(0, 1, n)
    )
    probs = norm.cdf(ycat)
    # combine variables into function outputs
    X = np.vstack((X1, X2, X3, X4, X5, X6, X7, X8, X9)).T
    Y = np.random.binomial(1, probs)[:, np.newaxis]
    Z = np.vstack((A, B, C)).T
    return X, Y, Z, probs


def sim_dat_chars_simp(Name: str, X: np.array, Y: np.array, Z: np.array, p: np.array):
    """
    Generate simulated population table of characteristics

    Args:
        Name: Name of the population.
        X: Coefficients / predictors. n x m.
        Y: Targets / outcomes. n x 1.
        Z: Indicators of group memberships. n x 3.
        probs: Outcome probabilities (used to geneerate outcomes). n x 1.

    Returns:
        simchars: Dataframe with population characteristics.
    """
    # add ref group
    refgrp = 1 - np.max(Z, axis=1)
    Z = np.hstack((Z, refgrp[:, np.newaxis]))
    # make dataframe for characteristics
    simchars = pd.DataFrame({"Population": [Name]})
    # prevalence
    simchars["Y"] = np.mean(Y)
    # group sizes
    for i, g in enumerate(["A", "B", "C"]):
        simchars[g] = np.mean(Z[:, i] == 1)
    # group prevalences
    for i, g in enumerate(["A", "B", "C", "Ref"]):
        simchars[f"prev_{g}"] = np.mean(Y[Z[:, i] == 1])
    # group pred prob
    learned_model = LogisticRegression(random_state=0).fit(X, Y)
    ppred = learned_model.predict_proba(X)[:, 1]
    for i, g in enumerate(["A", "B", "C", "Ref"]):
        simchars[f"TPR_{g}"] = np.mean(ppred[(Z[:, i] == 1) & (Y[:, 0] == 1)])
    return simchars
