import pandas as pd
import numpy as np
import random
import scipy
import sys
import warnings
import datetime

sys.path.append("code")
from simpop import *

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import loguniform
from typing import Optional

warnings.filterwarnings("ignore")


## generate the preliminaries for score function
def calc_score_parts(
    y_true: np.array,
    y_pred: np.array,
    X: np.array,
    Z: np.array,
    X_train: np.array,
    y_train: np.array,
    p_cut: float = 0.5,
    a: int = 3,
):
    """
    Calculates key intermediaries of all score functions

    Args:
        y_true: Target labels, test population.
        y_pred: Target predictions from estimator being scored, test population.
        X: Covariates for prediction, test population.
        Z: Group membership indicators, test population.
        X_train: Covariates for prediction, training population.
        y_train: Target predictions from estimator being scored, training population.
        p_cut: Classification threshold.
        a: Number of groups.

    Returns:
        acc: Scalar accuracy term for estimator of interest [0,1].
        acc_lr: Scalar accuracy term for unpenalized logistic regression.
        abs_op_dif: Array of TPR disparities for each group for estimator of interest.
        abs_op_dif_lr: Array of TPR disparities for each group for unpenalized logistic regression.
        ng: Array of population sizes for each group.
    """

    # generate reference predictions
    lr = LogisticRegression(fit_intercept=False).fit(X_train, y_train)
    y_lr_pred = ((lr.predict_proba(X)[:, 1] > p_cut) * 1)[:, np.newaxis]

    # accuracy terms
    acc = np.mean(y_true == y_pred)
    acc_lr = np.mean(y_true == y_lr_pred)

    ## calculate metrics for reference group
    # subset to group data
    yr = y_true * (Z[:, a][:, np.newaxis])
    y_lr_pred_ref = y_lr_pred * (Z[:, a][:, np.newaxis])
    y_pred_ref = y_pred * (Z[:, a][:, np.newaxis])
    # generate actual fractions for calculating
    ref_frac1 = np.sum(np.multiply(y_pred_ref, yr)) / np.sum(yr)
    ref_frac2 = np.sum(np.multiply(y_lr_pred_ref, yr)) / np.sum(yr)

    # get opportunity differences
    abs_op_dif = []
    abs_op_dif_lr = []

    # make group outcome and prediction matrices
    yg = np.stack([np.squeeze(y_true) * (Z[:, i]) for i in range(a)])
    y_lr_pred_z = np.stack([np.squeeze(y_lr_pred) * (Z[:, i]) for i in range(a)])
    y_pred_z = np.stack([np.squeeze(y_pred) * (Z[:, i]) for i in range(a)])
    ng = np.sum(Z, axis=0)[:a]

    # calculate TPRs, take differences from reference
    frac1 = np.sum(np.multiply(y_pred_z, yg), axis=1) / np.sum(yg, axis=1)
    abs_op_dif = np.maximum(ref_frac1 - frac1, np.zeros(3))
    frac1 = np.sum(np.multiply(y_lr_pred_z, yg), axis=1) / np.sum(yg, axis=1)
    abs_op_dif_lr = np.maximum(ref_frac2 - frac1, np.zeros(3))

    # calculate the corresponding score
    return acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng


def combine_score(
    acc: float,
    acc_lr: float,
    abs_op_dif: np.array,
    abs_op_dif_lr: np.array,
    ng: np.array,
    alpha: float = 0.5,
    combiner: str = "weighted",
):
    """
    Calculates particular score function using key intermediaries

    Args:
        acc: Scalar accuracy term for estimator of interest [0,1].
        acc_lr: Scalar accuracy term for unpenalized logistic regression.
        abs_op_dif: Array of TPR disparities for each group for estimator of interest.
        abs_op_dif_lr: Array of TPR disparities for each group for unpenalized logistic regression.
        ng: Array of population sizes for each group.
        alpha: Fairness score weight [0,1].
        combiner: Name of combination method of group-level fairness terms (weighted, unweighted, maximum)

    Returns:
        score: Scalar score for given penalty weight set to be maximized across sets.
    """

    # pick a way to combine across groups and do it
    if combiner == "weighted":
        op_difs = np.sum(abs_op_dif * ng)
        op_difs_lr = np.sum(abs_op_dif_lr * ng)
    elif combiner == "unweighted":
        op_difs = np.sum(abs_op_dif)
        op_difs_lr = np.sum(abs_op_dif_lr)
    else:
        op_difs = np.max(abs_op_dif)
        op_difs_lr = np.max(abs_op_dif_lr)

    # calculate the corresponding score
    return -(
        alpha * op_difs / max(op_difs_lr, 1e-3)
        + (1 - alpha) * (acc_lr - acc) / max(acc_lr - 0.5, 1e-3)
    )


## custom estimator for penalized regression
# performs preprocessing, fitting (including conversion to CSC and to weighted classification), prediction, and scoring (using score helpers)
class CustomRegression(BaseEstimator):
    def __init__(
        self,
        *,
        l1: float = 0,
        l2: float = 0,
        l3: float = 0,
        l4: float = 0,
        l5: float = 0,
        l6: float = 0,
        l7: float = 0,
        a: int = 3,
        alpha: float = 0.5,
        p_cut: float = 0.5,
    ):
        """
        Initialize custom regression object

        Args:
            l_: Penalty weight coefficient for the _th group.
            a: Number of groups.
            alpha: Fairness score weight.
            p_cut: Classification threshold.
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.l7 = l7
        self.a = a
        self.alpha = alpha
        self.p_cut = p_cut

    def preprocess(self, rhs: np.array):
        """
        Preprocess 'right hand side' data. Auxillary function not meant to be called by user but used by other class functions.

        Args:
            rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership.

        Returns:
            X: Array of covariates / predictors with intercept column.
            Z: Array of indicators of group and reference membership, in that order.
            n: Number of observations.
        """
        # preprocessing: separate out group identifiers
        Z = rhs[:, -(self.a + 1) :]
        X = rhs[:, : -(self.a + 1)]

        # preprocessing: add intercept column
        n = X.shape[0]
        X = np.hstack((X, np.ones((n, 1))))

        return X, Z, n

    def fit(self, rhs: np.array, y: np.array):
        """
        Fit the estimator using training data.

        Args:
            rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership.
            y: Array of binary target / outcome values.
        """

        # gather, shape the data
        X, Z, n = self.preprocess(rhs)
        self.X_ = X
        self.y_ = y
        self.Z_ = Z

        # Base requirements: establish information
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        # arrange penalties, formatted into list
        lg = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7][: self.a]
        lg = [np.float32(l) for l in lg]
        self.lg = lg

        # create data transformations; costs, weights
        Y = np.squeeze(y)
        C1 = (
            1
            - Y
            + Y
            * (
                sum(lg) * Z[:, self.a] / np.mean(Y * Z[:, self.a])
                - sum([l * Z[:, i] / np.mean(Y * Z[:, i]) for i, l in enumerate(lg)])
            )
        )
        W = np.abs(Y - C1)
        Ytemp = Y > C1

        # fit weighted logistic regression
        lr = LogisticRegression(fit_intercept=False)
        lr.fit(X, Ytemp, sample_weight=W)
        self.penlr_ = lr

        # Return the estimator with the parameters saved
        return self

    def predict_proba(self, rhs: np.array):
        """
        Generate predicted probabilities for the target / outcome

        Args:
            rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership. (Note that the group memberships are not actually used for the prediction)

        Returns:
            Array of predicted probabilities
        """
        # Check if fit has been called
        check_is_fitted(self, ["is_fitted_", "penlr_"])

        X, Z, n = self.preprocess(rhs)

        # Input validation
        X = check_array(X)

        return self.penlr_.predict_proba(X)[:, 1]

    def predict(self, rhs: np.array, p_cut: Optional[float] = None):
        """
        Generate predicted probabilities for the target / outcome

        Args:
            rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership. (Note that the group memberships are not actually used for the prediction)
            p_cut: Classification threshold. If none provided, the threshold used to initialize the estimator will be used.

        Returns:
            Array of binary predictions (indicators of predicted probability exceeding the classification threshold).
        """
        if p_cut is not None:
            self.p_cut = p_cut

        # Check if fit has been called
        check_is_fitted(self, ["is_fitted_", "penlr_"])

        X, Z, n = self.preprocess(rhs)

        # Input validation
        X = check_array(X)

        return (self.penlr_.predict_proba(X)[:, 1] > self.p_cut) * 1

    def score(
        self,
        rhs: np.array,
        y: np.array,
        alpha: Optional[float] = None,
        p_cut: Optional[float] = None,
    ):
        """
        Generate the score for a given set of penalty weights.

        Args:
            rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership.
            y: Array of binary target / outcome values.
            alpha: Fairness score weight. If none provided, the threshold used to initialize the estimator will be used.
            p_cut: Classification threshold. If none provided, the threshold used to initialize the estimator will be used.

        Returns:
            Scalar score value.
        """

        if p_cut is None:
            p_cut = self.p_cut
        if alpha is None:
            alpha = self.alpha
        y_pred = self.predict(rhs, p_cut=p_cut)[:, np.newaxis]

        X, Z, n = self.preprocess(rhs)

        # get score preliminaries, then calc score
        acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng = calc_score_parts(
            y, y_pred, X, Z, self.X_, self.y_, p_cut, self.a
        )

        return combine_score(acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng, alpha)


# custom cross-validator for weird score functions
def random_search_cv_custom(
    model_class,
    rhs: np.array,
    y: np.array,
    param_distributions: dict,
    n_iter: int = 10,
    cv_folds: int = 5,
    p_cut: float = 0.5,
    a: int = 3,
):
    """
    Performs Random Search Cross-Validation from scratch.

    Args:
        model_class: The class of your machine learning model (e.g., CustomRegression).
        rhs: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership.
        y: Target / outcome labels.
        param_distributions: Dictionary where keys are hyperparameter names and values are lists or scipy.stats distribution objects of possible values for the hyperparameters.
        n_iter: Number of penalty weight candidate sets to sample.
        cv_folds: Number of folds for K-fold cross-validation.
        p_cut: Classification threshold.
        a: Number of groups.

    Returns:
        A tuple containing dictionaries of the best parameters and the best score for each score function variation.
    """

    # make different score functions
    keys = []
    for al in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        for ver in ["weighted", "unweighted", "max"]:
            keys += [f"{ver}_{al}"]

    # initialize objects to return
    best_scores = dict(zip(keys, [-np.inf for _ in range(len(keys))]))
    best_params = dict(zip(keys, [{} for _ in range(len(keys))]))

    for _ in range(n_iter):
        ## 1. Sample random hyperparameter combination
        current_params = {}
        for param, values in param_distributions.items():
            if type(values) is list:
                current_params[param] = random.choice(values)
            elif type(values) is scipy.stats._distn_infrastructure.rv_continuous_frozen:
                current_params[param] = values.rvs()
            else:
                current_params[param] = values()

        ## 2. Perform K-fold cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True)
        fold_scores = dict(zip(keys, [[] for _ in range(len(keys))]))
        for train_index, test_index in kf.split(rhs):
            rhs_train, rhs_test = rhs[train_index], rhs[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # get predictions and other data for fold
            model = model_class(**current_params)
            model.fit(rhs_train, y_train)
            y_pred = model.predict(rhs_test, p_cut=p_cut)[:, np.newaxis]
            X_train, _, _ = model.preprocess(rhs_train)
            X_test, Z_test, _ = model.preprocess(rhs_test)
            # get the score components
            acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng = calc_score_parts(
                y_test, y_pred, X_test, Z_test, X_train, y_train, p_cut, a
            )
            # fit all scores
            for key in keys:
                fold_scores[key] = fold_scores[key] + [
                    combine_score(
                        acc,
                        acc_lr,
                        abs_op_dif,
                        abs_op_dif_lr,
                        ng,
                        alpha=float(key.split("_")[1]),
                        combiner=key.split("_")[0],
                    )
                ]

        fold_aves = dict(zip(keys, [np.mean(v) for v in fold_scores.values()]))

        # 3. Update best parameters if current combination is better
        for k, s in best_scores.items():
            if s < fold_aves[k]:
                best_scores[k] = fold_aves[k]
                best_params[k] = current_params

    return best_params, best_scores


# generate a results table given penalties and test and train sets
def model_evaluator(
    Idp_train: np.array,
    y_train: np.array,
    Idp_test: np.array,
    y_test: np.array,
    lg: list = [0, 0, 0],
    alpha: float = 0.5,
    p_cut: float = 0.5,
    kset: int = 0,
    misspec: int = 0,
    a: int = 3,
    ver: str = "weighted",
    runtime: datetime.timedelta = datetime.timedelta(seconds=5),
    n_iter: int = 40,
):
    """
    Gets performance stastistics for an estimator parametrized by penalty weight coefficients

    Args:
        Idp_train: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership. Training data.
        y_train: Target / outcome labels for training data.
        Idp_test: Same as Idp_train but for test data
        y_test: Target / outcome labels for test data.
        lg: List of penalty weight coefficients to use for the estimator.
        alpha: Fairness score weight.
        p_cut: Classification threshold.
        k_set: Index for the simulated population distribution.
        misspec: Indicator for whether the estimator is misspecified.
        a: Number of groups.
        ver: Name of the group synthesis mechanism.
        runtime: Runtime of the estimator fitting process.
        n_iter: Number of penalty weight candidate sets to sample.

    Returns:
        Dataframe of estimator characteristics and performance.
    """
    # generate predictions
    tester_custom = CustomRegression(l1=lg[0], l2=lg[1], l3=lg[2], alpha=alpha)
    tester_custom = tester_custom.fit(Idp_train, y_train)
    y_pred = tester_custom.predict(Idp_test, p_cut)[:, np.newaxis]
    Z_test = Idp_test[:, -(a + 1) :]

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
    # get unfairness metrics
    mean_unfair = np.sum(np.multiply(np.array(tpr_dif), np.array(ng))) / sum(ng)
    max_unfair = min(tpr_dif)
    # get score
    acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng = calc_score_parts(
        y_test,
        y_pred,
        Idp_test[:, : (Idp_test.shape[1] - a - 1)],
        Z_test,
        Idp_train[:, : (Idp_test.shape[1] - a - 1)],
        y_train,
        p_cut,
        a,
    )
    scoreval = combine_score(acc, acc_lr, abs_op_dif, abs_op_dif_lr, ng, alpha, ver)

    model_outputs = pd.DataFrame(
        {
            "l1": [lg[0]],
            "l2": [lg[1]],
            "l3": [lg[2]],
            "alpha": [alpha],
            "misspec": [misspec],
            "k_set": [kset],
            "accuracy": [acc],
            "mean_unfairness": [mean_unfair],
            "mean_unfairness_unwt": [np.mean(tpr_dif)],
            "max_unfairness": [max_unfair],
            "A_unfairness": [tpr_dif[0]],
            "B_unfairness": [tpr_dif[1]],
            "C_unfairness": [tpr_dif[2]],
            "Score": [scoreval],
            "group_synth": [ver],
            "n_iter": [n_iter],
            "runtime": [runtime],
        }
    )
    return model_outputs


def model_evaluator_gen(
    Idp_train: np.array,
    y_train: np.array,
    Idp_test: np.array,
    y_test: np.array,
    lg: list = [0, 0, 0],
    alpha: float = 0.5,
    p_cut: float = 0.5,
    a: int = 3,
    ver: str = "weighted",
):
    """
    Gets performance stastistics for an estimator parametrized by penalty weight coefficients

    Args:
        Idp_train: Array where rows are observations and first m columns are the m covariates / predictors X (with no intercept column), the next a columns are the indicators of group membership, and the final column is indicator for reference group membership. Training data.
        y_train: Target / outcome labels for training data.
        Idp_test: Same as Idp_train but for test data
        y_test: Target / outcome labels for test data.
        lg: List of penalty weight coefficients to use for the estimator.
        alpha: Fairness score weight.
        p_cut: Classification threshold.
        a: Number of groups.
        ver: Name of the group synthesis mechanism.

    Returns:
        Dataframe: Estimator characteristics and performance.
        CustomRegression: The fitted estimator.
    """
    # generate predictions
    tester_custom = CustomRegression(l1=lg[0], l2=lg[1], l3=lg[2], alpha=alpha)
    tester_custom = tester_custom.fit(Idp_train, y_train)
    y_pred = tester_custom.predict(Idp_test, p_cut)[:, np.newaxis]
    Z_test = Idp_test[:, -(a + 1) :]

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
    # get unfairness metrics
    mean_unfair = np.sum(np.multiply(np.array(tpr_dif), np.array(ng))) / sum(ng)
    max_unfair = min(tpr_dif)

    model_outputs = pd.DataFrame(
        {
            "l1": [lg[0]],
            "l2": [lg[1]],
            "l3": [lg[2]],
            "alpha": [alpha],
            "grp_synth": [ver],
            "accuracy": [acc],
            "sensi": [recall_score(y_test, y_pred)],
            "speci": [recall_score(np.logical_not(y_test), np.logical_not(y_pred))],
            "mean_unfairness": [mean_unfair],
            "mean_unfairness_unwt": [np.mean(tpr_dif)],
            "max_unfairness": [max_unfair],
            "A_unfairness": [tpr_dif[0]],
            "B_unfairness": [tpr_dif[1]],
            "C_unfairness": [tpr_dif[2]],
        }
    )
    return model_outputs, tester_custom


# make sim pop, perform grid search, and get results table
def run_sim_pipe(
    n: int = 1000000,
    k: list = [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
    kset: int = 0,
    p_cut: float = 0.5,
    ps: list = [0.2, 0.4, 0.6, 0.8, 0.95],
    misspec: int = 0,
    n_iter: int = 20,
):
    """
    Runs the simulation pipeline, including population generation, random search, and evaluation for many score function versions.

    Args:
        n: Simulated population size
        k: Parameters for the simulated population generation.
        k_set: Index for the simulated population distribution.
        p_cut: Classification threshold.
        ps: Additional parameters for the simulated population generation.
        misspec: Indicator for whether the estimator is misspecified.
        n_iter: Number of penalty weight candidate sets to sample.

    Returns:
        Dataframe of estimator characteristics and performance for the estimator resulting from each score function specification.
    """
    # make simulated population
    X, y, Z, Y_p = simpop(n, ps, k)
    if misspec == 1:
        X = X[:, 1:]
    refgrp = 1 - np.max(Z, axis=1)
    Z = np.hstack((Z, refgrp[:, np.newaxis]))
    rhs = np.hstack((X, Z))
    startime = datetime.datetime.now()
    # establish and fit the grid
    val_scale = loguniform(1e-3, 1e1)
    lams = {"l1": val_scale, "l2": val_scale, "l3": val_scale}
    best_params, _ = random_search_cv_custom(
        CustomRegression,
        rhs,
        y,
        param_distributions=lams,
        n_iter=n_iter,
        cv_folds=3,
        p_cut=p_cut,
    )
    runtime = startime - datetime.datetime.now()

    # fresh sim pull and evaluate
    X_test, y_test, Z_test, Y_p = simpop(n, ps, k)
    refgrp = 1 - np.max(Z_test, axis=1)
    Z_test = np.hstack((Z_test, refgrp[:, np.newaxis]))
    if misspec == 1:
        X_test = X_test[:, 1:]

    results_list = []
    for comps, lgs in best_params.items():
        comp_list = comps.split("_")
        results_list += [
            model_evaluator(
                np.hstack((X, Z)),
                y,
                np.hstack((X_test, Z_test)),
                y_test,
                [lgs["l1"], lgs["l2"], lgs["l3"]],
                float(comp_list[1]),
                p_cut,
                kset,
                misspec,
                3,
                comp_list[0],
                runtime,
                n_iter,
            )
        ]

    return pd.concat(results_list)


# for a particular population, get results row for essentially unpenalized version
def get0_spec(
    n: int = 1000000,
    k: list = [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
    kset: int = 0,
    alpha: float = 0.5,
    p_cut: float = 0.5,
    ps: list = [0.2, 0.4, 0.6, 0.8, 0.95],
    misspec: int = 0,
):
    """
    Evaluates simulation performance for an estimator with no penalties.

    Args:
        n: Simulated population size
        k: Parameters for the simulated population generation.
        k_set: Index for the simulated population distribution.
        alpha: Fairness score weight.
        p_cut: Classification threshold.
        ps: Additional parameters for the simulated population generation.
        misspec: Indicator for whether the estimator is misspecified.

    Returns:
        Dataframe of estimator characteristics and performance for the estimator resulting from each score function specification.
    """
    X, y, Z, Y_p = simpop(n, ps, k)
    refgrp = 1 - np.max(Z, axis=1)
    Z = np.hstack((Z, refgrp[:, np.newaxis]))
    if misspec == 1:
        X = X[:, 1:]
    # fresh sim pull and evaluate
    X_test, y_test, Z_test, Y_p = simpop(n, ps, k)
    refgrp = 1 - np.max(Z_test, axis=1)
    Z_test = np.hstack((Z_test, refgrp[:, np.newaxis]))
    if misspec == 1:
        X_test = X_test[:, 1:]
    results_table = model_evaluator(
        np.hstack((X, Z)),
        y,
        np.hstack((X_test, Z_test)),
        y_test,
        [0, 0, 0],
        alpha,
        p_cut,
        kset,
        misspec,
        3,
        "na",
        0,
    )

    return results_table


# run get0_spec iteratively to get unpenalized comp for groups
def get0s():
    """
    Evaluates simulation performance for an estimator with no penalties by calling helper function get0_spec() iteratively.

    Returns:
        Dataframe of estimator characteristics and performance for the unpenalized estimator.
    """
    res = []
    for misspec in [0, 1]:
        for kset in range(3):
            k = [
                [0.3, 0.3, 0.3, 0.02, 0.02, 0.02],
                [0.005, 0.3, 0.3, 0.002, 0.02, 0.02],
                [0.02, 0.02, 0.02, 0.002, 0.002, 0.002],
            ][kset]
            res.append(get0_spec(n=1000000, k=k, kset=kset, alpha=0, misspec=misspec))
    results_table = pd.concat(res)
    results_table.to_csv("tables/penalty_res_al0.csv", index=False)
    return results_table