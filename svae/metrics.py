import torch
import numpy as np

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression


def get_linear_score(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.score(x, y)


def linear_regression_metric(z, z_hat, num_samples=int(1e5), indices=None):

    score = get_linear_score(z_hat, z)
    # masking z_hat
    # TODO: this does not take into account case where z_block_size > 1
    if indices is not None:
        z_hat_m = z_hat[:, indices[-z.shape[0] :]]
        score_m = get_linear_score(z_hat_m, z)
    else:
        score_m = 0

    return score, score_m


def mean_corr_coef_np(x, y, method="pearson"):
    """
    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == "pearson":
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == "spearman":
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError("not a valid method: {}".format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)]
    return score, score.mean()


def mean_corr_coef(x, y, method="pearson"):
    if type(x) != type(y):
        raise ValueError(
            "inputs are of different types: ({}, {})".format(type(x), type(y))
        )
    if isinstance(x, np.ndarray):
        return mean_corr_coef_np(x, y, method)
    elif isinstance(x, torch.Tensor):
        return mean_corr_coef_pt(x, y, method)
    else:
        raise ValueError("not a supported input type: {}".format(type(x)))
