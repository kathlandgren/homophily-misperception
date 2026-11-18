"""perception.py — perceived supporter fraction and Bayesian rescaling for the sweep cores.

Vectorized counterparts of assess_opinion() and rescale_bayes() in opinion_functions.py,
shared by sbm_simulation.py, pa_simulation.py, matched_simulation.py and swap_simulation.py.
"""

import numpy as np


def perceived_fraction(supp_nbrs, degree, is_supp, narcissistic=True):
    """
    Per-node perceived supporter fraction.

    Parameters:
    supp_nbrs (np.ndarray): Number of supporter neighbors of each node.
    degree (np.ndarray): Degree of each node.
    is_supp (np.ndarray of bool): True where the node is a supporter.
    narcissistic (bool): If True (the paper's default, Eq. 1), each node counts its own
        opinion alongside its neighbors': (supp_nbrs + own) / (degree + 1).
        If False, only neighbors count: supp_nbrs / degree; an isolated node
        perceives its own opinion (as in opinion_functions.assess_opinion).

    Returns:
    np.ndarray: Perceived supporter fraction of each node.
    """
    if narcissistic:
        return (supp_nbrs + is_supp.astype(float)) / (degree + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(degree > 0, supp_nbrs / degree, is_supp.astype(float))


def rescale_bayes(y, delta, gamma):
    """Ψ(y; δ, γ) = δ^(1-γ) · y^γ / (δ^(1-γ) · y^γ + (1-y)^γ)"""
    num = delta ** (1 - gamma) * y ** gamma
    return num / (num + (1 - y) ** gamma)
