import numpy as np
from numpy.random import default_rng


def generate_bandit_feedback(
    contexts, full_rewards, policy, new_n=None, rng=default_rng(1)
):
    """
    Args:
        contexts (np.array): contexts, rows correspond to entries of rewards
        full_rewards (np.array): 2-dim numpy array with the same number of rows as X and
            number of columns corresponding to the number actions each row gives the
            reward that would be received for each action for the context in the
            corresponding row of X.

    Returns:
        new_contexts (np.array): new_n rows and same number of columns as in contexts
        actions (np.array): vector with new_n entries giving actions selected by the
            provided policy for the contexts in new_contexts
        observed_rewards (np.array): vector with new_n entries giving actions selected
            by the provided policy for the contexts in new_contexts
    """

    if new_n is None:
        new_n = contexts.shape[0]
    n, k = full_rewards.shape
    num_repeats = np.ceil(new_n / n).astype(int)
    new_contexts = np.tile(contexts, [num_repeats, 1])
    new_contexts = new_contexts[0:new_n]
    new_rewards = np.tile(full_rewards, [num_repeats, 1])
    new_rewards = new_rewards[0:new_n]
    actions, propensities = policy.select_actions(X=new_contexts, rng=rng)
    observed_rewards = new_rewards[np.arange(new_n), actions]
    return new_contexts, actions, observed_rewards, propensities
