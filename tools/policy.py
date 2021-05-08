import abc

import numpy as np
from numpy.random import default_rng
import pandas as pd


class Policy:
    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    @abc.abstractmethod
    def get_action_distribution(self, X):
        """
        This method is intended to be overridden by each implementation of Policy.

        Args:
            X (pd.DataFrame): contexts

        Returns:
            2-dim numpy array with the same number of rows as X and self.num_actions columns.
            Each rows gives the policy's probability distribution over actions conditioned
            on the context in the corresponding row of X
        """
        raise NotImplementedError("Must override method")

    def get_action_propensities(self, X, actions):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of actions
            actions (np.array): actions taken, represented by integers, corresponding to
            rows of X

        Returns:
            1-dim numpy array of probabilities (same size as actions) for taking each
            action in its corresponding context
        """
        probs = self.get_action_distribution(X)

        return probs[np.arange(probs.shape[0]), actions]

    def select_actions(self, X, rng=default_rng(1)):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of actions and
            propensities returned

        Returns:
            actions (np.array): 1-dim numpy array of length equal to the number of rows of X.
                Each entry is an integer indicating the action selected for the corresponding
                context in X.
                The action is selected randomly according to the policy, conditional on the
                context specified in the appropriate row of X.
            propensities (np.array): 1-dim numpy array of length equal to the number of
                rows of X; gives the propensity for each action selected in actions

        """
        action_prob = self.get_action_distribution(X)
        actions = np.array(
            [
                np.random.choice(np.arange(self.num_actions), p=action_prob[i])
                for i in range(len(action_prob))
            ]
        )
        propensities = self.get_action_propensities(X, actions)

        return actions, propensities

    def get_value_estimate(self, X, full_rewards):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of full_rewards
            full_rewards (np.array): 2-dim numpy array with the same number of rows as X
                and self.num_actions columns; each row gives the rewards that would be
                received for each action for the context in the corresponding row of X.
                This would only be known in a full-feedback bandit, or estimated in a
                direct method

        Returns:
            scalar value giving the expected average reward received for playing the
            policy for contexts X and the given full_rewards

        """
        rewards = full_rewards * self.get_action_distribution(X)
        return rewards.sum() / len(rewards)


class ModelPolicy(Policy):
    """
    ModelPolicy takes as input a user-defined model to create a policy with action
    distribution equal to the predicted probabilities of the model.
    """

    def __init__(self, model, num_actions=2):
        self.num_actions = num_actions
        self.model = model

    def get_action_distribution(self, X):
        """
        This method outputs the probability distribution of the policy, given the context.
        """
        return self.model.predict_proba(X)


class UniformPolicy(Policy):
    """
    UniformPolicy randomly selects an action, using a uniform distribution.
    """

    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def get_action_distribution(self, X):
        # define pdf as uniform distr
        return np.array([[1.0 / self.num_actions] * self.num_actions] * X.shape[0])


class NonuniformPolicy(Policy):
    """
    NonuniformPolicy randomly selects an action from a static nonuniform distribution.
    """

    def __init__(self, num_actions=2, rng=np.random.default_rng()):
        self.num_actions = num_actions
        weights = rng.integers(5, 15, size=self.num_actions)
        self.weights = weights / weights.sum()

    def get_action_distribution(self, X):
        return np.array([self.weights] * X.shape[0])


class CorrelatedPolicy(Policy):
    """
    Policy which randomly selects an action with weights that are a function of the covariates.
    """

    def __init__(self, num_actions=2, num_features=10, rng=np.random.default_rng):
        self.num_actions = num_actions
        self.num_features = num_features
        self.weights = rng.integers(5, 15, size=(self.num_features, self.num_actions))

    def get_action_distribution(self, X):
        tmp = X @ self.weights
        return tmp / tmp.sum(axis=1)[:, np.newaxis]
