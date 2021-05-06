import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier


def get_value_estimators(
    policy, contexts, actions, rewards, propensities, skip_nonlin=True
):
    """
    Args:
        policy (Policy): the policy we want to get a value estimate for
        contexts (np.array): contexts from bandit feedback
        actions (np.array): actions chosen for bandit feedback
        rewards (np.array): rewards received in bandit feedback
        propensities (np.array): the propensity for each action selected under the logging
            policy (which is not provided to this function)
    Returns:
        est (dict): keys are string describing the value estimator, values are the
        corresponding value estimates
    """

    est = {}
    weights = policy.get_action_propensities(contexts, actions) / propensities
    est["iw"] = np.sum(weights * rewards) / len(rewards)

    # direct method
    dm = np.zeros((len(contexts), policy.num_actions))
    dm_iw = np.zeros((len(contexts), policy.num_actions))
    dm_log = np.zeros((len(contexts), policy.num_actions))
    dm_log_iw = np.zeros((len(contexts), policy.num_actions))
    dm_rf = np.zeros((len(contexts), policy.num_actions))
    for a in range(policy.num_actions):
        # linear regression fits
        lr = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(
            contexts[actions == a], rewards[actions == a]
        )
        lr_iw = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(
            contexts[actions == a],
            rewards[actions == a],
            sample_weight=weights[actions == a],
        )

        # add predicted rewards for particular action for all contexts
        dm[:, a] = lr.predict(contexts)
        dm_iw[:, a] = lr_iw.predict(contexts)

        try:
            log_reg = LogisticRegression(max_iter=2500).fit(
                contexts[actions == a], rewards[actions == a]
            )
            dm_log[:, a] = log_reg.predict_proba(contexts)[:, 1]
            log_reg_iw= LogisticRegression(max_iter=2500).fit(
                contexts[actions == a],
                rewards[actions == a],
                sample_weight=weights[actions==a]
            )
            dm_log_iw[:, a] = log_reg_iw.predict_proba(contexts)[:, 1]
        except ValueError:
            pass

        if not skip_nonlin:
            # random forest fit
            rf = RandomForestClassifier(
                criterion="entropy", n_estimators=100, min_samples_leaf=5
            )
            rf = rf.fit(contexts[actions == a], rewards[actions == a])

            # add predicted rewards for particular action for all contexts
            dm_rf[:, a] = rf.predict(contexts)

    # get the policy's probability distribution over each action conditioned on the context
    props_new = policy.get_action_distribution(contexts)

    est["dm"] = np.mean((dm * props_new).sum(axis=1))
    est["dm_iw"] = np.mean((dm_iw * props_new).sum(axis=1))
    est["dm_log"] = np.mean((dm_log * props_new).sum(axis=1))
    est["dm_log_iw"] = np.mean((dm_log_iw * props_new).sum(axis=1))

    est["dr"] = np.mean(
        (dm * props_new).sum(axis=1)
        + weights * (rewards - dm[np.arange(dm.shape[0]), actions])
    )
    est["dr_log"] = np.mean(
        (dm_log * props_new).sum(axis=1)
        + weights * (rewards - dm_log[np.arange(dm_log.shape[0]), actions])
    )
    # est['dr_iw'] = np.mean((dm_iw*props_new).sum(axis=1) + weights*(rewards - dm_iw[np.arange(dm.shape[0]),actions]))

    if not skip_nonlin:
        est["dm_rf"] = np.mean((dm_rf * props_new).sum(axis=1))
        est["dr_rf"] = np.mean(
            (dm_rf * props_new).sum(axis=1)
            + weights * (rewards - dm_rf[np.arange(dm_rf.shape[0]), actions])
        )

    return est


def get_estimator_stats(estimates, true_parameter_value=None):
    """

     Args:
        estimates (pd.DataFrame): each row corresponds to collection of estimates for a
            sample and each column corresponds to an estimator
        true_parameter_value (float): the true parameter value that we will be comparing
            estimates to

    Returns:
        pd.Dataframe where each row represents data about a single estimator
    """
    est_stat = []
    for est in estimates.columns:
        pred_means = estimates[est]
        stat = {}
        stat["stat"] = est
        stat["mean"] = np.mean(pred_means)
        stat["SD"] = np.std(pred_means)
        stat["var"] = stat["SD"] ** 2
        stat["SE"] = np.std(pred_means) / np.sqrt(len(pred_means))
        if true_parameter_value:
            stat["bias"] = abs(stat["mean"] - true_parameter_value)
            stat["RMSE"] = np.sqrt(np.mean((pred_means - true_parameter_value) ** 2))
        est_stat.append(stat)

    return pd.DataFrame(est_stat)
