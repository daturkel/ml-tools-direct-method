import math
import random
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd

datasets = [
    "ecoli",
    "glass",
    "letter-recognition",
    "optdigits",
    "pendigits",
    "yeast",
]

header_mapper = {
    "ecoli": [
        "Sequence_Name",
        "mcg",
        "gvh",
        "lip",
        "chg",
        "aac",
        "alm1",
        "alm2",
        "class",
    ],
    "glass": ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"],
    "letter-recognition": [
        "class",
        "x-box",
        "y-box",
        "width",
        "high",
        "onpix",
        "x-bar",
        "y-bar",
        "x2bar",
        "y2bar",
        "xybar",
        "x2ybr",
        "xy2br",
        "x-edge",
        "xegvy",
        "y-edge",
        "yegvx",
    ],
    "optdigits": ["x" + str(i) for i in range(64)] + ["class"],
    "pendigits": ["x" + str(i) for i in range(16)] + ["class"],
    "yeast": [
        "Sequence_Name",
        "mcg",
        "gvh",
        "alm",
        "mit",
        "erl",
        "pox",
        "vac",
        "nuc",
        "class",
    ],
}

suffices = {
    "ecoli": [".data"],
    "glass": [".data"],
    "letter-recognition": [".data"],
    "optdigits": [".tes", ".tra"],
    "pendigits": [".tes", ".tra"],
    "satimage": ["sat.trn", "sat.tst"],
    "vehicle": [
        "xaa.dat",
        "xab.dat",
        "xac.dat",
        "xad.dat",
        "xae.dat",
        "xaf.dat",
        "xag.dat",
        "xah.dat",
        "xai.dat",
    ],
    "yeast": [".data"],
}


def data_reader(dataset_use):
    assert (
        (dataset_use in datasets)
        and (dataset_use in header_mapper)
        and (dataset_use in suffices)
    )

    file_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/{df}/{df}"

    content = []
    for suf in suffices[dataset_use]:
        data = file_name.format(df=dataset_use) + suf

        req = urlopen(data)
        for line in req.readlines():
            content.append(
                [
                    item
                    for item in re.sub("\s+", ",", line.decode("utf-8").rstrip()).split(
                        ","
                    )
                    if item
                ]
            )

    return content


def get_bandit(dataset_use):
    """
    This function loads in a multiclass classification dataset and converts to a fully observed bandit dataset.

    """
    # set location of file
    content = data_reader(dataset_use)

    df = pd.DataFrame(content, columns=header_mapper[dataset_use])

    # remove non-numerical columns from contexts
    col_list = ["class"]
    cat_ls = ["Sequence_Name", "ID"]
    for cat in cat_ls:
        if cat in df.columns:
            col_list.append(cat)
    X = df.drop(columns=col_list)

    # convert categorical data to numerical categories
    y = df["class"]
    y = y.astype("category").cat.codes

    # get full rewards
    n = len(y)
    k = max(y) + 1
    full_rewards = np.zeros([n, k])
    full_rewards[np.arange(0, n), y] = 1
    contexts = X
    best_actions = y

    return contexts, full_rewards, best_actions


def split_data(contexts, full_rewards, best_actions):
    # ensure all actions are in training set by selecting one instance of each action first
    ind_ls = []
    for cat in best_actions.unique():
        filter_ = contexts.index[best_actions == cat]
        ind = np.random.choice(filter_)
        ind_ls.append(ind)

    # get remaining indices from contexts list
    rem_inds = list(set(contexts.index) - set(ind_ls))

    # roughly equal split of dataset
    mid = math.ceil(len(contexts) / 2)

    # add more samples to training to sum up to roughly equal split of dataset
    new_train_ind = random.sample(rem_inds, mid - len(ind_ls))
    train_ind = ind_ls + new_train_ind

    # get indices for test set
    test_ind = np.setdiff1d(np.arange(len(contexts)), train_ind, assume_unique=True)

    # assign datasets based on indices
    X_train = contexts.iloc[train_ind].to_numpy().astype(float)
    y_train = best_actions.iloc[train_ind].to_numpy().astype(int)
    X_test = contexts.iloc[test_ind].to_numpy().astype(float)
    y_test = best_actions.iloc[test_ind].to_numpy().astype(int)
    full_rewards_test = full_rewards[test_ind]

    return X_train, y_train, X_test, y_test, full_rewards_test
