# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""


# %% Early stopping
class earlyStopper(object):
    """
    Early stopping class.

    Training is stopped if neither RMSE or Pearson"s correlation
    is improving after "patience" epochs.
    """

    def __init__(self, patience):
        self.best_rmse = 1e10
        self.best_r_p = -1e10
        self.cnt = -1
        self.patience = patience
        self.best = False

    def step(self, r):
        self.best = False
        if r["r_p_mean_file"] > self.best_r_p:
            self.best_r_p = r["r_p_mean_file"]
            self.cnt = -1
        if r["rmse_map_mean_file"] < self.best_rmse:
            self.best_rmse = r["rmse_map_mean_file"]
            self.cnt = -1
            self.best = True
        self.cnt += 1

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early


class earlyStopper_dim(object):
    """
    Early stopping class for dimension model.

    Training is stopped if neither RMSE or Pearson"s correlation
    is improving after "patience" epochs.
    """

    def __init__(self, patience):

        self.best_rmse = 1e10
        self.best_rmse_noi = 1e10
        self.best_rmse_col = 1e10
        self.best_rmse_dis = 1e10
        self.best_rmse_loud = 1e10

        self.best_r_p = -1e10
        self.best_r_p_noi = -1e10
        self.best_r_p_col = -1e10
        self.best_r_p_dis = -1e10
        self.best_r_p_loud = -1e10

        self.cnt = -1
        self.patience = patience
        self.best = False

    def step(self, r):

        self.best = False

        if r["r_p_mean_file"] > self.best_r_p:
            self.best_r_p = r["r_p_mean_file"]
            self.cnt = -1
        if r["r_p_mean_file_noi"] > self.best_r_p_noi:
            self.best_r_p_noi = r["r_p_mean_file_noi"]
            self.cnt = -1
        if r["r_p_mean_file_col"] > self.best_r_p_col:
            self.best_r_p_col = r["r_p_mean_file_col"]
            self.cnt = -1
        if r["r_p_mean_file_dis"] > self.best_r_p_dis:
            self.best_r_p_dis = r["r_p_mean_file_dis"]
            self.cnt = -1
        if r["r_p_mean_file_loud"] > self.best_r_p_loud:
            self.best_r_p_loud = r["r_p_mean_file_loud"]
            self.cnt = -1

        if r["rmse_map_mean_file"] < self.best_rmse:
            self.best_rmse = r["rmse_map_mean_file"]
            self.cnt = -1
            self.best = True
        if r["rmse_map_mean_file_noi"] < self.best_rmse_noi:
            self.best_rmse_noi = r["rmse_map_mean_file_noi"]
            self.cnt = -1
        if r["rmse_map_mean_file_col"] < self.best_rmse_col:
            self.best_rmse_col = r["rmse_map_mean_file_col"]
            self.cnt = -1
        if r["rmse_map_mean_file_dis"] < self.best_rmse_dis:
            self.best_rmse_dis = r["rmse_map_mean_file_dis"]
            self.cnt = -1
        if r["rmse_map_mean_file_loud"] < self.best_rmse_loud:
            self.best_rmse_loud = r["rmse_map_mean_file_loud"]
            self.cnt = -1

        self.cnt += 1

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early


def get_lr(optimizer):
    """
    Get current learning rate from Pytorch optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
