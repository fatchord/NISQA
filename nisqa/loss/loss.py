# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import numpy as np
from scipy.stats import pearsonr
import torch


# %% Loss
class BiasLoss(object):
    """
    Bias loss class.

    Calculates loss while considering database bias.
    """

    def __init__(self, db, anchor_db=None, mapping="first_order", min_r=0.7, loss_weight=0.0, do_print=True):

        self.db = db
        self.mapping = mapping
        self.min_r = min_r
        self.anchor_db = anchor_db
        self.loss_weight = loss_weight
        self.do_print = do_print

        self.b = np.zeros((len(db), 4))
        self.b[:, 1] = 1
        self.do_update = False

        self.apply_bias_loss = True
        if (self.min_r is None) or (self.mapping is None):
            self.apply_bias_loss = False

    def get_loss(self, yb, yb_hat, idx):

        if self.apply_bias_loss:
            b = torch.tensor(self.b, dtype=torch.float).to(yb_hat.device)
            b = b[idx, :]

            yb_hat_map = b[:, 0] + b[:, 1] * yb_hat[:, 0] + b[:, 2] * yb_hat[:, 0] ** 2 + b[:, 3] * yb_hat[:, 0] ** 3
            yb_hat_map = yb_hat_map.view(-1, 1)

            loss_bias = self._nan_mse(yb_hat_map, yb)
            loss_normal = self._nan_mse(yb_hat, yb)

            loss = loss_bias + self.loss_weight * loss_normal
        else:
            loss = self._nan_mse(yb_hat, yb)

        return loss

    def update_bias(self, y, y_hat):

        if self.apply_bias_loss:
            y_hat = y_hat.reshape(-1)
            y = y.reshape(-1)

            if not self.do_update:
                r = pearsonr(y[~np.isnan(y)], y_hat[~np.isnan(y)])[0]

                if self.do_print:
                    print("--> bias update: min_r {:0.2f}, r_p {:0.2f}".format(r, self.min_r))
                if r > self.min_r:
                    self.do_update = True

            if self.do_update:
                if self.do_print:
                    print("--> bias updated")
                for db_name in self.db.unique():

                    db_idx = (self.db == db_name).to_numpy().nonzero()
                    y_hat_db = y_hat[db_idx]
                    y_db = y[db_idx]

                    if not np.isnan(y_db).any():
                        if self.mapping == "first_order":
                            b_db = self._calc_bias_first_order(y_hat_db, y_db)
                        else:
                            raise NotImplementedError
                        if not db_name == self.anchor_db:
                            self.b[db_idx, :len(b_db)] = b_db

    def _calc_bias_first_order(self, y_hat, y):
        A = np.vstack([np.ones(len(y_hat)), y_hat]).T
        btmp = np.linalg.lstsq(A, y, rcond=None)[0]
        b = np.zeros((4))
        b[0:2] = btmp
        return b

    def _nan_mse(self, y, y_hat):
        err = (y - y_hat).view(-1)
        idx_not_nan = ~torch.isnan(err)
        nan_err = err[idx_not_nan]
        return torch.mean(nan_err ** 2)
