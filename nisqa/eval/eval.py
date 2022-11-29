# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize
import torch
from torch.utils.data import DataLoader
import pandas as pd
pd.options.mode.chained_assignment = None


# %% Evaluation
def predict_mos(model, ds, bs, dev, num_workers=0):
    """
    predict_mos: predicts MOS of the given dataset with given model. Used for
    NISQA and NISQA_DE model.
    """
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [[model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()]
                      for xb, yb, (idx, n_wins) in dl]
    yy = np.concatenate(y_hat_list, axis=1)
    y_hat = yy[0, :, 0].reshape(-1, 1)
    y = yy[1, :, 0].reshape(-1, 1)
    ds.df["mos_pred"] = y_hat.astype(dtype=float)
    return y_hat, y


def predict_dim(model, ds, bs, dev, num_workers=0):
    """
    predict_dim: predicts MOS and dimensions of the given dataset with given
    model. Used for NISQA_DIM model.
    """
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [[model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, (idx, n_wins) in
                      dl]
    yy = np.concatenate(y_hat_list, axis=1)

    y_hat = yy[0, :, :]
    y = yy[1, :, :]

    ds.df["mos_pred"] = y_hat[:, 0].reshape(-1, 1)
    ds.df["noi_pred"] = y_hat[:, 1].reshape(-1, 1)
    ds.df["dis_pred"] = y_hat[:, 2].reshape(-1, 1)
    ds.df["col_pred"] = y_hat[:, 3].reshape(-1, 1)
    ds.df["loud_pred"] = y_hat[:, 4].reshape(-1, 1)

    return y_hat, y


def is_const(x):
    if np.linalg.norm(x - np.mean(x)) < 1e-13 * np.abs(np.mean(x)):
        return True
    elif np.all(x == x[0]):
        return True
    else:
        return False


def calc_eval_metrics(y, y_hat, y_hat_map=None, d=None, ci=None):
    """
    Calculate RMSE, mapped RMSE, mapped RMSE* and Pearson"s correlation.
    See ITU-T P.1401 for details on RMSE*.
    """
    r = {
        "r_p": np.nan,
        "rmse": np.nan,
        "rmse_map": np.nan,
        "rmse_star_map": np.nan,
    }
    if is_const(y_hat) or any(np.isnan(y)):
        r["r_p"] = np.nan
    else:
        r["r_p"] = pearsonr(y, y_hat)[0]
    r["rmse"] = calc_rmse(y, y_hat)
    if y_hat_map is not None:
        r["rmse_map"] = calc_rmse(y, y_hat_map, d=d)
        if ci is not None:
            r["rmse_star_map"] = calc_rmse_star(y, y_hat_map, ci, d)[0]
    return r


def calc_rmse(y_true, y_pred, d=0):
    if d == 0:
        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    else:
        N = y_true.shape[0]
        if (N - d) < 1:
            rmse = np.nan
        else:
            rmse = np.sqrt(1 / (N - d) * np.sum(np.square(y_true - y_pred)))  # Eq (7-29) P.1401
    return rmse


def calc_rmse_star(mos_sub, mos_obj, ci, d):
    N = mos_sub.shape[0]
    error = mos_sub - mos_obj

    if np.isnan(ci).any():
        p_error = np.nan
        rmse_star = np.nan
    else:
        p_error = (abs(error) - ci).clip(min=0)  # Eq (7-27) P.1401
        if (N - d) < 1:
            rmse_star = np.nan
        else:
            rmse_star = np.sqrt(1 / (N - d) * sum(p_error ** 2))  # Eq (7-29) P.1401

    return rmse_star, p_error, error


def calc_mapped(x, b):
    N = x.shape[0]
    order = b.shape[0] - 1
    A = np.zeros([N, order + 1])
    for i in range(order + 1):
        A[:, i] = x ** (i)
    return A @ b


def fit_first_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b


def fit_second_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat ** 2]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b


def fit_third_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat ** 2, y_con_hat ** 3]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]

    p = np.poly1d(np.flipud(b))
    p2 = np.polyder(p)
    rr = np.roots(p2)
    r = rr[np.imag(rr) == 0]
    monotonic = all(np.logical_or(r > max(y_con_hat), r < min(y_con_hat)))
    if monotonic == False:
        print("Not monotonic!!!")
    return b


def fit_monotonic_third_order(
        dfile_db,
        dcon_db=None,
        pred=None,
        target_mos=None,
        target_ci=None,
        mapping=None):
    """
    Fits third-order function with the constrained to be monotonically.
    increasing. This function may not return an optimal fitting.
    """
    y = dfile_db[target_mos].to_numpy()

    y_hat = dfile_db[pred].to_numpy()

    if dcon_db is None:
        if target_ci in dfile_db:
            ci = dfile_db[target_ci].to_numpy()
        else:
            ci = 0
    else:
        y_con = dcon_db[target_mos].to_numpy()

        if target_ci in dcon_db:
            ci = dcon_db[target_ci].to_numpy()
        else:
            ci = 0

    x = y_hat
    y_hat_min = min(y_hat) - 0.01
    y_hat_max = max(y_hat) + 0.01

    def polynomial(p, x):
        return p[0] + p[1] * x + p[2] * x ** 2 + p[3] * x ** 3

    def constraint_2nd_der(p):
        return 2 * p[2] + 6 * p[3] * x

    def constraint_1st_der(p):
        x = np.arange(y_hat_min, y_hat_max, 0.1)
        return p[1] + 2 * p[2] * x + 3 * p[3] * x ** 2

    def objective_con(p):
        x_map = polynomial(p, x)
        dfile_db["x_map"] = x_map
        x_map_con = dfile_db.groupby("con").mean().x_map.to_numpy()
        err = x_map_con - y_con
        if mapping == "pError":
            p_err = (abs(err) - ci).clip(min=0)
            return (p_err ** 2).sum()
        elif mapping == "error":
            return (err ** 2).sum()
        else:
            raise NotImplementedError

    def objective_file(p):
        x_map = polynomial(p, x)
        err = x_map - y
        if mapping == "pError":
            p_err = (abs(err) - ci).clip(min=0)
            return (p_err ** 2).sum()
        elif mapping == "error":
            return (err ** 2).sum()
        else:
            raise NotImplementedError

    cons = dict(type="ineq", fun=constraint_1st_der)

    if dcon_db is None:
        res = minimize(
            objective_file,
            x0=np.array([0., 1., 0., 0.]),
            method="SLSQP",
            constraints=cons,
        )
    else:
        res = minimize(
            objective_con,
            x0=np.array([0., 1., 0., 0.]),
            method="SLSQP",
            constraints=cons,
        )
    b = res.x
    return b


def calc_mapping(
        dfile_db,
        mapping=None,
        dcon_db=None,
        target_mos=None,
        target_ci=None,
        pred=None,
):
    """
    Computes mapping between subjective and predicted MOS.
    """
    if dcon_db is not None:
        y = dcon_db[target_mos].to_numpy()
        y_hat = dfile_db.groupby("con").mean().get(pred).to_numpy()
    else:
        y = dfile_db[target_mos].to_numpy()
        y_hat = dfile_db[pred].to_numpy()

    if mapping == None:
        b = np.array([0, 1, 0, 0])
        d_map = 0
    elif mapping == "first_order":
        b = fit_first_order(y, y_hat)
        d_map = 1
    elif mapping == "second_order":
        b = fit_second_order(y, y_hat)
        d_map = 3
    elif mapping == "third_order_not_monotonic":
        b = fit_third_order(y, y_hat)
        d_map = 4
    elif mapping == "third_order":
        b = fit_monotonic_third_order(
            dfile_db,
            dcon_db=dcon_db,
            pred=pred,
            target_mos=target_mos,
            target_ci=target_ci,
            mapping="error",
        )
        d_map = 4
    else:
        raise NotImplementedError

    return b, d_map


def eval_results(
        df,
        dcon=None,
        target_mos="mos",
        target_ci="mos_ci",
        pred="mos_pred",
        mapping=None,
        do_print=False,
        do_plot=False
):
    """
    Evaluates a trained model on given dataset.
    """
    # Loop through databases
    db_results_df = []
    df["y_hat_map"] = np.nan

    for db_name in df.db.astype("category").cat.categories:

        df_db = df.loc[df.db == db_name]
        if dcon is not None:
            dcon_db = dcon.loc[dcon.db == db_name]
        else:
            dcon_db = None

        # per file -----------------------------------------------------------
        y = df_db[target_mos].to_numpy()
        if np.isnan(y).any():
            r = {"r_p": np.nan, "r_s": np.nan, "rmse": np.nan, "r_p_map": np.nan,
                 "r_s_map": np.nan, "rmse_map": np.nan}
        else:
            y_hat = df_db[pred].to_numpy()

            b, d = calc_mapping(
                df_db,
                mapping=mapping,
                target_mos=target_mos,
                target_ci=target_ci,
                pred=pred
            )
            y_hat_map = calc_mapped(y_hat, b)

            r = calc_eval_metrics(y, y_hat, y_hat_map=y_hat_map, d=d)
            r.pop("rmse_star_map")
        r = {f"{k}_file": v for k, v in r.items()}

        # per con ------------------------------------------------------------
        r_con = {"r_p": np.nan, "r_s": np.nan, "rmse": np.nan, "r_p_map": np.nan,
                 "r_s_map": np.nan, "rmse_map": np.nan, "rmse_star_map": np.nan}

        if (dcon_db is not None) and ("con" in df_db):

            y_con = dcon_db[target_mos].to_numpy()
            y_con_hat = df_db.groupby("con").mean().get(pred).to_numpy()

            if not np.isnan(y_con).any():

                if target_ci in dcon_db:
                    ci_con = dcon_db[target_ci].to_numpy()
                else:
                    ci_con = None

                b_con, d = calc_mapping(
                    df_db,
                    dcon_db=dcon_db,
                    mapping=mapping,
                    target_mos=target_mos,
                    target_ci=target_ci,
                    pred=pred
                )

                df_db["y_hat_map"] = calc_mapped(y_hat, b_con)
                df["y_hat_map"].loc[df.db == db_name] = df_db["y_hat_map"]

                y_con_hat_map = df_db.groupby("con").mean().get("y_hat_map").to_numpy()
                r_con = calc_eval_metrics(y_con, y_con_hat, y_hat_map=y_con_hat_map, d=d, ci=ci_con)

        r_con = {f"{k}_con": v for k, v in r_con.items()}
        r = {**r, **r_con}

        # ---------------------------------------------------------------------
        db_results_df.append({"db": db_name, **r})
        # Plot  ------------------------------------------------------------------
        if do_plot and (not np.isnan(y).any()):
            xx = np.arange(0, 6, 0.01)
            yy = calc_mapped(xx, b)

            plt.figure(figsize=(3.0, 3.0), dpi=300)
            plt.clf()
            plt.plot(y_hat, y, "o", label="Original data", markersize=2)
            plt.plot([0, 5], [0, 5], "gray")
            plt.plot(xx, yy, "r", label="Fitted line")
            plt.axis([1, 5, 1, 5])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.grid(True)
            plt.xticks(np.arange(1, 6))
            plt.yticks(np.arange(1, 6))
            plt.title(db_name + " per file")
            plt.ylabel("Subjective " + target_mos.upper())
            plt.xlabel("Predicted " + target_mos.upper())
            # plt.savefig("corr_diagram_fr_" + db_name + ".pdf", dpi=300, bbox_inches="tight")
            plt.show()

            if (dcon_db is not None) and ("con" in df_db):
                xx = np.arange(0, 6, 0.01)
                yy = calc_mapped(xx, b_con)

                plt.figure(figsize=(3.0, 3.0), dpi=300)
                plt.clf()
                plt.plot(y_con_hat, y_con, "o", label="Original data", markersize=3)
                plt.plot([0, 5], [0, 5], "gray")
                plt.plot(xx, yy, "r", label="Fitted line")
                plt.axis([1, 5, 1, 5])
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid(True)
                plt.xticks(np.arange(1, 6))
                plt.yticks(np.arange(1, 6))
                plt.title(db_name + " per con")
                plt.ylabel("Sub " + target_mos.upper())
                plt.xlabel("Pred " + target_mos.upper())
                # plt.savefig(db_name + ".pdf", dpi=300, bbox_inches="tight")
                plt.show()

        # Print ------------------------------------------------------------------
        if do_print and (not np.isnan(y).any()):
            if (dcon_db is not None) and ("con" in df_db):
                print(
                    "%-30s r_p_file: %0.2f, rmse_map_file: %0.2f, r_p_con: %0.2f, rmse_map_con: %0.2f, rmse_star_map_con: %0.2f"
                    % (db_name + ":", r["r_p_file"], r["rmse_map_file"], r["r_p_con"], r["rmse_map_con"],
                       r["rmse_star_map_con"]))
            else:
                print("%-30s r_p_file: %0.2f, rmse_map_file: %0.2f"
                      % (db_name + ":", r["r_p_file"], r["rmse_map_file"]))

    # Save individual database results in DataFrame
    db_results_df = pd.DataFrame(db_results_df)

    r_average = {}
    r_average["r_p_mean_file"] = db_results_df.r_p_file.mean()
    r_average["rmse_mean_file"] = db_results_df.rmse_file.mean()
    r_average["rmse_map_mean_file"] = db_results_df.rmse_map_file.mean()

    if dcon_db is not None:
        r_average["r_p_mean_con"] = db_results_df.r_p_con.mean()
        r_average["rmse_mean_con"] = db_results_df.rmse_con.mean()
        r_average["rmse_map_mean_con"] = db_results_df.rmse_map_con.mean()
        r_average["rmse_star_map_mean_con"] = db_results_df.rmse_star_map_con.mean()
    else:
        r_average["r_p_mean_con"] = np.nan
        r_average["rmse_mean_con"] = np.nan
        r_average["rmse_map_mean_con"] = np.nan
        r_average["rmse_star_map_mean_con"] = np.nan

    # Get overall per file results
    y = df[target_mos].to_numpy()
    y_hat = df[pred].to_numpy()

    r_total_file = calc_eval_metrics(y, y_hat)
    r_total_file = {"r_p_all": r_total_file["r_p"], "rmse_all": r_total_file["rmse"]}

    overall_results = {
        **r_total_file,
        **r_average
    }

    return db_results_df, overall_results
