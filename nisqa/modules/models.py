# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import copy
import torch
import torch.nn as nn

from .pooling import Pooling
from .framewise import Framewise
from .time_dependency import TimeDependency
from .alignment import Alignment
from .fusion import Fusion


# %% Models
class NISQA(nn.Module):
    """
    NISQA: The main speech quality model without speech quality dimension
    estimation (MOS only). The module loads the submodules for framewise
    modelling (e.g. CNN), time-dependency modelling (e.g. Self-Attention
    or LSTM), and pooling (e.g. max-pooling or attention-pooling)
    """

    def __init__(self,
                 ms_seg_length=15,
                 ms_n_mels=48,

                 cnn_model="adapt",
                 cnn_c_out_1=16,
                 cnn_c_out_2=32,
                 cnn_c_out_3=64,
                 cnn_kernel_size=3,
                 cnn_dropout=0.2,
                 cnn_pool_1=[24, 7],
                 cnn_pool_2=[12, 5],
                 cnn_pool_3=[6, 3],
                 cnn_fc_out_h=None,

                 td="self_att",
                 td_sa_d_model=64,
                 td_sa_nhead=1,
                 td_sa_pos_enc=None,
                 td_sa_num_layers=2,
                 td_sa_h=64,
                 td_sa_dropout=0.1,
                 td_lstm_h=128,
                 td_lstm_num_layers=1,
                 td_lstm_dropout=0,
                 td_lstm_bidirectional=True,

                 td_2="skip",
                 td_2_sa_d_model=None,
                 td_2_sa_nhead=None,
                 td_2_sa_pos_enc=None,
                 td_2_sa_num_layers=None,
                 td_2_sa_h=None,
                 td_2_sa_dropout=None,
                 td_2_lstm_h=None,
                 td_2_lstm_num_layers=None,
                 td_2_lstm_dropout=None,
                 td_2_lstm_bidirectional=None,

                 pool="att",
                 pool_att_h=128,
                 pool_att_dropout=0.1,

                 ):
        super().__init__()

        self.name = "NISQA"

        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
        )

        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
        )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
        )

        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
        )

    def forward(self, x, n_wins):
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        x = self.pool(x, n_wins)
        return x


class NISQA_DIM(nn.Module):
    """
    NISQA_DIM: The main speech quality model with speech quality dimension
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), and pooling
    (e.g. max-pooling or attention-pooling)
    """

    def __init__(self,
                 ms_seg_length=15,
                 ms_n_mels=48,

                 cnn_model="adapt",
                 cnn_c_out_1=16,
                 cnn_c_out_2=32,
                 cnn_c_out_3=64,
                 cnn_kernel_size=3,
                 cnn_dropout=0.2,
                 cnn_pool_1=[24, 7],
                 cnn_pool_2=[12, 5],
                 cnn_pool_3=[6, 3],
                 cnn_fc_out_h=None,

                 td="self_att",
                 td_sa_d_model=64,
                 td_sa_nhead=1,
                 td_sa_pos_enc=None,
                 td_sa_num_layers=2,
                 td_sa_h=64,
                 td_sa_dropout=0.1,
                 td_lstm_h=128,
                 td_lstm_num_layers=1,
                 td_lstm_dropout=0,
                 td_lstm_bidirectional=True,

                 td_2="skip",
                 td_2_sa_d_model=None,
                 td_2_sa_nhead=None,
                 td_2_sa_pos_enc=None,
                 td_2_sa_num_layers=None,
                 td_2_sa_h=None,
                 td_2_sa_dropout=None,
                 td_2_lstm_h=None,
                 td_2_lstm_num_layers=None,
                 td_2_lstm_dropout=None,
                 td_2_lstm_bidirectional=None,

                 pool="att",
                 pool_att_h=128,
                 pool_att_dropout=0.1,

                 ):
        super().__init__()

        self.name = "NISQA_DIM"

        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
        )

        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
        )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
        )

        pool = Pooling(
            self.time_dependency.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
        )

        self.pool_layers = self._get_clones(pool, 5)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x, n_wins):
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out


class NISQA_DE(nn.Module):
    """
    NISQA: The main speech quality model for double-ended prediction.
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), time-alignment,
    feature fusion and pooling (e.g. max-pooling or attention-pooling)
    """

    def __init__(self,
                 ms_seg_length=15,
                 ms_n_mels=48,

                 cnn_model="adapt",
                 cnn_c_out_1=16,
                 cnn_c_out_2=32,
                 cnn_c_out_3=64,
                 cnn_kernel_size=3,
                 cnn_dropout=0.2,
                 cnn_pool_1=[24, 7],
                 cnn_pool_2=[12, 5],
                 cnn_pool_3=[6, 3],
                 cnn_fc_out_h=None,

                 td="self_att",
                 td_sa_d_model=64,
                 td_sa_nhead=1,
                 td_sa_pos_enc=None,
                 td_sa_num_layers=2,
                 td_sa_h=64,
                 td_sa_dropout=0.1,
                 td_lstm_h=128,
                 td_lstm_num_layers=1,
                 td_lstm_dropout=0,
                 td_lstm_bidirectional=True,

                 td_2="skip",
                 td_2_sa_d_model=None,
                 td_2_sa_nhead=None,
                 td_2_sa_pos_enc=None,
                 td_2_sa_num_layers=None,
                 td_2_sa_h=None,
                 td_2_sa_dropout=None,
                 td_2_lstm_h=None,
                 td_2_lstm_num_layers=None,
                 td_2_lstm_dropout=None,
                 td_2_lstm_bidirectional=None,

                 pool="att",
                 pool_att_h=128,
                 pool_att_dropout=0.1,

                 de_align="dot",
                 de_align_apply="hard",
                 de_fuse_dim=None,
                 de_fuse=True,

                 ):
        super().__init__()

        self.name = "NISQA_DE"

        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
        )

        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
        )

        self.align = Alignment(
            de_align,
            de_align_apply,
            q_dim=self.time_dependency.fan_out,
            y_dim=self.time_dependency.fan_out,
        )

        self.fuse = Fusion(
            in_feat=self.time_dependency.fan_out,
            fuse_dim=de_fuse_dim,
            fuse=de_fuse,
        )

        self.time_dependency_2 = TimeDependency(
            input_size=self.fuse.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
        )

        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
        )

    def _split_ref_deg(self, x, n_wins):
        (x, y) = torch.chunk(x, 2, dim=2)
        (n_wins_x, n_wins_y) = torch.chunk(n_wins, 2, dim=1)
        n_wins_x = n_wins_x.view(-1)
        n_wins_y = n_wins_y.view(-1)
        return x, y, n_wins_x, n_wins_y

    def forward(self, x, n_wins):
        x, y, n_wins_x, n_wins_y = self._split_ref_deg(x, n_wins)

        x = self.cnn(x, n_wins_x)
        y = self.cnn(y, n_wins_y)

        x, n_wins_x = self.time_dependency(x, n_wins_x)
        y, n_wins_y = self.time_dependency(y, n_wins_y)

        y = self.align(x, y, n_wins_y)

        x = self.fuse(x, y)

        x, n_wins_x = self.time_dependency_2(x, n_wins_x)

        x = self.pool(x, n_wins_x)

        return x
