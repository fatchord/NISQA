# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import os
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from .spectrogram import segment_specs, get_librosa_melspec


# %% Dataset
class SpeechQualityDataset(Dataset):
    """
    Dataset for Speech Quality Model.
    """

    def __init__(
            self,
            df,
            df_con=None,
            data_dir="",
            folder_column="",
            filename_column="filename",
            mos_column="MOS",
            seg_length=15,
            max_length=None,
            to_memory=False,
            to_memory_workers=0,
            transform=None,
            seg_hop_length=1,
            ms_n_fft=1024,
            ms_hop_length=80,
            ms_win_length=170,
            ms_n_mels=32,
            ms_sr=48e3,
            ms_fmax=16e3,
            ms_channel=None,
            double_ended=False,
            filename_column_ref=None,
            dim=False,
    ):

        self.df = df
        self.df_con = df_con
        self.data_dir = data_dir
        self.folder_column = folder_column
        self.filename_column = filename_column
        self.filename_column_ref = filename_column_ref
        self.mos_column = mos_column
        self.seg_length = seg_length
        self.seg_hop_length = seg_hop_length
        self.max_length = max_length
        self.transform = transform
        self.to_memory_workers = to_memory_workers
        self.ms_n_fft = ms_n_fft
        self.ms_hop_length = ms_hop_length
        self.ms_win_length = ms_win_length
        self.ms_n_mels = ms_n_mels
        self.ms_sr = ms_sr
        self.ms_fmax = ms_fmax
        self.ms_channel = ms_channel
        self.double_ended = double_ended
        self.dim = dim

        # if True load all specs to memory
        self.to_memory = False
        if to_memory:
            self._to_memory()

    def _to_memory_multi_helper(self, idx):
        return [self._load_spec(i) for i in idx]

    def _to_memory(self):
        if self.to_memory_workers == 0:
            self.mem_list = [self._load_spec(idx) for idx in tqdm(range(len(self)))]
        else:
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx) / buffer_size)
            idx = idx[:buffer_size * n_bufs].reshape(-1, buffer_size).tolist() + idx[buffer_size * n_bufs:].reshape(1,
                                                                                                                    -1).tolist()
            pool = multiprocessing.Pool(processes=self.to_memory_workers)
            mem_list = []
            for out in tqdm(pool.imap(self._to_memory_multi_helper, idx), total=len(idx)):
                mem_list = mem_list + out
            self.mem_list = mem_list
            pool.terminate()
            pool.join()
        self.to_memory = True

    def _load_spec(self, index):

        # Load spec
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])

        if self.double_ended:
            file_path_ref = os.path.join(self.data_dir, self.df[self.filename_column_ref].iloc[index])

        spec = get_librosa_melspec(
            file_path,
            sr=self.ms_sr,
            n_fft=self.ms_n_fft,
            hop_length=self.ms_hop_length,
            win_length=self.ms_win_length,
            n_mels=self.ms_n_mels,
            fmax=self.ms_fmax,
            ms_channel=self.ms_channel
        )

        if self.double_ended:
            spec_ref = get_librosa_melspec(
                file_path_ref,
                sr=self.ms_sr,
                n_fft=self.ms_n_fft,
                hop_length=self.ms_hop_length,
                win_length=self.ms_win_length,
                n_mels=self.ms_n_mels,
                fmax=self.ms_fmax
            )
            spec = (spec, spec_ref)

        return spec

    def __getitem__(self, index):
        assert isinstance(index, int), "index must be integer (no slice)"

        if self.to_memory:
            spec = self.mem_list[index]
        else:
            spec = self._load_spec(index)

        if self.double_ended:
            spec, spec_ref = spec

        # Apply transformation if given
        if self.transform:
            spec = self.transform(spec)

        # Segment specs
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])
        if self.seg_length is not None:
            x_spec_seg, n_wins = segment_specs(file_path,
                                               spec,
                                               self.seg_length,
                                               self.seg_hop_length,
                                               self.max_length)

            if self.double_ended:
                x_spec_seg_ref, n_wins_ref = segment_specs(file_path,
                                                           spec_ref,
                                                           self.seg_length,
                                                           self.seg_hop_length,
                                                           self.max_length)
        else:
            x_spec_seg = spec
            n_wins = spec.shape[1]
            if self.max_length is not None:
                x_padded = np.zeros((x_spec_seg.shape[0], self.max_length))
                x_padded[:, :n_wins] = x_spec_seg
                x_spec_seg = np.expand_dims(x_padded.transpose(1, 0), axis=(1, 3))
                if not torch.is_tensor(x_spec_seg):
                    x_spec_seg = torch.tensor(x_spec_seg, dtype=torch.float)

            if self.double_ended:
                x_spec_seg_ref = spec
                n_wins_ref = spec.shape[1]
                if self.max_length is not None:
                    x_padded = np.zeros((x_spec_seg_ref.shape[0], self.max_length))
                    x_padded[:, :n_wins] = x_spec_seg_ref
                    x_spec_seg_ref = np.expand_dims(x_padded.transpose(1, 0), axis=(1, 3))
                    if not torch.is_tensor(x_spec_seg_ref):
                        x_spec_seg_ref = torch.tensor(x_spec_seg_ref, dtype=torch.float)

        if self.double_ended:
            x_spec_seg = torch.cat((x_spec_seg, x_spec_seg_ref), dim=1)
            n_wins = np.concatenate((n_wins.reshape(1), n_wins_ref.reshape(1)), axis=0)

        # Get MOS (apply NaN in case of prediction only mode)
        if self.dim:
            if self.mos_column == "predict_only":
                y = np.full((5, 1), np.nan).reshape(-1).astype("float32")
            else:
                y_mos = self.df["mos"].iloc[index].reshape(-1).astype("float32")
                y_noi = self.df["noi"].iloc[index].reshape(-1).astype("float32")
                y_dis = self.df["dis"].iloc[index].reshape(-1).astype("float32")
                y_col = self.df["col"].iloc[index].reshape(-1).astype("float32")
                y_loud = self.df["loud"].iloc[index].reshape(-1).astype("float32")
                y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        else:
            if self.mos_column == "predict_only":
                y = np.full(1, np.nan).reshape(-1).astype("float32")
            else:
                y = self.df[self.mos_column].iloc[index].reshape(-1).astype("float32")

        return x_spec_seg, y, (index, n_wins)

    def __len__(self):
        return len(self.df)

