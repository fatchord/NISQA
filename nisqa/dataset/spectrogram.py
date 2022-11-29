# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import time
import librosa as lb
import numpy as np
import torch


# %% Spectrograms
def segment_specs(file_path, x, seg_length, seg_hop=1, max_length=None):
    """
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step,
    the neighboring bins are included as input to the CNN. For example
    for a seg_length of 7, the previous 3 and the follwing 3 frequency
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    """
    if seg_length % 2 == 0:
        raise ValueError("seg_length must be odd! (seg_lenth={})".format(seg_length))
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    n_wins = x.shape[1] - (seg_length - 1)

    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1, 0)[idx3, :].unsqueeze(1).transpose(3, 2)

    if seg_hop > 1:
        x = x[::seg_hop, :]
        n_wins = int(np.ceil(n_wins / seg_hop))

    if max_length is not None:
        if max_length < n_wins:
            raise ValueError(
                "n_wins {} > max_length {} --- {}. Increase max window length ms_max_segments!".format(n_wins,
                                                                                                       max_length,
                                                                                                       file_path))
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins, :] = x
        x = x_padded

    return x, np.array(n_wins)


def get_librosa_melspec(
        file_path,
        sr=48e3,
        n_fft=1024,
        hop_length=80,
        win_length=170,
        n_mels=32,
        fmax=16e3,
        ms_channel=None,
):
    """
    Calculate mel-spectrograms with Librosa.
    """
    # Calc spec
    try:
        start = time.perf_counter()
        y, sr = lb.core.load(file_path, sr=sr)
        y = y[100*sr:152*sr]
        if len(y) < sr * 2:
            print(f"Warning - short audio {file_path}")
            y = np.zeros((sr * 52,)).astype(np.float32)

        end = time.perf_counter()
        print(f"{end - start:.2f}s to load {file_path}")
    except:
        raise ValueError("Could not load file {}".format(file_path))

    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,

        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm="slaney",
    )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec
