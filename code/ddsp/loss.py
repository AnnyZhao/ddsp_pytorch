# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss

class Loss(_Loss):

    def __init__(self):
        super(Loss, self).__init__()
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass

# Lambda for computing squared amplitude
amp = lambda x: x[...,0]**2 + x[...,1]**2


class MSSTFT(nn.Module):
    def __init__(self, scales=[64, 128, 256, 512], overlap=0.75):
        super(MSSTFT, self).__init__()
        self.scales = scales
        self.overlap = overlap
        self.windows = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False)
            for scale in self.scales)

    def forward(self, x):
        stfts = []
        device = x.device
        # Compute multiple STFT for x
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1 - self.overlap) * scale), center=False)
            stfts.append(amp(cur_fft).to(device))
        return stfts


class MSSTFTLoss(Loss):
    """
    Multi-scale STFT loss.
    Compute the FFT of a signal at multiple scales.

    Arguments:
            scales ([int])  : array of N_FFT paramters, corresponding to the different scales of FFT.
            overlap (float) : amount of overlap of the FFT windows.
    """

    def __init__(self, scales=[64, 128, 256, 512], overlap=0.75):
        super(MSSTFTLoss, self).__init__()
        self.apply(self.init_parameters)
        self.msstft = MSSTFT(scales=scales, overlap=overlap)

    def init_parameters(self, m):
        pass

    def forward(self, x, stfts_orig):
        # First compute multiple STFT for x
        stfts = self.msstft(x).to(stfts_orig[0].device)
        # Compute loss
        lin_loss = sum([torch.mean(torch.abs(stfts_orig[i][j] - stfts[i][j])) for i in range(len(stfts)) for j in range(len(stfts[i]))])
        log_loss = sum([torch.mean(torch.abs(torch.log(stfts_orig[i][j] + 1e-4) - torch.log(stfts[i][j] + 1e-4))) for i in range(len(stfts)) for j in range(len(stfts[i])) ])
        return lin_loss + log_loss
