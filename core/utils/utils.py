import sys
import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import skhep.math as hep
import os
from functools import reduce
from matplotlib.colors import LogNorm
from pathlib import Path
#from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import awkward as ak
import random
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
import mplhep as mhep

plt.style.use(mhep.style.CMS)

torch.autograd.set_detect_anomaly(True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def inverse_standardize_t(X, tmin, tmax):
    mean = tmin
    std = tmax
    original_X = ((X * (std - mean)) + mean)
    return original_X
    
def mask_zero_padding(input_data):
    # Mask input for zero-padded particles. Set to zero values between -10^-4 and 10^-4
    px = input_data[:,0,:]
    py = input_data[:,1,:]
    pz = input_data[:,2,:]
    mask_px = ((px <= -0.0001) | (px >= 0.0001))
    mask_py = ((py <= -0.0001) | (py >= 0.0001))
    mask_pz = ((pz <= -0.0001) | (pz >= 0.0001))
    masked_px = (px * mask_px) + 0.0
    masked_py = (py * mask_py) + 0.0
    masked_pz = (pz * mask_pz) + 0.0
    data = torch.stack([masked_px, masked_py, masked_pz], dim=1)
    return data

def mask_min_pt(output_data):
    # Mask output for min-pt
    min_pt_cut = 0.25
    mask =  output_data[:,0,:] * output_data[:,0,:] + output_data[:,1,:] * output_data[:,1,:] > min_pt_cut**2
    # Expand over the features' dimension
    mask = mask.unsqueeze(1)
    # Then, you can apply the mask
    data_masked = mask * output_data
    return data_masked

def compute_eta(pz, pt):
    eta = np.nan_to_num(np.arcsinh(pz/pt))
    return eta

def compute_phi(px, py):
    phi = np.arctan2(py, px)
    return phi

def particle_pT(p_part):
    p_px = p_part[:, :, 0]
    p_py = p_part[:, :, 1]
    p_pt = np.sqrt(p_px*p_px + p_py*p_py)
    return p_pt

def ptetaphim_particles(in_dataset):
    part_pt = particle_pT(in_dataset)
    part_eta = compute_eta(in_dataset[:,:,2],part_pt)
    part_phi = compute_phi(in_dataset[:,:,0],in_dataset[:,:,1])
    part_mass = in_dataset[:,:,3]
    return np.stack((part_pt, part_eta, part_phi, part_mass), axis=2)

def jet_features(jets, mask_bool=False, mask=None):
    vecs = ak.zip({
        "pt": jets[:, :, 0],
        "eta": jets[:, :, 1],
        "phi": jets[:, :, 2],
        "mass": jets[:, :, 3],
        }, with_name="PtEtaPhiMLorentzVector")
    sum_vecs = vecs.sum(axis=1)
    jf = np.stack((ak.to_numpy(sum_vecs.mass), ak.to_numpy(sum_vecs.pt), ak.to_numpy(sum_vecs.energy), ak.to_numpy(sum_vecs.eta), ak.to_numpy(sum_vecs.phi)), axis=1)
    return ak.to_numpy(jf)

def part_flatten(jet):
    # jet has shape (num_jets, num_particles, num_features)
    jetx = jet[:,:,0].flatten()
    jety = jet[:,:,1].flatten()
    jetz = jet[:,:,2].flatten()
    jet_flat = np.stack((jetx,jety,jetz),axis=1)
    # jet_flat has shape (num_jets*num_particles, num_features)
    return jet_flat


def rmse(s, o):
    """
        Root Mean-squared error


        input:
        s: simulated
        o: observed
    output:
        rmse: root mean-squared error
    """
    rmse = np.sqrt(np.mean((o - s) ** 2))
    return rmse


def index_agreement(s, o):
    """
        index of agreement

        Willmott (1981, 1982)
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 - (np.sum((o - s) ** 2)) / (
        np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    )
    return ia


def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        index of agreement

        Willmott (1981, 1982)
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 - (torch.sum((o - s) ** 2)) / (
        torch.sum((torch.abs(s - torch.mean(o)) + torch.abs(o - torch.mean(o))) ** 2)
    )

    return ia
