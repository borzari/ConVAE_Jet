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
