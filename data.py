#########################################################################################
# VAE 3D Sparse loss - trained on JEDI-net gluons' dataset
#########################################################################################
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

from core.utils.utils import *
from core.models.vae import *

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default='config/config_default.json', help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--load', type=str, help='this param load model')

    # parse the arguments
    args = parser.parse_args()

    return args


args = parse_args()

# load configurations of model and others
configs = json.load(open(args.config, 'r'))

# Hyperparameters
# Input data specific params
num_particles = configs['physics']['num_particles']
jet_type = configs['physics']['jet_type']

# Training params
n_epochs = configs['training']['n_epochs']
batch_size = configs['training']['batch_size']
learning_rate = configs['training']['learning_rate']
saving_epoch = configs['training']['saving_epoch']
n_filter = configs['training']['n_filter']
n_classes = configs['training']['n_classes']
latent_dim_seq = [configs['training']['latent_dim_seq']]
beta = configs['training']['beta'] # equivalent to beta=5000 in the old setup
num_features = configs['training']['num_features']

# Regularizer for loss penalty
# Jet features loss weighting
gamma = configs['training']['gamma']
gamma_1 = configs['training']['gamma_1']
gamma_2 = configs['training']['gamma_2']
#gamma_2 = 1.0
n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

# Particle features loss weighting
alpha = configs['training']['alpha']

####################################### LOAD DATA #######################################
def generate_datasets():
    train_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['train_dataset']))
    valid_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['valid_dataset']))
    test_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['test_dataset']))

    train_dataset = train_dataset.view(len(train_dataset),1,num_features,num_particles)
    valid_dataset = valid_dataset.view(len(valid_dataset),1,num_features,num_particles)
    test_dataset = test_dataset.view(len(test_dataset),1,num_features,num_particles)

    train_dataset = train_dataset.cpu()
    valid_dataset = valid_dataset.cpu()
    test_dataset = test_dataset.cpu()

    num_features = len(train_dataset[0,0]) #verificar se é necessário

    for i in range(num_features):
        tr_max.append(torch.max(train_dataset[:,0,i]))
        tr_min.append(torch.min(train_dataset[:,0,i]))
        if normalize is True:
            train_dataset[:,0,i] = (train_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
            valid_dataset[:,0,i] = (valid_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
            test_dataset[:,0,i] = (test_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])

    gen_dataset = torch.zeros([test_dataset.shape[0], 1, num_features, num_particles])

    return train_dataset, valid_dataset, test_dataset, gen_dataset

def create_loaders(train_dataset, valid_dataset, test_dataset, gen_dataset):
    # Create iterable data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(dataset=gen_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, gen_loader


def generate_folders():
    model_name = '_test_generation_originaltest_fulldata_'+str(latent_dim)+'latent_'+str(n_filter)+'filters_'+str(n_epochs)+'epochs_0p0to1p0_sparse_nnd_beta0p9998_train_evaluatemin'+str(num_particles)+'p_jetpt_jetmass_'
    dir_name='generation_beta0p9998_dir_second'+model_name+'test'

    cur_jets_dir = os.path.join(configs['paths']['jets_dir'], dir_name)
    cur_report_dir = os.path.join(configs['paths']['report_dir'], dir_name)
    cur_model_dir = os.path.join(configs['paths']['model_dir'], dir_name)

    # create folder recursively
    Path(cur_jets_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_report_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_model_dir).mkdir(parents=True, exist_ok=True)

    return cur_jets_dir, cur_report_dir, cur_model_dir


#falta gerar funções

#for n_filter in seq_n_filter:
for latent_dim in latent_dim_seq:

    #model_name = '_test_generation_originaltest_fulldata_'+str(latent_dim)+'latent_'+str(n_filter)+'filters_'+str(n_epochs)+'epochs_0p0to1p0_sparse_nnd_beta0p9998_train_evaluatemin'+str(num_particles)+'p_jetpt_jetmass_'
    #dir_name='generation_beta0p9998_dir_second'+model_name+'test'
    model_name = 'fulldata_'+str(latent_dim)+'latent_'+str(n_filter)+'filters_'+str(n_epochs)+'epochs_'+str(beta).replace(".","p")+'beta_min'+str(num_particles)+'p_jetpt_jetmass'
    dir_name='dir_'+str(vae_mode)'_'+model_name

    cur_jets_dir = os.path.join(configs['paths']['jets_dir'], dir_name)
    cur_report_dir = os.path.join(configs['paths']['report_dir'], dir_name)
    cur_model_dir = os.path.join(configs['paths']['model_dir'], dir_name)

    # create folder recursively
    Path(cur_jets_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_report_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_model_dir).mkdir(parents=True, exist_ok=True)
