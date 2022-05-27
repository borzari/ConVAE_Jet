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

import optuna
from optuna.trial import TrialState

#trial = TrialState()

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default='config/config_opt.json', help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--load', type=str, help='this param load model')

    # parse the arguments
    args = parser.parse_args()

    return args


args = parse_args()

# load configurations of model and others
configs = json.load(open(args.config, 'r'))


tr_max, tr_min = [],[]

####################################### LOAD DATA #######################################
class DataT():
    def __init__(self, trial):
        super(DataT, self).__init__()
        self.tr_max = []
        self.tr_min = []

        print("To no init! ")

        args = parse_args()

        # load configurations of model and others
        configs = json.load(open(args.config, 'r'))

        # Hyperparameters
        # Input data specific params
        self.num_particles = configs['physics']['num_particles']
        self.jet_type = configs['physics']['jet_type']

        # Training params
        self.n_epochs = configs['training']['n_epochs']
        self.batch_size = trial.suggest_int('batch_size', configs['training']['batch_size_min'],configs['training']['batch_size_max'])
        #self.n_filter = configs['training']['n_filter']
        self.n_classes = configs['training']['n_classes']
        self.latent_dim_seq = [configs['training']['latent_dim_seq']]
        #self.latent_dim = configs['training']['latent_dim']
        self.beta = trial.suggest_float("beta", configs['training']['beta_min'], configs['training']['beta_max'])
        self.num_features = configs['training']['num_features']
        self.vae_mode = configs['training']['vae_mode']
        self.dataset_description = configs['data']['dataset_description']
        self.normalize = configs['data']['normalize']

        # Regularizer for loss penalty
        # Jet features loss weighting
        self.gamma = trial.suggest_float("gamma", configs['training']['gamma_min'], configs['training']['gamma_max'])
        self.gamma_1 = trial.suggest_float("gamma_1", configs['training']['gamma_1_min'], configs['training']['gamma_1_max'])
        self.gamma_2 = trial.suggest_float("gamma_2", configs['training']['gamma_2_min'], configs['training']['gamma_2_max'])
        #gamma_2 = 1.0
        n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

        # Particle features loss weighting
        self.alpha = trial.suggest_float("alpha", configs['training']['alpha_min'], configs['training']['alpha_max'])

        self.train_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['train_dataset']))
        self.valid_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['valid_dataset']))
        self.test_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['test_dataset']))

        self.train_dataset = self.train_dataset[:int(len(self.train_dataset)*configs['data']['data_percentage'])]
        self.valid_dataset = self.valid_dataset[:int(len(self.valid_dataset)*configs['data']['data_percentage'])]
        self.test_dataset = self.test_dataset[:int(len(self.test_dataset)*configs['data']['data_percentage'])]

        self.train_dataset = self.train_dataset.view(len(self.train_dataset),1,self.num_features, self.num_particles)
        self.valid_dataset = self.valid_dataset.view(len(self.valid_dataset),1,self.num_features, self.num_particles)
        self.test_dataset = self.test_dataset.view(len(self.test_dataset),1,self.num_features, self.num_particles)

        self.train_dataset = self.train_dataset.cpu()
        self.valid_dataset = self.valid_dataset.cpu()
        self.test_dataset = self.test_dataset.cpu()

        #num_features = len(train_dataset[0,0]) #verificar se é necessário

        for i in range(self.num_features):
            self.tr_max.append(torch.max(self.train_dataset[:,0,i]))
            self.tr_min.append(torch.min(self.train_dataset[:,0,i]))
            if self.normalize is True:
                self.train_dataset[:,0,i] = (self.train_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])
                self.valid_dataset[:,0,i] = (self.valid_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])
                self.test_dataset[:,0,i] = (self.test_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])

        self.gen_dataset = torch.zeros([self.test_dataset.shape[0], 1, self.num_features, self.num_particles])

        print("\ntrain_dataset.shape: ",self.train_dataset.shape)
        print("valid_dataset.shape: ",self.valid_dataset.shape)
        print("test_dataset.shape:  ",self.test_dataset.shape)
        print("gen_dataset.shape:   ",self.gen_dataset.shape,"\n")

    def create_loaders(self):
        print("To dentro de loaders: ",len(self.train_dataset))
        # Create iterable data loaders
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        gen_loader = DataLoader(dataset=self.gen_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader, gen_loader


    ###### fim classe ##########

def norm(norm_in, tr_max, tr_min):
    if configs['data']['normalize'] is True:
        norm_out = torch.clone(norm_in)
        for i in range(configs['training']['num_features']):
                norm_out[:,0,i] = (norm_out[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
        return norm_out
    else: return norm_in

def denorm(denorm_in, tr_max, tr_min):
    if configs['data']['normalize'] is True:
        denorm_out = torch.clone(denorm_in)
        for i in range(configs['training']['num_features']):
            denorm_out[:,0,i] = ((denorm_out[:,0,i] * (tr_max[i] - tr_min[i])) + tr_min[i])
        return denorm_out
    else: return denorm_in
