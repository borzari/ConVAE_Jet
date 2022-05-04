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
latent_dim = configs['training']['latent_dim']
beta = configs['training']['beta'] # equivalent to beta=5000 in the old setup
num_features = configs['training']['num_features']
vae_mode = configs['training']['vae_mode']
dataset_description = configs['data']['dataset_description']
normalize = configs['data']['normalize']

# Regularizer for loss penalty
# Jet features loss weighting
gamma = configs['training']['gamma']
#gamma_1 = configs['training']['gamma_1']
#gamma_2 = configs['training']['gamma_2']
n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

# Particle features loss weighting
alpha = configs['training']['alpha']

tr_max, tr_min = [],[]

####################################### LOAD DATA #######################################
class DataT():
    def __init__(self):
        super(DataT, self).__init__()
        self.tr_max = []
        self.tr_min = []

        print("To no init! ")

        self.train_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['train_dataset']))
        self.valid_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['valid_dataset']))
        self.test_dataset = torch.load(os.path.join(configs['paths']['dataset_dir'], configs['data']['test_dataset']))

        # This calculates the value of gamma_2 using the ratio between the pt and the mass of the jets, with the datasets of 150p
        self.gamma_1 = 1.0
        px_aux = self.train_dataset[:,0]
        py_aux = self.train_dataset[:,1]
        pz_aux = self.train_dataset[:,2]
        mass_aux = np.zeros((pz_aux.shape[0], 150))

        input_data = np.stack((px_aux, py_aux, pz_aux, mass_aux), axis=2)
        hadr_input_data = ptetaphim_particles(input_data)
        jets_input_data = jet_features(hadr_input_data)

        jets_pt = jets_input_data[:,1]
        jets_mass = jets_input_data[:,0]

        jets_pt_mean = np.mean(jets_pt)
        jets_mass_mean = np.mean(jets_mass)

        self.gamma_2 = jets_pt_mean/jets_mass_mean

        self.train_dataset = self.train_dataset[:int(len(self.train_dataset)*configs['data']['data_percentage']),:,:num_particles]
        self.valid_dataset = self.valid_dataset[:int(len(self.valid_dataset)*configs['data']['data_percentage']),:,:num_particles]
        self.test_dataset = self.test_dataset[:int(len(self.test_dataset)*configs['data']['data_percentage']),:,:num_particles]

        self.train_dataset = self.train_dataset.view(len(self.train_dataset),1,num_features,num_particles)
        self.valid_dataset = self.valid_dataset.view(len(self.valid_dataset),1,num_features,num_particles)
        self.test_dataset = self.test_dataset.view(len(self.test_dataset),1,num_features,num_particles)

        self.train_dataset = self.train_dataset.cpu()
        self.valid_dataset = self.valid_dataset.cpu()
        self.test_dataset = self.test_dataset.cpu()


        for i in range(num_features):
            self.tr_max.append(torch.max(self.train_dataset[:,0,i]))
            self.tr_min.append(torch.min(self.train_dataset[:,0,i]))
            if normalize is True:
                self.train_dataset[:,0,i] = (self.train_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])
                self.valid_dataset[:,0,i] = (self.valid_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])
                self.test_dataset[:,0,i] = (self.test_dataset[:,0,i] - self.tr_min[i])/(self.tr_max[i] - self.tr_min[i])

        self.gen_dataset = torch.zeros([self.test_dataset.shape[0], 1, num_features, num_particles])

        print("\ntrain_dataset.shape: ",self.train_dataset.shape)
        print("valid_dataset.shape: ",self.valid_dataset.shape)
        print("test_dataset.shape:  ",self.test_dataset.shape)
        print("gen_dataset.shape:   ",self.gen_dataset.shape,"\n")

        #return self.train_dataset, self.valid_dataset, self.test_dataset, self.gen_dataset, tr_max, tr_min

    def create_folders(self, cur_latent_dim):
        model_name = dataset_description+'_'+jet_type+'jets_'+str(cur_latent_dim)+'latent_'+str(n_filter)+'filters_'+str(n_epochs)+'epochs_'+str(beta).replace(".","p")+'beta_'+str(num_particles)+'p_jetpt_jetmass'
        dir_name='dir_'+str(vae_mode)+'_'+model_name
        print("to dentro do create_folders: ", model_name)
        cur_jets_dir = os.path.join(configs['paths']['jets_dir'], dir_name)
        cur_report_dir = os.path.join(configs['paths']['report_dir'], dir_name)
        cur_model_dir = os.path.join(configs['paths']['model_dir'], dir_name)

        # create folder recursively
        Path(cur_jets_dir).mkdir(parents=True, exist_ok=True)
        Path(cur_report_dir).mkdir(parents=True, exist_ok=True)
        Path(cur_model_dir).mkdir(parents=True, exist_ok=True)

        return cur_jets_dir, cur_report_dir, cur_model_dir, model_name, dir_name

    def create_loaders(self):
        print("To dentro de loaders: ",len(self.train_dataset))

        # Create iterable data loaders
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        gen_loader = DataLoader(dataset=self.gen_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader, gen_loader


    ###### fim classe ##########

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

    #num_features = len(train_dataset[0,0]) #verificar se é necessário

    for i in range(num_features):
        tr_max.append(torch.max(train_dataset[:,0,i]))
        tr_min.append(torch.min(train_dataset[:,0,i]))
        if normalize is True:
            train_dataset[:,0,i] = (train_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
            valid_dataset[:,0,i] = (valid_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
            test_dataset[:,0,i] = (test_dataset[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])

    gen_dataset = torch.zeros([test_dataset.shape[0], 1, num_features, num_particles])

    return train_dataset, valid_dataset, test_dataset, gen_dataset, tr_max, tr_min

def create_loaders(train_dataset, valid_dataset, test_dataset, gen_dataset):
    # Create iterable data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(dataset=gen_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, gen_loader

def tr(): #solução péssima. apenas para teste
    train_dataset, valid_dataset, test_dataset, gen_dataset, tr_max, tr_min = generate_datasets()
    return tr_max, tr_min

def generate_folders(latent_dim):
    model_name = dataset_description+'_'+str(latent_dim)+'latent_'+str(n_filter)+'filters_'+str(n_epochs)+'epochs_'+str(beta).replace(".","p")+'beta_min'+str(num_particles)+'p_jetpt_jetmass'
    dir_name='dir_'+str(vae_mode)+'_'+model_name

    cur_jets_dir = os.path.join(configs['paths']['jets_dir'], dir_name)
    cur_report_dir = os.path.join(configs['paths']['report_dir'], dir_name)
    cur_model_dir = os.path.join(configs['paths']['model_dir'], dir_name)

    # create folder recursively
    Path(cur_jets_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_report_dir).mkdir(parents=True, exist_ok=True)
    Path(cur_model_dir).mkdir(parents=True, exist_ok=True)

    return cur_jets_dir, cur_report_dir, cur_model_dir, model_name, dir_name

def norm(norm_in, tr_max, tr_min):
    if normalize is True:
        norm_out = torch.clone(norm_in)
        for i in range(num_features):
                norm_out[:,0,i] = (norm_out[:,0,i] - tr_min[i])/(tr_max[i] - tr_min[i])
        return norm_out
    else: return norm_in

def denorm(denorm_in, tr_max, tr_min):
    if normalize is True:
        denorm_out = torch.clone(denorm_in)
        for i in range(num_features):
            denorm_out[:,0,i] = ((denorm_out[:,0,i] * (tr_max[i] - tr_min[i])) + tr_min[i])
        return denorm_out
    else: return denorm_in

def main():
    cur_jets_dir, cur_report_dir, cur_model_dir = generate_folders()
    print(cur_jets_dir)
    print(cur_report_dir)
    print(cur_model_dir)

if __name__=='__main__':
    main()
