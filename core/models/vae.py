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
from core.data.data import *
from core.utils.utils import *

import optuna
from optuna.trial import TrialState

plt.style.use(mhep.style.CMS)

torch.autograd.set_detect_anomaly(True)

device = torch.device(get_free_gpu())

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

####################################### DEFINE MODEL ####################################
# # Define models' architecture & helper functions
class ConvNet(nn.Module):
    def __init__(self, configs, tr_max, tr_min, trial):
        super(ConvNet, self).__init__()

        self.tr_max = tr_max
        self.tr_min = tr_min

        # Hyperparameters
        # Input data specific params
        self.num_features = configs['training']['num_features']
        #self.latent_dim = configs['training']['latent_dim']
        self.latent_dim = trial.suggest_int('latent_dim', configs['training']['latent_dim_min'],configs['training']['latent_dim_max'],step=10)

        print("num_features: "+str(self.num_features))
        print("latent_dim: "+str(self.latent_dim))

        self.num_particles = configs['physics']['num_particles']
        self.jet_type = configs['physics']['jet_type']

        # Training params
        self.n_epochs = configs['training']['n_epochs']
        self.batch_size = trial.suggest_int('batch_size', configs['training']['batch_size_min'],configs['training']['batch_size_max'])
        self.learning_rate = trial.suggest_float("learning_rate", configs['training']['learning_rate_min'], configs['training']['learning_rate_max'], log=True)
        self.saving_epoch = configs['training']['saving_epoch']
        self.n_filter = trial.suggest_int('n_filter', configs['training']['n_filter_min'],configs['training']['n_filter_max'],step=5)
        self.n_classes = configs['training']['n_classes']
        self.n_linear = trial.suggest_int('n_linear', configs['training']['n_linear_min'],configs['training']['n_linear_max'],step=100)
        self.n_kernel = trial.suggest_int('n_kernel', configs['training']['n_kernel_min'],configs['training']['n_kernel_max'])
        self.latent_dim_seq = [configs['training']['latent_dim_seq']]
        self.beta = trial.suggest_float("beta", configs['training']['beta_min'], configs['training']['beta_max'])
        self.n_layers = trial.suggest_int("n_layers", configs['training']['n_layers_min'], configs['training']['n_layers_max'])
        self.act_function = trial.suggest_categorical("act_function",configs['training']['act_function'])
        
        # Regularizer for loss penalty
        # Jet features loss weighting
        self.gamma = trial.suggest_float("gamma", configs['training']['gamma_min'], configs['training']['gamma_max'])
        self.gamma_1 = trial.suggest_float("gamma_1", configs['training']['gamma_1_min'], configs['training']['gamma_1_max'])
        self.gamma_2 = trial.suggest_float("gamma_2", configs['training']['gamma_2_min'], configs['training']['gamma_2_max'])
        #gamma_2 = 1.0
        n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

        # Particle features loss weighting
        self.alpha = trial.suggest_float("alpha", configs['training']['alpha_min'], configs['training']['alpha_max'])

        # Probability to keep a node in the dropout layer
        self.drop_prob = trial.suggest_float("drop_prob", configs['training']['drop_prob_min'], configs['training']['drop_prob_max'])

        seed = configs['training']['seed']

        set_seed(seed)

        #self.conv1 = nn.Conv2d(1, 1 * self.n_filter, kernel_size=(self.num_features,self.n_kernel), stride=(1), padding=(0))
        #self.conv2 = nn.Conv2d(1 * self.n_filter, 2 * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0))
        #self.conv3 = nn.Conv2d(2 * self.n_filter, 4 * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0))
        #
        layers_enc = []
        for i in range(self.n_layers): 
            if i==0: layers_enc.append(nn.Conv2d(1, (2**i) * self.n_filter, kernel_size=(self.num_features,self.n_kernel), stride=(1), padding=(0)))
            else: layers_enc.append(nn.Conv2d((2**(i-1)) * self.n_filter, (2**i) * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0)))
            #layers_enc.append(nn.ReLU())
            if self.act_function=="ReLU": layers_enc.append(nn.ReLU())
            if self.act_function=="GeLU": layers_enc.append(nn.GELU())
            if self.act_function=="ELU": layers_enc.append(nn.ELU())
            if self.act_function=="SELU": layers_enc.append(nn.SELU())
            if self.act_function=="LeakyReLU": layers_enc.append(nn.LeakyReLU(0.1))
        self.enc_conv_layers = nn.Sequential(*layers_enc)
        #
        self.fc1 = nn.Linear(1 * int(self.num_particles - ((self.n_kernel-1)*self.n_layers)) * (2**(self.n_layers-1)) * self.n_filter, self.n_linear)
        self.fc2 = nn.Linear(self.n_linear, 2 * self.latent_dim)
        #
        self.fc3 = nn.Linear(self.latent_dim, self.n_linear)
        self.fc4 = nn.Linear(self.n_linear, 1 * int(self.num_particles - ((self.n_kernel-1)*self.n_layers)) * (2**(self.n_layers-1)) *self.n_filter)
        #self.conv4 = nn.ConvTranspose2d(4 * self.n_filter, 2 * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0))
        #self.conv5 = nn.ConvTranspose2d(2 * self.n_filter, 1 * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0))
        #self.conv6 = nn.ConvTranspose2d(1 * self.n_filter, 1, kernel_size=(self.num_features,self.n_kernel), stride=(1), padding=(0))
        #
        layers_dec = []
        for i in range(self.n_layers): 
            if i==self.n_layers-1: layers_dec.append(nn.ConvTranspose2d(self.n_filter, 1, kernel_size=(self.num_features,self.n_kernel), stride=(1), padding=(0)))
            else: layers_dec.append(nn.ConvTranspose2d((2**(self.n_layers-i-1)) * self.n_filter, (2**(self.n_layers-i-2)) * self.n_filter, kernel_size=(1,self.n_kernel), stride=(1), padding=(0)))
            if i != (self.n_layers-1):# layers_dec.append(nn.ReLU())
                if self.act_function=="ReLU": layers_dec.append(nn.ReLU())
                if self.act_function=="GeLU": layers_dec.append(nn.GELU())
                if self.act_function=="ELU": layers_dec.append(nn.ELU())
                if self.act_function=="SELU": layers_dec.append(nn.SELU())
                if self.act_function=="LeakyReLU": layers_dec.append(nn.LeakyReLU(0.1))
        self.dec_conv_layers = nn.Sequential(*layers_dec)
        
        self.drop = nn.Dropout(self.drop_prob)
        
        print("Latent dim: {} -- N layers: {} -- N filter: {} -- N kernel: {} -- N linear: {}".format(self.latent_dim,self.n_layers,self.n_filter,self.n_kernel,self.n_linear))

    def encode(self, x):
        #out = self.conv1(x)
        #out = torch.relu(out)
        #out = self.conv2(out)
        #out = torch.relu(out)
        #out = self.conv3(out)
        #out = torch.relu(out)
        #print("Enc1: {}".format(x.shape))
        out = self.enc_conv_layers(x)
        #print("Enc2: {}".format(out.shape))
        out = out.view(out.size(0), -1) # flattening
        #print("Enc3: {}".format(out.shape))
        out = self.fc1(out)
        out = torch.relu(out)
        #print("Enc4: {}".format(out.shape))
        out = self.fc2(out)
        #print("Enc5: {}".format(out.shape))
        mean = out[:,:self.latent_dim]
        logvar = 1e-6 + (out[:,self.latent_dim:])
        return mean, logvar

    def decode(self, z):
        #print("Dec1: {}".format(z.shape))
        out = self.fc3(z)
        out = torch.relu(out)
        #print("Dec2: {}".format(out.shape))
        out = self.fc4(out)
        out = torch.relu(out)
        #print("Dec3: {}".format(out.shape))
        out = out.view(self.batch_size, (2**(self.n_layers-1)) * self.n_filter, 1, int(self.num_particles - ((self.n_kernel-1)*self.n_layers))) # reshaping
        #out = self.conv4(out)
        #out = torch.relu(out)
        #out = self.conv5(out)
        #out = torch.relu(out)
        #out = self.conv6(out)
        #print("Dec4: {}".format(out.shape))
        out = self.dec_conv_layers(out)
        #print("Dec5: {}".format(out.shape))
        out = torch.sigmoid(out)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        KL_divergence = (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / self.batch_size)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out, KL_divergence

# Jet observables manual calculation
def jet_p(p_part): # input should be of shape[batch_size, features, Nparticles]
    pjet = torch.sum(p_part, dim=2).to(device) # [batch, features (px, py, pz)]
    return pjet

def jet_Energy (p_part): # input should be of shape [batch_size, features, Nparticles]
    E_particles = torch.sqrt(torch.sum(p_part*p_part, dim=1)) # E per particle shape: [100, 30]
    E_jet = torch.sum(E_particles, dim=1).to(device) # Energy per jet [100]
    return E_jet

def jet_mass (p_part):
    jet_e = jet_Energy(p_part)
    P_jet = jet_p(p_part)
    m_jet = torch.sqrt(jet_e*jet_e - (P_jet[:,0]*P_jet[:,0]) - (P_jet[:,1]*P_jet[:,1]) - (P_jet[:,2]*P_jet[:,2])).to(device)
    return m_jet # mass per jet [100]

def jet_pT(p_part):# input should be of shape[batch_size, features, Nparticles]
    p_jet = jet_p(p_part) # [100, 3]
    jet_px = p_jet[:, 0]  # [100]
    jet_py = p_jet[:, 1]  # [100]
    jet_pt = torch.sqrt(jet_px*jet_px + jet_py*jet_py)
    return jet_pt

# Custom loss function VAtrE
# def compute_loss(denorm(x), denorm(x_decoded), KL_divergence, tr_max, tr_min):

def compute_loss(model, x, x_decoded, KL_divergence):

    x_aux = torch.clone(denorm(x, model.tr_max, model.tr_min))
    x_decoded_aux = torch.clone(denorm(x_decoded, model.tr_max, model.tr_min))

    pdist = nn.PairwiseDistance(p=2) # Euclidean distance
    x_pos = torch.zeros(model.batch_size, 1, model.num_features, model.num_particles).to(device) #variaveis do config
    x_pos = x_aux.to(device) # [100, 1, 3, 30]
    jets_pt = (jet_pT(x_pos[:,0,:,:]).unsqueeze(1).to(device))#/jet_pt_std # [100, 1]
    jets_mass = (jet_mass(x_pos[:,0,:,:]).unsqueeze(1).to(device))#/jet_mass_std
    x_pos = torch.transpose(x_pos,dim0=2,dim1=3)

    x_decoded_pos = torch.zeros(model.batch_size,1,model.num_features,model.num_particles).to(device)
    x_decoded_pos = x_decoded_aux.to(device) # [100, 1, 3, 30]
    jets_pt_reco = (jet_pT(x_decoded_pos[:,0,:,:]).unsqueeze(1).to(device))#/jet_pt_std # [100, 1]
    jets_mass_reco = (jet_mass(x_decoded_pos[:,0,:,:]).unsqueeze(1).to(device))#/jet_mass_std
    x_decoded_pos = torch.transpose(x_decoded_pos,dim0=2,dim1=3)
    x_decoded_pos = x_decoded_pos.view(model.batch_size, 1, model.num_particles, 1, model.num_features)
    x_decoded_pos = x_decoded_pos.repeat(1,1,1,model.num_particles,1)

    # Permutation-invariant Loss / NND / 3D Sparse Loss
    dist_aux = torch.pow(pdist(x_pos, x_decoded_pos),2)
    dist_diag = torch.diagonal(dist_aux, dim1=0, dim2=1)
    dist = torch.squeeze(torch.transpose(dist_diag.view(1,model.num_particles,model.num_particles,model.batch_size),dim0=0,dim1=3))

    # NND original version
    jet_pt_dist = torch.pow(pdist(jets_pt, jets_pt_reco),2)
    jet_mass_dist = torch.pow(pdist(jets_mass, jets_mass_reco),2)

    # For every output value, find its closest input value; for every input value, find its closest output value.
    ieo = torch.min(dist, dim = 1)  # Get min distance per row - Find the closest input to the output
    oei = torch.min(dist, dim = 2)  # Get min distance per column - Find the closest output to the input
    # Symmetrical euclidean distances
    eucl = (ieo.values + oei.values)#*x_decoded_aux[:,0,3] # [100, 30]1

    # Loss per jet (batch size)
    loss_rec_p = model.alpha*(torch.sum(eucl, dim=1))
    loss_rec_j = model.gamma*(model.gamma_1*(jet_pt_dist) + model.gamma_2*(jet_mass_dist))
    eucl = loss_rec_p + loss_rec_j  # [100]

    # Loss individual components
    loss_rec_p = torch.sum(loss_rec_p)
    loss_rec_j = torch.sum(loss_rec_j)
    jet_pt_dist = torch.sum(jet_pt_dist)
    jet_mass_dist = torch.sum(jet_mass_dist)

    # Average symmetrical euclidean distance per image
    eucl = (torch.sum(eucl) / model.batch_size)
    reconstruction_loss = - eucl

    # Separate particles' loss components to plot them
    #KL_divergence = (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size)
    ELBO = ((1-model.beta)*reconstruction_loss) - (model.beta*KL_divergence)
    loss = - ELBO

    return loss, eucl, loss_rec_p, loss_rec_j, jet_pt_dist, jet_mass_dist

##### Training function per batch #####
def train(model, batch_data_train, optimizer):
    """train_loss = 0.0
    train_KLD_loss = 0.0
    train_reco_loss = 0.0"""

    tr_max = model.tr_max
    tr_min = model.tr_min

    input_train = batch_data_train[:, :, :].to(device)

    output_train, train_KLD_loss = model(input_train)

    # loss per batch
    train_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass  = compute_loss(model, input_train, output_train, train_KLD_loss)

    # Backprop and perform Adam optimisation
    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    # Adam optimization using the gradients from backprop
    optimizer.step()

    return input_train, output_train, train_loss, train_KLD_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass

##### Validation function per batch #####
def validate(model, batch_data_valid):
    """valid_loss = 0
    valid_KLD_loss = 0
    valid_reco_loss = 0"""

    tr_max = model.tr_max
    tr_min = model.tr_min

    model.eval()
    #testar

    with torch.no_grad():
        #test
        input_valid = batch_data_valid.to(device)
        x_decoded, KL_divergence = model(input_valid)

        # loss per batch
        valid_loss, valid_reco_loss, valid_reco_loss_p, valid_reco_loss_j, valid_reco_loss_pt, valid_reco_loss_mass = compute_loss(model, x_decoded, input_valid, KL_divergence)

        return valid_loss, KL_divergence, valid_reco_loss

##### Test function #####
def test_unseed_data(model, batch_data_test):

    tr_max = model.tr_max
    tr_min = model.tr_min

    model.eval()
    with torch.no_grad():
        #test
        input_test = batch_data_test.to(device)
        x_decoded, KL_divergence = model(input_test)

        test_loss, test_reco_loss, loss_particle, loss_jet, jet_pt_loss, jet_mass_loss = compute_loss(model, x_decoded, input_test, KL_divergence)
    return input_test, x_decoded, test_loss, KL_divergence, test_reco_loss
