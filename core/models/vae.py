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

####################################### DEFINE MODEL ####################################
# # Define models' architecture & helper functions
class ConvNet(nn.Module):
    def __init__(self, configs):
        super(ConvNet, self).__init__()

        # Hyperparameters
        # Input data specific params
        self.num_features = configs['training']['num_features']
        self.latent_dim = configs['training']['latent_dim']

        print("num_features: "+str(self.num_features))
        print("latent_dim: "+str(self.latent_dim))

        self.num_particles = configs['physics']['num_particles']
        self.jet_type = configs['physics']['jet_type']

        # Training params
        self.n_epochs = configs['training']['n_epochs']
        self.batch_size = configs['training']['batch_size']
        self.learning_rate = configs['training']['learning_rate']
        self.saving_epoch = configs['training']['saving_epoch']
        self.n_filter = configs['training']['n_filter']
        self.n_classes = configs['training']['n_classes']
        self.latent_dim_seq = [configs['training']['latent_dim_seq']]
        self.beta = configs['training']['beta'] # equivalent to beta=5000 in the old setup

        # Regularizer for loss penalty
        # Jet features loss weighting
        gamma = configs['training']['gamma']
        gamma_1 = configs['training']['gamma_1']
        gamma_2 = configs['training']['gamma_2']
        #gamma_2 = 1.0
        n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

        # Particle features loss weighting
        alpha = configs['training']['alpha']

        # Probability to keep a node in the dropout layer
        drop_prob = configs['training']['drop_prob']

        seed = configs['training']['seed']

        set_seed(seed)

        self.conv1 = nn.Conv2d(1, 1 * self.n_filter, kernel_size=(self.num_features,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(1 * self.n_filter, 2 * self.n_filter, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(2 * self.n_filter, 4 * self.n_filter, kernel_size=(1,5), stride=(1), padding=(0))
        #
        self.fc1 = nn.Linear(1 * int(self.num_particles - 12) * 4 * self.n_filter, 1500)
        self.fc2 = nn.Linear(1500, 2 * self.latent_dim)
        #
        self.fc3 = nn.Linear(self.latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * int(self.num_particles - 12) * 4*self.n_filter)
        self.conv4 = nn.ConvTranspose2d(4 * self.n_filter, 2 * self.n_filter, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(2 * self.n_filter, 1 * self.n_filter, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(1 * self.n_filter, 1, kernel_size=(self.num_features,5), stride=(1), padding=(0))

        self.drop = nn.Dropout(drop_prob)

    def encode(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1) # flattening
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        mean = out[:,:self.latent_dim]
        logvar = 1e-6 + (out[:,self.latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = out.view(self.batch_size, 4 * self.n_filter, 1, int(self.num_particles - 12)) # reshaping
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.conv6(out)
        out = torch.sigmoid(out)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        # calcular KLD aqui em vez da compute_loss
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out

# Jet observables manual calculation
def jet_p(p_part): # input should be of shape[batch_size, features, Nparticles]
    pjet = torch.sum(p_part, dim=2).cuda() # [batch, features (px, py, pz)]
    return pjet

def jet_Energy (p_part): # input should be of shape [batch_size, features, Nparticles]
    E_particles = torch.sqrt(torch.sum(p_part*p_part, dim=1)) # E per particle shape: [100, 30]
    E_jet = torch.sum(E_particles, dim=1).cuda() # Energy per jet [100]
    return E_jet

def jet_mass (p_part):
    jet_e = jet_Energy(p_part)
    P_jet = jet_p(p_part)
    m_jet = torch.sqrt(jet_e*jet_e - (P_jet[:,0]*P_jet[:,0]) - (P_jet[:,1]*P_jet[:,1]) - (P_jet[:,2]*P_jet[:,2])).cuda()
    return m_jet # mass per jet [100]

def jet_pT(p_part):# input should be of shape[batch_size, features, Nparticles]
    p_jet = jet_p(p_part) # [100, 3]
    jet_px = p_jet[:, 0]  # [100]
    jet_py = p_jet[:, 1]  # [100]
    jet_pt = torch.sqrt(jet_px*jet_px + jet_py*jet_py)
    return jet_pt

# Custom loss function VAtrE
def compute_loss(model, x):
    #Trocar model por x_decoded
    print("num_features de dentro da compute loss: " + str(model.num_features))
    tr_0_max = torch.max(train_dataset[:,0,0])
    tr_1_max = torch.max(train_dataset[:,0,1])
    tr_2_max = torch.max(train_dataset[:,0,2])

    tr_0_min = torch.min(train_dataset[:,0,0])
    tr_1_min = torch.min(train_dataset[:,0,1])
    tr_2_min = torch.min(train_dataset[:,0,2])
    mean, logvar = model.encode(x)

    #substituir
    z = model.reparameterize(mean, logvar)
    x_decoded = model.decode(z)

    x_aux = torch.clone(x)
    #fim para calc div_kl

    x_decoded_aux = torch.clone(x_decoded)

    # if norm = true
    x_aux[:,0,0] = ((x_aux[:,0,0] * (tr_0_max - tr_0_min)) + tr_0_min)#/part_px_std
    x_aux[:,0,1] = ((x_aux[:,0,1] * (tr_1_max - tr_1_min)) + tr_1_min)#/part_py_std
    x_aux[:,0,2] = ((x_aux[:,0,2] * (tr_2_max - tr_2_min)) + tr_2_min)#/part_pz_std

    x_decoded_aux[:,0,0] = ((x_decoded_aux[:,0,0] * (tr_0_max - tr_0_min)) + tr_0_min)#/part_px_std
    x_decoded_aux[:,0,1] = ((x_decoded_aux[:,0,1] * (tr_1_max - tr_1_min)) + tr_1_min)#/part_py_std
    x_decoded_aux[:,0,2] = ((x_decoded_aux[:,0,2] * (tr_2_max - tr_2_min)) + tr_2_min)#/part_pz_std
    ###

    pdist = nn.PairwiseDistance(p=2) # Euclidean distance
    x_pos = torch.zeros(batch_size,num_features,num_particles).cuda() #variaveis do config
    x_pos = x_aux[:,0,:,:].cuda() # [100, 3, 30]
    jets_pt = (jet_pT(x_pos).unsqueeze(1).cuda())#/jet_pt_std # [100, 1]
    jets_mass = (jet_mass(x_pos).unsqueeze(1).cuda())#/jet_mass_std
    x_pos = x_pos.view(batch_size, num_features, 1, num_particles)

    x_decoded_pos = torch.zeros(batch_size,num_features,num_particles).cuda()
    x_decoded_pos = x_decoded_aux[:,0,:,:].cuda()
    jets_pt_reco = (jet_pT(x_decoded_pos).unsqueeze(1).cuda())#/jet_pt_std # [100, 1]
    jets_mass_reco = (jet_mass(x_decoded_pos).unsqueeze(1).cuda())#/jet_mass_std
    x_decoded_pos = x_decoded_pos.view(batch_size, num_features, num_particles, 1)
    x_decoded_pos.repeat(1,1,1,num_particles)

    # Permutation-invariant Loss / NND / 3D Sparse Loss
    dist = torch.pow(pdist(x_pos, x_decoded_pos),2)

    # NND original version
    jet_pt_dist = torch.pow(pdist(jets_pt, jets_pt_reco),2)
    jet_mass_dist = torch.pow(pdist(jets_mass, jets_mass_reco),2)

    # For every output value, find its closest input value; for every input value, find its closest output value.
    ieo = torch.min(dist, dim = 1)  # Get min distance per row - Find the closest input to the output
    oei = torch.min(dist, dim = 2)  # Get min distance per column - Find the closest output to the input
    # Symmetrical euclidean distances
    eucl = (ieo.values + oei.values)#*x_decoded_aux[:,0,3] # [100, 30]1

    # Loss per jet (batch size)
    loss_rec_p = alpha*(torch.sum(eucl, dim=1))
    loss_rec_j = gamma*(gamma_1*(jet_pt_dist) + gamma_2*(jet_mass_dist))
    eucl = loss_rec_p + loss_rec_j  # [100]

    # Loss individual components
    loss_rec_p = torch.sum(loss_rec_p)
    loss_rec_j = torch.sum(loss_rec_j)
    jet_pt_dist = torch.sum(jet_pt_dist)
    jet_mass_dist = torch.sum(jet_mass_dist)

    # Average symmetrical euclidean distance per image
    eucl = (torch.sum(eucl) / batch_size)
    reconstruction_loss = - eucl

    # Separate particles' loss components to plot them
    KL_divergence = (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size)
    ELBO = ((1-beta)*reconstruction_loss) - (beta*KL_divergence)
    loss = - ELBO

    return loss, KL_divergence, eucl, loss_rec_p, loss_rec_j, jet_pt_dist, jet_mass_dist, x_decoded

##### Training function per batch #####
def train(model, batch_data_train, optimizer):
    """train_loss = 0.0
    train_KLD_loss = 0.0
    train_reco_loss = 0.0"""
    input_train = batch_data_train[:, :, :].cuda()

    # rodar encode decode (lembrar de ver a paradinha da KLD)

    # loss per batch
    train_loss, train_KLD_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass, output_train  = compute_loss(model, input_train)

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

    model.eval()
    with torch.no_grad():
        input_valid = batch_data_valid.cuda()
#            output_valid = model(input_valid)
        # loss per batch
        valid_loss, valid_KLD_loss, valid_reco_loss, valid_reco_loss_p, valid_reco_loss_j, valid_reco_loss_pt, valid_reco_loss_mass, output_valid = compute_loss(model, input_valid)

        return valid_loss, valid_KLD_loss, valid_reco_loss

##### Test function #####
def test_unseed_data(model, batch_data_test):
    model.eval()
    with torch.no_grad():
        input_test = batch_data_test.cuda()
#            output_test = model(input_test)
        test_loss, test_KLD_loss, test_reco_loss, loss_particle, loss_jet, jet_pt_loss, jet_mass_loss, output_test = compute_loss(model, input_test)
    return input_test, output_test, test_loss, test_KLD_loss, test_reco_loss
