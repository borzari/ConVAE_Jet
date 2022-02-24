#########################################################################################
# VAE 3D Sparse loss - trained on JEDI-net gluons' dataset
#########################################################################################
import torch
import torch.nn as nn
from typing import Callable
from typing import Tuple
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
import time
import numpy as np
import mplhep as mhep
plt.style.use(mhep.style.CMS)
import skhep.math as hep
from functools import reduce
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance
from torch.distributions import MultivariateNormal
from torch.autograd import Variable
import awkward as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)
spdyellow = (234/255, 171/255, 0/255)

num_features = 3
num_particles = 30
latent_dim = 30
batch_size = 100
num_filters = 50
num_flows = 40
act_func = "linear"
beta = '998'
num_householder = 4
num_ortho_vecs = 10

train_dataset = torch.load('/data/convae_jet/datasets/train_data_pxpypz_g_min30p.pt')
test_dataset = torch.load('/data/convae_jet/datasets/test_data_pxpypz_g_min30p.pt')
latent_data = torch.load('../beta0p'+beta+'latent_1500epochs_dim_values_for_training.pt')
latent_data=latent_data.type(torch.FloatTensor)

#train_dataset = train_dataset[:int((len(train_dataset))/10),:,:]
#test_dataset = test_dataset[:int((len(test_dataset))/10),:,:]

train_dataset = train_dataset.view(len(train_dataset),1,3,num_particles)
test_dataset = test_dataset.view(len(test_dataset),1,3,num_particles)
latent_data = latent_data[:len(test_dataset)]

ygauss = torch.randn(size=(int(1*len(latent_data[:,0])),latent_dim)).cuda()
gauss_loader = DataLoader(dataset=ygauss, batch_size=batch_size, shuffle=True)

tr_0_max = torch.max(train_dataset[:,0,0])
tr_1_max = torch.max(train_dataset[:,0,1])
tr_2_max = torch.max(train_dataset[:,0,2])

tr_0_min = torch.min(train_dataset[:,0,0])
tr_1_min = torch.min(train_dataset[:,0,1])
tr_2_min = torch.min(train_dataset[:,0,2])

train_dataset[:,0,0] = (train_dataset[:,0,0] - tr_0_min)/(tr_0_max - tr_0_min)
train_dataset[:,0,1] = (train_dataset[:,0,1] - tr_1_min)/(tr_1_max - tr_1_min)
train_dataset[:,0,2] = (train_dataset[:,0,2] - tr_2_min)/(tr_2_max - tr_2_min)

test_dataset[:,0,0] = (test_dataset[:,0,0] - tr_0_min)/(tr_0_max - tr_0_min)
test_dataset[:,0,1] = (test_dataset[:,0,1] - tr_1_min)/(tr_1_max - tr_1_min)
test_dataset[:,0,2] = (test_dataset[:,0,2] - tr_2_min)/(tr_2_max - tr_2_min)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
latent_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=True)

gen_dataset = torch.zeros([test_dataset.shape[0], 1, num_features, num_particles])

####################################### DEFINE MODEL ####################################
# # Define models' architecture & helper functions
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 1*num_filters, kernel_size=(num_features,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(1*num_filters, 2*num_filters, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(2*num_filters, 4*num_filters, kernel_size=(1,5), stride=(1), padding=(0))
        #
        self.fc1 = nn.Linear(1 * int(num_particles - 12) * 4*num_filters, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        #
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 4*num_filters)
        self.conv4 = nn.ConvTranspose2d(4*num_filters, 2*num_filters, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(2*num_filters, 1*num_filters, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(1*num_filters, 1, kernel_size=(num_features,5), stride=(1), padding=(0))

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
        mean = out[:,:latent_dim]
        logvar = 1e-6 + (out[:,latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = out.view(batch_size, 4*num_filters, 1, int(num_particles - 12)) # reshaping
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
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out

# Define Sylvester flows
class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        # The repeats are used to allow any number of batch size
        qr2 = qr2.repeat(batch_size,1,1)
        qr1 = qr1.repeat(batch_size,1,1)
        b = b.repeat(batch_size,1,1)

        # The calculation is z' = z + qr1T.b + z.qr2.qr1T
        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(r2qzb, qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        # The function h is set as linear for the inversion
        diag_j = diag_r1 * diag_r2
        diag_j = torch.bmm(torch.ones(100, 1, 30).cuda(), qr2).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)

class HouseholderSylvester(nn.Module):
    """
    Variational auto-encoder with householder sylvester flows in the decoder.
    """

    def __init__(self, num_householder):
        super().__init__()

        ## Initialize log-det-jacobian to zero
        self.log_det_j = 0

        # Flow parameters
        flow = Sylvester
        self.num_flows_func = num_flows
        self.num_householder = num_householder
        assert self.num_householder > 0

        identity = torch.eye(latent_dim, latent_dim)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(latent_dim, latent_dim), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, latent_dim).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()
        
        self.amor_d = nn.Linear(latent_dim, self.num_flows_func * latent_dim * latent_dim)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(latent_dim, self.num_flows_func * latent_dim),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(latent_dim, self.num_flows_func * latent_dim),
            self.diag_activation
        )

        self.amor_q = nn.Linear(latent_dim, self.num_flows_func * latent_dim * self.num_householder)

        self.amor_b = nn.Linear(latent_dim, self.num_flows_func * latent_dim)

        # Parameters are set from "zero", and can fit any number of batches

        self.param_q = nn.Parameter(torch.randn(1,self.num_flows_func*latent_dim*self.num_householder).normal_(0, 0.1))
        self.param_b = nn.Parameter(torch.randn(1,1,latent_dim,self.num_flows_func).normal_(0, 0.1))
        self.param_d = nn.Parameter(torch.randn(1,latent_dim,latent_dim,self.num_flows_func).normal_(0, 0.1))
        self.param_d1 = torch.tanh(nn.Parameter(torch.randn(1,latent_dim,self.num_flows_func).normal_(0, 0.1)))
        self.param_d2 = torch.tanh(nn.Parameter(torch.randn(1,latent_dim,self.num_flows_func).normal_(0, 0.1)))

        # Normalizing flow layers
        for k in range(self.num_flows_func):
            flow_k = flow(latent_dim)

            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):

        # Reshape to shape (num_flows * batch_size * num_householder, z_size)
        q = q.view(-1, latent_dim)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        v = torch.div(q, norm)

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))

        amat = self._eye - 2 * vvT

        # Reshaping: first dimension is batch_size * num_flows
        amat = amat.view(-1, self.num_householder, latent_dim, latent_dim)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_flows_func, latent_dim, latent_dim)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, z):

        full_d = self.param_d

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = self.param_d1.cuda()
        r2[:, self.diag_idx, self.diag_idx, :] = self.param_d2.cuda()

        q = self.param_q
        b = self.param_b

        return r1, r2, q, b

    def forward(self, z):
        self.log_det_j = 0

        r1, r2, q, b = self.encode(z)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [z]

        # Lists to save transformation parameters
        r1l = []
        r2l = []
        qorthol = []
        bl = []

        # Normalizing flows
        for k in range(self.num_flows_func):

            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_k, b[:, :, :, k], sum_ldj=True)

            r1l.append(r1[:, :, :, k])
            r2l.append(r2[:, :, :, k])
            qorthol.append(q_k)
            bl.append(b[:, :, :, k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        return self.log_det_j, z[0], z[-1], r1l, r2l, qorthol, bl

# Initialize model and load it on GPU
model = ConvNet()
model.load_state_dict(torch.load('/home/brenoorzari/ConVAE_Jet/reports/dir_generation_fulldata_30latent_50filters_1500epochs_0p'+beta+'beta_min30p_jetpt_jetmass/model_fulldata_30latent_50filters_1500epochs_0p'+beta+'beta_min30p_jetpt_jetmass.pt'))
model = model.cuda()

model2 = HouseholderSylvester(num_householder)
model2.load_state_dict(torch.load('serio_copy_best_validated_model_sylvesterflow_nflows40_linear_beta0p998.pt'))
model2 = model2.cuda()

gen_output = np.empty(shape=(0, 1, num_features, num_particles))
gen_outputv = np.empty(shape=(0, 1, num_features, num_particles))

for g, (gauss_data) in enumerate(gauss_loader): # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES
#for g, (jets) in enumerate(latent_loader): # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES

    outyl = []
    if g == (len(gauss_loader) - 1): # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES
#    if g == (len(latent_loader) - 1): # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES
        break

    jets = latent_data[int(100*g):int(100*(g+1))].cuda()
    log_det_j, y0, y, r1, r2, q, b = model2(jets)

    # Repeating the transformation of the network for validation
    for k in range(len(b)):

        qr2 = torch.bmm(q[k], r2[k].transpose(2, 1))
        qr1 = torch.bmm(q[k], r1[k])

        qr2 = qr2.repeat(batch_size,1,1)
        qr1 = qr1.repeat(batch_size,1,1)

        b[k] = b[k].repeat(batch_size,1,1)

        jets = jets.view(batch_size,1,latent_dim)

        if k == 0:
            r2qzb = torch.bmm(jets, qr2) + b[k]
            outy = jets + torch.bmm(r2qzb, qr1.transpose(2, 1))
            outyl.append(outy)
        else:
            r2qzb = torch.bmm(outy, qr2) + b[k]
            outy = outy + torch.bmm(r2qzb, qr1.transpose(2, 1))
            outyl.append(outy.view(batch_size,latent_dim))

        jets = jets.view(batch_size,latent_dim)
        b[k] = b[k][0].view(1,1,latent_dim)

    k = 0

    # Performing the inversion of the flow; random gaussian data and the latent vector values can be set here
    for k in range(len(b)):

        qr2 = torch.bmm(q[num_flows-k-1], r2[num_flows-k-1].transpose(2, 1))
        qr1 = torch.bmm(q[num_flows-k-1], r1[num_flows-k-1])

        qr2 = qr2.repeat(batch_size,1,1)
        qr1 = qr1.repeat(batch_size,1,1)

        b[num_flows-k-1] = b[num_flows-k-1].repeat(batch_size,1,1)

        # The calculation is z' = z + qr1T.b + z.qr2.qr1T
        # The inversion is then z' - qr1T.b = z(I + qr2.qr1T) -> z = (z' - qr1T.b).(I + qr2.qr1T)^-1

        invaux = torch.inverse(torch.bmm(qr2,qr1.transpose(2,1))+torch.eye(latent_dim).cuda()) # This is (I + qr2.qr1T)^-1

        invaux2 = torch.bmm(b[num_flows-k-1],qr1.transpose(2,1)) # This is qr1T.b
        invaux3 = gauss_data.view(batch_size,1,latent_dim)-invaux2 # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES; it is (z' - qr1T.b) if z' are random gaussian values
#        invaux3 = y.view(batch_size,1,latent_dim)-invaux2 # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES; it is (z' - qr1T.b) if z' are transformed latent vector values
        invy = torch.bmm(invaux3,invaux)
        invy = invy.view(batch_size,latent_dim)
        b[num_flows-k-1] = b[num_flows-k-1][0].view(1,1,latent_dim)

        gauss_data = invy # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES
#        y = invy # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES

    # generation
#    z = jets # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES
#    generated_output = model.decode(y) # THIS CONTROL THE GENERATION STARTING FROM KNOWN LATENT DIM VALUES
    z = torch.randn(batch_size, latent_dim).cuda() # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES
    generated_output = model.decode(gauss_data) # THIS CONTROL THE RANDOM GENERATION STARTING FROM RANDOM GAUSSIAN VALUES
    generated_outputv = model.decode(z) # THIS IS FOR COMPARING THE GENERATION OF VAE WITH VAE+NF

    batch_gen_output = generated_output.cpu().detach().numpy()
    gen_output = np.concatenate((gen_output, batch_gen_output), axis=0)

    batch_gen_outputv = generated_outputv.cpu().detach().numpy()
    gen_outputv = np.concatenate((gen_outputv, batch_gen_outputv), axis=0)

# From now on is the same as in the VAE for producing the jets histograms

px = test_dataset[:,0,0,:]
py = test_dataset[:,0,1,:]
pz = test_dataset[:,0,2,:]

px_gen = gen_output[:,0,0,:]
py_gen = gen_output[:,0,1,:]
pz_gen = gen_output[:,0,2,:]

px_gen_0 = torch.from_numpy(px_gen)
py_gen_0 = torch.from_numpy(py_gen)
pz_gen_0 = torch.from_numpy(pz_gen)

px_genv = gen_outputv[:,0,0,:]
py_genv = gen_outputv[:,0,1,:]
pz_genv = gen_outputv[:,0,2,:]

px_gen_0v = torch.from_numpy(px_genv)
py_gen_0v = torch.from_numpy(py_genv)
pz_gen_0v = torch.from_numpy(pz_genv)

def inverse_standardize_t(X, tmin, tmax):
    mean = tmin
    std = tmax
    original_X = ((X * (std - mean)) + mean)
    return original_X

px_r_0 = inverse_standardize_t(px,tr_0_min,tr_0_max)
py_r_0 = inverse_standardize_t(py,tr_1_min,tr_1_max)
pz_r_0 = inverse_standardize_t(pz,tr_2_min,tr_2_max)

px_gen_r_0 = inverse_standardize_t(px_gen_0, tr_0_min,tr_0_max)
py_gen_r_0 = inverse_standardize_t(py_gen_0, tr_1_min,tr_1_max)
pz_gen_r_0 = inverse_standardize_t(pz_gen_0, tr_2_min,tr_2_max)

px_gen_r_0v = inverse_standardize_t(px_gen_0v, tr_0_min,tr_0_max)
py_gen_r_0v = inverse_standardize_t(py_gen_0v, tr_1_min,tr_1_max)
pz_gen_r_0v = inverse_standardize_t(pz_gen_0v, tr_2_min,tr_2_max)

def mask_zero_padding(input_data):
    # Mask input for zero-padded particles. Set to zero values between -10^-8 and 10^-8
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

inputs = torch.stack([px_r_0, py_r_0, pz_r_0], dim=1)
masked_inputs = mask_zero_padding(inputs)

gen_outputs = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
masked_gen_outputs = mask_zero_padding(gen_outputs)

gen_outputsv = torch.stack([px_gen_r_0v, py_gen_r_0v, pz_gen_r_0v], dim=1)
masked_gen_outputsv = mask_zero_padding(gen_outputsv)

######################################################################################
# Output constraints
def mask_min_pt(output_data):
    # Mask output for min-pt
    min_pt_cut = 0.25
    mask =  output_data[:,0,:] * output_data[:,0,:] + output_data[:,1,:] * output_data[:,1,:] > min_pt_cut**2
    # Expand over the features' dimension
    mask = mask.unsqueeze(1)
    # Then, you can apply the mask
    data_masked = mask * output_data
    return data_masked

# Gen data
masked_gen_outputs_0 = mask_min_pt(masked_gen_outputs)
masked_gen_outputs_0v = mask_min_pt(masked_gen_outputsv)

px_r_masked = masked_inputs[:,0,:].detach().cpu().numpy()
py_r_masked = masked_inputs[:,1,:].detach().cpu().numpy()
pz_r_masked = masked_inputs[:,2,:].detach().cpu().numpy()
mass = np.zeros((pz_r_masked.shape[0], num_particles))

input_data = np.stack((px_r_masked, py_r_masked, pz_r_masked, mass), axis=2)

# Gen data
px_gen_r_0_masked = masked_gen_outputs_0[:,0,:].detach().cpu().numpy()
py_gen_r_0_masked = masked_gen_outputs_0[:,1,:].detach().cpu().numpy()
pz_gen_r_0_masked = masked_gen_outputs_0[:,2,:].detach().cpu().numpy()
mass_gen_0 = np.zeros((pz_gen_r_0_masked.shape[0], num_particles))

# Gen data
gen_output_data_0 = np.stack((px_gen_r_0_masked, py_gen_r_0_masked, pz_gen_r_0_masked, mass_gen_0), axis=2)

# Gen data
px_gen_r_0_maskedv = masked_gen_outputs_0v[:,0,:].detach().cpu().numpy()
py_gen_r_0_maskedv = masked_gen_outputs_0v[:,1,:].detach().cpu().numpy()
pz_gen_r_0_maskedv = masked_gen_outputs_0v[:,2,:].detach().cpu().numpy()
mass_gen_0v = np.zeros((pz_gen_r_0_maskedv.shape[0], num_particles))

# Gen data
gen_output_data_0v = np.stack((px_gen_r_0_maskedv, py_gen_r_0_maskedv, pz_gen_r_0_maskedv, mass_gen_0v), axis=2)

def compute_eta(pz, pt):
    eta = np.nan_to_num(np.arcsinh(pz/pt))
    return eta
def compute_phi(px, py):
    phi = np.arctan2(py, px)
    return phi
def particle_pT(p_part): # input of shape [n_jets, 3_features, n_particles]
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

hadr_input_data = ptetaphim_particles(input_data)
hadr_gen_output_data = ptetaphim_particles(gen_output_data_0)
hadr_gen_output_datav = ptetaphim_particles(gen_output_data_0v)

input_cart = np.empty(shape=(0,3))
input_hadr = np.empty(shape=(0,3))
gen_cart = np.empty(shape=(0,3))
gen_hadr = np.empty(shape=(0,3))
gen_cartv = np.empty(shape=(0,3))
gen_hadrv = np.empty(shape=(0,3))

print(input_data.shape,hadr_input_data.shape,gen_output_data_0.shape,hadr_gen_output_data.shape,gen_output_data_0v.shape,hadr_gen_output_datav.shape)

for d in range(len(gen_output_data_0)):
    input_cart = np.concatenate((input_cart,input_data[d,:,:3]),axis=0)
    input_hadr = np.concatenate((input_hadr,hadr_input_data[d,:,:3]),axis=0)
    gen_cart = np.concatenate((gen_cart,gen_output_data_0[d,:,:3]),axis=0)
    gen_hadr = np.concatenate((gen_hadr,hadr_gen_output_data[d,:,:3]),axis=0)
    gen_cartv = np.concatenate((gen_cartv,gen_output_data_0v[d,:,:3]),axis=0)
    gen_hadrv = np.concatenate((gen_hadrv,hadr_gen_output_datav[d,:,:3]),axis=0)

inppx, bins, _ = plt.hist(input_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genpx = plt.hist(gen_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genpxv = plt.hist(gen_cartv[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $p_{x}$ (GeV)')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_part_gen_px_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_px_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part px")

inppy, bins, _ = plt.hist(input_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genpy = plt.hist(gen_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genpyv = plt.hist(gen_cartv[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $p_{y}$ (GeV)')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_part_gen_py_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_py_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part py")

inppz, bins, _ = plt.hist(input_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genpz = plt.hist(gen_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genpzv = plt.hist(gen_cartv[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $p_{z}$ (GeV)')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_part_gen_pz_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_pz_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part pz")

inppt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 300], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 300], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genptv = plt.hist(gen_hadrv[:,0], bins=100, range = [0, 300], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $p_{T}$ (GeV)')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_part_gen_pt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_pt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part pt")

inplowpt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genlowpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genlowptv = plt.hist(gen_hadrv[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $p_{T}$ (GeV)')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_part_gen_lowpt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_lowpt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part lowpt")

inpeta, bins, _ = plt.hist(input_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
geneta = plt.hist(gen_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genetav = plt.hist(gen_hadrv[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $\eta$')
plt.yscale('log')
plt.legend(loc='lower center', prop={'size': 24})
plt.savefig('test_part_gen_eta_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_eta_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part eta")

inpphi, bins, _ = plt.hist(input_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
genphi = plt.hist(gen_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
genphiv = plt.hist(gen_hadrv[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Particle $\phi$')
plt.yscale('linear')
plt.legend(loc='lower center', prop={'size': 24})
plt.savefig('test_part_gen_phi_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_part_reco_phi_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("part phi")

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

jets_input_data = jet_features(hadr_input_data)
jets_gen_output_data = jet_features(hadr_gen_output_data)
jets_gen_output_datav = jet_features(hadr_gen_output_datav)

minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
mgen = plt.hist(jets_gen_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
mgenv = plt.hist(jets_gen_output_datav[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Jet Mass (GeV)')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_jet_gen_mass_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_jet_reco_mass_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("jet mass")

ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range = [0, 3000], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
ptgen = plt.hist(jets_gen_output_data[:,1], bins=100, range = [0, 3000], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
ptgenv = plt.hist(jets_gen_output_datav[:,1], bins=100, range = [0, 3000], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Jet $p_T$ (GeV)')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_jet_gen_pt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_jet_reco_pt_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("jet pt")

einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200, 4000], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
egen = plt.hist(jets_gen_output_data[:,2], bins=100, range = [200, 4000], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
egenv = plt.hist(jets_gen_output_datav[:,2], bins=100, range = [200, 4000], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Jet Energy (GeV)')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_jet_gen_energy_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_jet_reco_energy_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("jet energy")

etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
etagen = plt.hist(jets_gen_output_data[:,3], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
etagenv = plt.hist(jets_gen_output_datav[:,3], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Jet $\eta$')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('test_jet_gen_eta_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_jet_reco_eta_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("jet eta")

phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Input', color = spdred,linewidth=1.5)
phigen = plt.hist(jets_gen_output_data[:,4], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Gen VAE+NF', color = spdblue,linewidth=1.5)
phigenv = plt.hist(jets_gen_output_datav[:,4], bins=100, range = [-3, 3], histtype = 'step', density=False, label='Gen VAE', color = spdyellow,linewidth=1.5)
plt.ylabel("Entries")
plt.xlabel('Jet $\phi$')
plt.yscale('linear')
plt.legend(loc='lower center', prop={'size': 24})
plt.savefig('test_jet_gen_phi_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('test_jet_reco_phi_GeV_presentation_sylvester.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print("jet phi")

minp = (minp/minp.sum()) + 0.000000000001
ptinp = (ptinp/ptinp.sum()) + 0.000000000001
einp = (einp/einp.sum()) + 0.000000000001
etainp = (etainp/etainp.sum()) + 0.000000000001
phiinp = (phiinp/phiinp.sum()) + 0.000000000001

mgen = (mgen[0]/mgen[0].sum()) + 0.000000000001
ptgen = (ptgen[0]/ptgen[0].sum()) + 0.000000000001
egen = (egen[0]/egen[0].sum()) + 0.000000000001
etagen = (etagen[0]/etagen[0].sum()) + 0.000000000001
phigen = (phigen[0]/phigen[0].sum()) + 0.000000000001

mgenv = (mgenv[0]/mgenv[0].sum()) + 0.000000000001
ptgenv = (ptgenv[0]/ptgenv[0].sum()) + 0.000000000001
egenv = (egenv[0]/egenv[0].sum()) + 0.000000000001
etagenv = (etagenv[0]/etagenv[0].sum()) + 0.000000000001
phigenv = (phigenv[0]/phigenv[0].sum()) + 0.000000000001

emdg_m = wasserstein_distance(mgen,minp)
emdg_pt = wasserstein_distance(ptgen,ptinp)
emdg_e = wasserstein_distance(egen,einp)
emdg_eta = wasserstein_distance(etagen,etainp)
emdg_phi = wasserstein_distance(phigen,phiinp)
emdg_sum = emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi

emdg_mv = wasserstein_distance(mgenv,minp)
emdg_ptv = wasserstein_distance(ptgenv,ptinp)
emdg_ev = wasserstein_distance(egenv,einp)
emdg_etav = wasserstein_distance(etagenv,etainp)
emdg_phiv = wasserstein_distance(phigenv,phiinp)
emdg_sumv = emdg_mv + emdg_ptv + emdg_ev + emdg_etav + emdg_phiv

print("EMD gauss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_m, emdg_pt, emdg_e, emdg_eta, emdg_phi, emdg_sum))
print("EMD gauss VAE: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_mv, emdg_ptv, emdg_ev, emdg_etav, emdg_phiv, emdg_sumv))
