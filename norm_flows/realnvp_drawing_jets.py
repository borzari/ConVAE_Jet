import torch
from torch import nn
import rnvp
import numpy as np
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt
import mplhep as mhep
plt.style.use(mhep.style.CMS)
import skhep.math as hep
from functools import reduce
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
#from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from torch.distributions import MultivariateNormal
import awkward as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
import warnings
warnings.filterwarnings("ignore")
import jetnet

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)
spdyellow = (234/255, 171/255, 0/255)

num_features = 3
num_particles = 30
latent_dim = 30
num_filters = 50

EMBEDDING_DIM = 30 # The dimension of the embeddings
FLOW_N = 50
RNVP_TOPOLOGY = [700]
BATCH_SIZE = 100 # Batch size
CONDITIONING_SIZE = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################################################

# VAE model

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
        out = out.view(-1, 4*num_filters, 1, int(num_particles - 12)) # reshaping
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

################################################################################################################

train_dataset = torch.load('/data/convae_jet/datasets/train_data_pxpypz_g_min30p.pt')
train_dataset = train_dataset.view(len(train_dataset),1,3,num_particles)
test_dataset = torch.load('/data/convae_jet/datasets/test_data_pxpypz_g_min30p.pt')

tr_0_max = torch.max(train_dataset[:,0,0])
tr_1_max = torch.max(train_dataset[:,0,1])
tr_2_max = torch.max(train_dataset[:,0,2])

tr_0_min = torch.min(train_dataset[:,0,0])
tr_1_min = torch.min(train_dataset[:,0,1])
tr_2_min = torch.min(train_dataset[:,0,2])

latent_data_train = torch.load('../jets/beta0p001latent_300epochs_dim_values_for_training.pt').to(device).float()
latent_data_valid = torch.load('../jets/beta0p001latent_300epochs_dim_values_for_validation.pt').to(device).float()
latent_data_test = torch.load('../jets/beta0p001latent_300epochs_dim_values_for_testing.pt').to(device).float()

embs_loader = torch.utils.data.DataLoader(latent_data_train, BATCH_SIZE)

# See the file realmvp.py for the full definition
nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N, batch_norm=True,
                      mask_type='odds', conditioning_size=CONDITIONING_SIZE, use_permutation=True, single_function=True)

#nf_model.load_state_dict(torch.load('/home/brenoorzari/realnvp-demo-pytorch/realnvp_nfmodel_[800]hidden_75nflows.pt'))
nf_model.load_state_dict(torch.load('/home/brenoorzari/realnvp-demo-pytorch/realnvp_nfmodel_noConditioning_[700]hidden_50nflows_1e-05beta_2000epochs.pt'))

nf_model = nf_model.to(device)

model_VAE = ConvNet()

model_VAE.load_state_dict(torch.load('/home/brenoorzari/ConVAE_Jet/reports/dir_generation_fulldata_30latent_50filters_2000epochs_1e-05beta_min30p_jetpt_jetmass/model_fulldata_30latent_50filters_2000epochs_1e-05beta_min30p_jetpt_jetmass.pt'))

model_VAE = model_VAE.to(device)

all_gen_test = np.empty(shape=(0, 1, num_features, num_particles))

for i in range(1):

    #sample_n = len(test_dataset)
    sample_n = 50000

    nf_model.eval()
    with torch.no_grad():
        for j in range(1):
    
            emb, d = nf_model.sample(sample_n, return_logdet=True)
    
            out = model_VAE.decode(emb)
            all_gen_test = np.concatenate((all_gen_test,out.cpu().detach().numpy()), axis=0)
    
px = test_dataset[:,0,:]
py = test_dataset[:,1,:]
pz = test_dataset[:,2,:]

px_gen = torch.from_numpy(all_gen_test[:,0,0,:])
py_gen = torch.from_numpy(all_gen_test[:,0,1,:])
pz_gen = torch.from_numpy(all_gen_test[:,0,2,:])

def inverse_standardize_t(X, tmin, tmax):
    mean = tmin
    std = tmax
    original_X = ((X * (std - mean)) + mean)
    return original_X

px_gen_r_0 = inverse_standardize_t(px_gen, tr_0_min,tr_0_max)
py_gen_r_0 = inverse_standardize_t(py_gen, tr_1_min,tr_1_max)
pz_gen_r_0 = inverse_standardize_t(pz_gen, tr_2_min,tr_2_max)

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

inputs = torch.stack([px, py, pz], dim=1)
masked_inputs = mask_zero_padding(inputs)

gen_outputs = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
masked_gen_outputs = mask_zero_padding(gen_outputs)

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
    return np.stack((part_pt, part_eta, part_phi, part_mass), axis=2), np.stack((part_eta, part_phi, part_pt), axis=2)

hadr_input_data, jetnet_inp = ptetaphim_particles(input_data)
hadr_gen_output_data, jetnet_gen = ptetaphim_particles(gen_output_data_0)

def jet_features(jets, mask_bool=False, mask=None):
    vecs = ak.zip({
            "pt": jets[:, :, 0],
            "eta": jets[:, :, 1],
            "phi": jets[:, :, 2],
            "mass": jets[:, :, 3],
            }, with_name="PtEtaPhiMLorentzVector")

    sum_vecs = vecs.sum(axis=1)

    jf = np.stack((ak.to_numpy(sum_vecs.mass), ak.to_numpy(sum_vecs.pt), ak.to_numpy(sum_vecs.energy), ak.to_numpy(sum_vecs.eta), ak.to_numpy(sum_vecs.phi), ak.to_numpy(sum_vecs.px), ak.to_numpy(sum_vecs.py), ak.to_numpy(sum_vecs.pz)), axis=1)

    return ak.to_numpy(jf)

jets_input_data = jet_features(hadr_input_data)
jets_gen_output_data = jet_features(hadr_gen_output_data)

jetpt_inp_repeat = (torch.from_numpy(jets_input_data[:,1]).view(-1,1).repeat(1,30)).numpy()
jetpt_gen_repeat = (torch.from_numpy(jets_gen_output_data[:,1]).view(-1,1).repeat(1,30)).numpy()

jeteta_inp_repeat = (torch.from_numpy(jets_input_data[:,3]).view(-1,1).repeat(1,30)).numpy()
jeteta_gen_repeat = (torch.from_numpy(jets_gen_output_data[:,3]).view(-1,1).repeat(1,30)).numpy()

jetphi_inp_repeat = (torch.from_numpy(jets_input_data[:,4]).view(-1,1).repeat(1,30)).numpy()
jetphi_gen_repeat = (torch.from_numpy(jets_gen_output_data[:,4]).view(-1,1).repeat(1,30)).numpy()

jetnet_inp[:,:,0] = -(jetnet_inp[:,:,0]-jeteta_inp_repeat)
jetnet_gen[:,:,0] = -(jetnet_gen[:,:,0]-jeteta_gen_repeat)

jetnet_inp[:,:,1] = -(jetnet_inp[:,:,1]-jetphi_inp_repeat)
jetnet_gen[:,:,1] = -(jetnet_gen[:,:,1]-jetphi_gen_repeat)

jetnet_inp[:,:,2] = jetnet_inp[:,:,2]/jetpt_inp_repeat
jetnet_gen[:,:,2] = jetnet_gen[:,:,2]/jetpt_gen_repeat

plt.hist(jetnet_inp[:,:,0].flatten(),range=[-1.,1.],bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
plt.yscale('log')
plt.savefig("jet_releta.pdf", format="pdf", bbox_inches='tight')
plt.clf()

plt.hist(jetnet_inp[:,:,1].flatten(),range=[-0.6,0.6],bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
plt.yscale('log')
plt.savefig("jet_relphi.pdf", format="pdf", bbox_inches='tight')
plt.clf()

plt.hist(jetnet_inp[:,:,2].flatten(),range=[0.,1.],bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
plt.yscale('log')
plt.savefig("jet_relpt.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fpnd_inp = jetnet.evaluation.fpnd(jetnet_inp,jet_type='g',use_tqdm=False)
fpnd_gen = jetnet.evaluation.fpnd(jetnet_gen,jet_type='g',use_tqdm=False)
print("The FPND value for inp is: {:.1f}".format(fpnd_inp))
print("The FPND value for gen is: {:.1f}".format(fpnd_gen))

w1efp = jetnet.evaluation.w1efp(jetnet_inp,jetnet_gen)
print("The W1EFP value is: {:.4f} +/- {:.4f}".format(w1efp[0],w1efp[1]))

w1m = jetnet.evaluation.w1m(jetnet_inp,jetnet_gen)
print("The W1M value is: {:.4f} +/- {:.4f}".format(w1m[0],w1m[1]))

w1p = jetnet.evaluation.w1p(jetnet_inp,jetnet_gen)
print("The W1P value is: {:.4f} +/- {:.4f}".format(w1p[0],w1p[1]))

cov,mmd = jetnet.evaluation.cov_mmd(jetnet_inp,jetnet_gen,use_tqdm=False)
print("The COV and the MMD values are, respectively: {:.4f}, {:.4f}".format(cov,mmd))

efp_inp = jetnet.utils.efps(jetnet_inp)
efp_gen = jetnet.utils.efps(jetnet_gen)

labels = ["Jet $EFP_1$", "Jet $EFP_2$", "Jet $EFP_3$", "Jet $EFP_4$", "Jet $EFP_5$"]
names = ["jet_efp1_ratio", "jet_efp2_ratio", "jet_efp3_ratio", "jet_efp4_ratio", "jet_efp5_ratio"]
x_reco = [efp_inp[:,0], efp_inp[:,1], efp_inp[:,2], efp_inp[:,3], efp_inp[:,4]]
x_gen_out = [efp_gen[:,0], efp_gen[:,1], efp_gen[:,2], efp_gen[:,3], efp_gen[:,4]]

x_min = [0., 0., 0., 0., 0.]
x_max = [0.0015, 0.0015, 0.0015, 0.0015, 0.0015]

# Plot jet features with ratio
for i in range(0,5):
    ratio = []
    ratio_error = []
    bins_centers = []
    gs = gridspec.GridSpec(2,1)
    # Define figure
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(gs[0])
    n_x2err, bin_edges_x2err = np.histogram(x_reco[i], range = [x_min[i], x_max[i]], bins=100)
    n_x3err, bin_edges_x3err = np.histogram(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100)
    plt.clf()
    ax = fig.add_subplot(gs[0])
    if i==0:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[i], x_max[i]], bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100,  label="Randomly Generated", density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)
        #plt.ylim(0,0.004)
    else:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[i], x_max[i]], bins=100, density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100, density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)

    ax.set_ylabel(r'Entries (norm.)', size =20)
    # set x axis to empty values
    ax.axes.get_xaxis().set_ticks([])
    if i==0: ax.legend(prop={'size': 18}, loc='upper right')
    plt.yscale('log')
    plt.draw()
    ratio = (np.divide(n_x3err, n_x2err, where=(n_x2err!=0)))
    # Error on ratio
    ratio_error = np.sqrt(np.divide((n_x3err*n_x3err + n_x3err*n_x2err), np.power(n_x2err, 3), where=(n_x2err != 0)))
    bins_centers = 0.5*(bin_edges_x3err[1:] + bin_edges_x3err[:-1])
    ax = fig.add_subplot(gs[1])
    ax.errorbar(bins_centers, ratio, yerr=ratio_error, fmt=".", color=spdblue, capsize=2)
    ax.axhline(y=1.0, color='grey', linestyle='-')
    ax.set_ylabel(r'Gen/Input', size =20)
    ax.set_xlabel(labels[i], size =20)
    ax.set_ylim([0.0, 2])
    fig.align_labels()
    plt_ratio = 0.15
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*plt_ratio)
    plt.subplots_adjust(wspace=0, hspace=0, bottom= 0)
    fig.tight_layout(h_pad=-1.5)
    plt.draw()
    plt.savefig(str(names[i])+".pdf", format="pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

labels = ["Jet $p_{x}$ (GeV)", "Jet $p_{y}$ (GeV)", "Jet $p_{z}$ (GeV)", "Jet $p_{T}$ (GeV)", "Jet Mass (GeV)", "Jet Energy (GeV)", "Jet $\eta$", "Jet $\phi$"]
names = ["jet_px_ratio", "jet_py_ratio", "jet_pz_ratio", "jet_pt_ratio", "jet_mass_ratio", "jet_energy_ratio", "jet_eta_ratio", "jet_phi_ratio"]
x_reco = [jets_input_data[:,5], jets_input_data[:,6], jets_input_data[:,7], jets_input_data[:,1], jets_input_data[:,0], jets_input_data[:,2], jets_input_data[:,3], jets_input_data[:,4]]
x_gen_out = [jets_gen_output_data[:,5], jets_gen_output_data[:,6], jets_gen_output_data[:,7], jets_gen_output_data[:,1], jets_gen_output_data[:,0], jets_gen_output_data[:,2], jets_gen_output_data[:,3], jets_gen_output_data[:,4]]

x_min = [-2000, -2000, -3000, 0., 0., 200., -3.0, -3.0]
x_max = [2000, 2000, 3000, 3000., 400., 4000., +3.0, +3.0]

# Plot jet features with ratio
for i in range(0,8):
    ratio = []
    ratio_error = []
    bins_centers = []
    gs = gridspec.GridSpec(2,1)
    # Define figure
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(gs[0])
    n_x2err, bin_edges_x2err = np.histogram(x_reco[i], range = [x_min[i], x_max[i]], bins=100)
    n_x3err, bin_edges_x3err = np.histogram(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100)
    plt.clf()
    ax = fig.add_subplot(gs[0])
    if i==4:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[i], x_max[i]], bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100,  label="Randomly Generated", density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)
        #plt.ylim(0,0.004)
    else:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[i], x_max[i]], bins=100, density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[i], x_max[i]], bins=100, density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)

    ax.set_ylabel(r'Entries (norm.)', size =20)
    # set x axis to empty values
    ax.axes.get_xaxis().set_ticks([])
    if i==4: ax.legend(prop={'size': 18}, loc='upper right')
    plt.draw()
    ratio = (np.divide(n_x3err, n_x2err, where=(n_x2err!=0)))
    # Error on ratio
    ratio_error = np.sqrt(np.divide((n_x3err*n_x3err + n_x3err*n_x2err), np.power(n_x2err, 3), where=(n_x2err != 0)))
    bins_centers = 0.5*(bin_edges_x3err[1:] + bin_edges_x3err[:-1])
    ax = fig.add_subplot(gs[1])
    ax.errorbar(bins_centers, ratio, yerr=ratio_error, fmt=".", color=spdblue, capsize=2)
    ax.axhline(y=1.0, color='grey', linestyle='-')
    ax.set_ylabel(r'Gen/Input', size =20)
    ax.set_xlabel(labels[i], size =20)
    ax.set_ylim([0.0, 2])
    fig.align_labels()
    plt_ratio = 0.15
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*plt_ratio)
    plt.subplots_adjust(wspace=0, hspace=0, bottom= 0)
    fig.tight_layout(h_pad=-1.5)
    plt.draw()
    plt.savefig(str(names[i])+".pdf", format="pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

########## Compute pt flows / check substructure modeling
# We need the particle pT, eta & phi, as well as the jet quantities
# Particles

# Define rings and ΔR
n_rings = 4
DR_0 = 0.0
DR_1 = 0.1
DR_2 = 0.15
DR_3 = 0.3
DR_4 = 0.8

def heaviside_negative_n_to_zero(data): # equivalent to torch.heaviside(tensor, 1)
    # Sets numbers <0 to 0 and everything else to 1.
    binary_tensor = torch.where(data < torch.zeros_like(data), torch.zeros_like(data), torch.ones_like(data))
    return binary_tensor

def heaviside_negativeOrzero_n_to_zero(data): # equivalent to torch.heaviside(tensor, 0)
    # Sets numbers <=0 to 0 and everything else to 1.
    binary_tensor = torch.where(data <= torch.zeros_like(data), torch.zeros_like(data), torch.ones_like(data))
    return binary_tensor

def compute_pt_flow (particle_ptetaphi, jet_ptetaphi, DR_lower_bound, DR_upper_bound):
    DR = torch.sqrt((particle_ptetaphi[:, :, 1] - jet_ptetaphi[:, 3].unsqueeze(1))*(particle_ptetaphi[:, :, 1] - jet_ptetaphi[:, 3].unsqueeze(1)) +
            (particle_ptetaphi[:, :, 2] - jet_ptetaphi[:, 4].unsqueeze(1))*(particle_ptetaphi[:, :, 2] - jet_ptetaphi[:, 4].unsqueeze(1)))
    pt_flow = torch.sum(particle_ptetaphi[:, :, 0]*heaviside_negative_n_to_zero(DR - DR_lower_bound)*(1-heaviside_negativeOrzero_n_to_zero(DR_upper_bound - DR)), dim=1)/jet_ptetaphi[:,1]
    return pt_flow

jet_ptetaphi_gen = torch.from_numpy(jets_gen_output_data).float()
jet_ptetaphi_inp = torch.from_numpy(jets_input_data).float()

p_ptetaphi_inp = torch.from_numpy(hadr_input_data).float()
p_ptetaphi_gen = torch.from_numpy(hadr_gen_output_data).float()

# Compute flows for DL & Reco
pt_flow_gen_1 = compute_pt_flow(p_ptetaphi_gen, jet_ptetaphi_gen, DR_0, DR_1).numpy()
pt_flow_gen_2 = compute_pt_flow(p_ptetaphi_gen, jet_ptetaphi_gen, DR_1, DR_2).numpy()
pt_flow_gen_3 = compute_pt_flow(p_ptetaphi_gen, jet_ptetaphi_gen, DR_2, DR_3).numpy()
pt_flow_gen_4 = compute_pt_flow(p_ptetaphi_gen, jet_ptetaphi_gen, DR_3, DR_4).numpy()

pt_flow_inp_1 = compute_pt_flow(p_ptetaphi_inp, jet_ptetaphi_inp, DR_0, DR_1).numpy()
pt_flow_inp_2 = compute_pt_flow(p_ptetaphi_inp, jet_ptetaphi_inp, DR_1, DR_2).numpy()
pt_flow_inp_3 = compute_pt_flow(p_ptetaphi_inp, jet_ptetaphi_inp, DR_2, DR_3).numpy()
pt_flow_inp_4 = compute_pt_flow(p_ptetaphi_inp, jet_ptetaphi_inp, DR_3, DR_4).numpy()

labels = ["$p_{T} \ Flow_1$", "$p_{T} \ Flow_2$", "$p_{T} \ Flow_3$", "$p_{T} \ Flow_4$"]
names = ["jet_pt_flow1_ratio", "jet_pt_flow2_ratio", "jet_pt_flow3_ratio", "jet_pt_flow4_ratio"]
x_reco = [pt_flow_inp_1,pt_flow_inp_2,pt_flow_inp_3,pt_flow_inp_4]
x_gen_out = [pt_flow_gen_1,pt_flow_gen_2,pt_flow_gen_3,pt_flow_gen_4]

x_min = [0.]
x_max = [1.]

# Plot jet features with ratio
for i in range(0,4):
    ratio = []
    ratio_error = []
    bins_centers = []
    gs = gridspec.GridSpec(2,1)
    # Define figure
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(gs[0])
    n_x2err, bin_edges_x2err = np.histogram(x_reco[i], range = [x_min[0], x_max[0]], bins=100)
    n_x3err, bin_edges_x3err = np.histogram(x_gen_out[i], range = [x_min[0], x_max[0]], bins=100)
    plt.clf()
    ax = fig.add_subplot(gs[0])
    if i==0:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[0], x_max[0]], bins=100, label="Input Test", density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[0], x_max[0]], bins=100,  label="Randomly Generated", density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)
    else:
        n_x2, bin_edges_x2, patches_x2 = ax.hist(x_reco[i], range = [x_min[0], x_max[0]], bins=100, density=True, color = spdred, histtype='step', fill=False, linewidth=1.5)
        n_x3, bin_edges_x3, patches_x3 = ax.hist(x_gen_out[i], range = [x_min[0], x_max[0]], bins=100, density=True, color = spdblue, histtype='step', fill=False, linewidth=1.5)

    ax.set_ylabel(r'Entries (norm.)', size =20)
    # set x axis to empty values
    ax.axes.get_xaxis().set_ticks([])
    if i==0: ax.legend(prop={'size': 18}, loc='upper right')
    plt.yscale('log')
    plt.draw()
    ratio = (np.divide(n_x3err, n_x2err, where=(n_x2err!=0)))
    # Error on ratio
    ratio_error = np.sqrt(np.divide((n_x3err*n_x3err + n_x3err*n_x2err), np.power(n_x2err, 3), where=(n_x2err != 0)))
    bins_centers = 0.5*(bin_edges_x3err[1:] + bin_edges_x3err[:-1])
    ax = fig.add_subplot(gs[1])
    ax.errorbar(bins_centers, ratio, yerr=ratio_error, fmt=".", color=spdblue, capsize=2)
    ax.axhline(y=1.0, color='grey', linestyle='-')
    ax.set_ylabel(r'Gen/Input', size =20)
    ax.set_xlabel(labels[i], size =20)
    ax.set_ylim([0.0, 2])
    fig.align_labels()
    plt_ratio = 0.15
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*plt_ratio)
    plt.subplots_adjust(wspace=0, hspace=0, bottom= 0)
    fig.tight_layout(h_pad=-1.5)
    plt.draw()
    plt.savefig(str(names[i])+".pdf", format="pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

minp = np.histogram(jets_input_data[:,0], bins=100, range = [0, 400])[0]
mgen = np.histogram(jets_gen_output_data[:,0], bins=100, range = [0, 400])[0]

ptinp = np.histogram(jets_input_data[:,1], bins=100, range = [0, 3000])[0]
ptgen = np.histogram(jets_gen_output_data[:,1], bins=100, range = [0, 3000])[0]

einp = np.histogram(jets_input_data[:,2], bins=100, range = [200, 4000])[0]
egen = np.histogram(jets_gen_output_data[:,2], bins=100, range = [200, 4000])[0]

etainp = np.histogram(jets_input_data[:,3], bins=100, range = [-3, 3])[0]
etagen = np.histogram(jets_gen_output_data[:,3], bins=100, range = [-3, 3])[0]

phiinp = np.histogram(jets_input_data[:,4], bins=100, range = [-3, 3])[0]
phigen = np.histogram(jets_gen_output_data[:,4], bins=100, range = [-3, 3])[0]

minp = (minp/minp.sum()) + 0.000000000001
ptinp = (ptinp/ptinp.sum()) + 0.000000000001
einp = (einp/einp.sum()) + 0.000000000001
etainp = (etainp/etainp.sum()) + 0.000000000001
phiinp = (phiinp/phiinp.sum()) + 0.000000000001

mgen = (mgen/mgen.sum()) + 0.000000000001
ptgen = (ptgen/ptgen.sum()) + 0.000000000001
egen = (egen/egen.sum()) + 0.000000000001
etagen = (etagen/etagen.sum()) + 0.000000000001
phigen = (phigen/phigen.sum()) + 0.000000000001

emdg_m = wasserstein_distance(mgen,minp)
emdg_pt = wasserstein_distance(ptgen,ptinp)
emdg_e = wasserstein_distance(egen,einp)
emdg_eta = wasserstein_distance(etagen,etainp)
emdg_phi = wasserstein_distance(phigen,phiinp)
emdg_sum = emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi

print("EMD gauss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_m, emdg_pt, emdg_e, emdg_eta, emdg_phi, emdg_sum))
