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
#from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from torch.distributions import MultivariateNormal
import awkward as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
import warnings
warnings.filterwarnings("ignore")
import optuna
from optuna.trial import TrialState

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)
spdyellow = (234/255, 171/255, 0/255)

num_features = 3
num_particles = 30
latent_dim = 30
num_filters = 50
EMBEDDING_DIM = 30 # The dimension of the embeddings
CONDITIONING_SIZE = 0
BATCH_SIZE = 100 # Batch size
NF_EPOCHS = 20 # Epochs for training the normalizing flow

# Set the random seeds
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = torch.load('/data/convae_jet/datasets/train_data_pxpypz_g_min30p.pt')
test_dataset = torch.load('/data/convae_jet/datasets/test_data_pxpypz_g_min30p.pt')

tr_0_max = torch.max(train_dataset[:,0])
tr_1_max = torch.max(train_dataset[:,1])
tr_2_max = torch.max(train_dataset[:,2])

tr_0_min = torch.min(train_dataset[:,0])
tr_1_min = torch.min(train_dataset[:,1])
tr_2_min = torch.min(train_dataset[:,2])

latent_data_train = torch.load('../jets/beta1e-05latent_2000epochs_dim_values_for_training.pt').to(device).float()
latent_data_test = torch.load('../jets/beta1e-05latent_2000epochs_dim_values_for_testing.pt').to(device).float()

embs_loader = torch.utils.data.DataLoader(latent_data_train, BATCH_SIZE)

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

model_VAE = ConvNet()

model_VAE.load_state_dict(torch.load('/home/brenoorzari/ConVAE_Jet/reports/dir_generation_fulldata_30latent_50filters_2000epochs_1e-05beta_min30p_jetpt_jetmass/model_fulldata_30latent_50filters_2000epochs_1e-05beta_min30p_jetpt_jetmass.pt'))

model_VAE = model_VAE.to(device)

FLOW_N_RANGE = [5,100] # This is the one that works for 0.0001 and 0.00001
RNVP_TOPOLOGY_LAYERS = [1,7]
RNVP_TOPOLOGY_NODES = [10,400]
LEARNING_RATE = [1e-5,1e-1]

def objective(trial):

    layers = []
    emdg = []

    n_layers = trial.suggest_int("n_layers", RNVP_TOPOLOGY_LAYERS[0], RNVP_TOPOLOGY_LAYERS[1])

    initial_nodes = 0

    for i in range(n_layers): 
        out_features = trial.suggest_int("n_units_l{}".format(i),RNVP_TOPOLOGY_NODES[0], RNVP_TOPOLOGY_NODES[1])
        if(i=0): initial_nodes = out_features
        layers.append(out_features)

    layers.append(initial_nodes)

    lr = trial.suggest_float("lr",LEARNING_RATE[0],LEARNING_RATE[1],log=True)

    nflows = trial.suggest_int("lr",FLOW_N_RANGE[0],FLOW_N_RANGE[1],step=5)

    # See the file realmvp.py for the full definition
    nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=layers, flow_n=FLOW_N, batch_norm=True,
                          mask_type='odds', conditioning_size=CONDITIONING_SIZE, use_permutation=True, single_function=True)
    nf_model = nf_model.to(device)
    
    optimizer1 = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=lr, weight_decay=lr/10.0)
    
    nf_model.train()
    for epoch in range(NF_EPOCHS):
    
        for batch_idx, data in enumerate(embs_loader):
    
            emb = data
    
            emb = emb.to(device)
    
            # Get the inverse transformation and the corresponding log determinant of the Jacobian
            u, log_det = nf_model.forward(emb)
    
            # Train via maximum likelihood
            prior_logprob = nf_model.logprob(u)
            log_prob = -torch.mean(prior_logprob.sum(1) + log_det)
    
            nf_model.zero_grad()
    
            log_prob.backward()
    
            optimizer1.step()
    
        sample_n = len(test_dataset)
    
        nf_model.eval()
        with torch.no_grad():
            for j in range(1):
        
                emb, d = nf_model.sample(sample_n, return_logdet=True)
    
                out = model_VAE.decode(emb)
        
        px = test_dataset[:,0,:]
        py = test_dataset[:,1,:]
        pz = test_dataset[:,2,:]
        
        px_gen = out[:,0,0,:]
        py_gen = out[:,0,1,:]
        pz_gen = out[:,0,2,:]
        
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
            try:
                y = np.arcsinh(pz/pt)
            except (RuntimeError, TypeError, NameError):
                pass
            eta = np.nan_to_num(y)
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
    
        minp, bins = np.histogram(jets_input_data[:,0], bins=100, range = [0, 400])
        mgen = np.histogram(jets_gen_output_data[:,0], bins=100, range = [0, 400])
        
        ptinp, bins = np.histogram(jets_input_data[:,1], bins=100, range = [0, 3000])
        ptgen = np.histogram(jets_gen_output_data[:,1], bins=100, range = [0, 3000])
        
        einp, bins = np.histogram(jets_input_data[:,2], bins=100, range = [200, 4000])
        egen = np.histogram(jets_gen_output_data[:,2], bins=100, range = [200, 4000])
        
        etainp, bins = np.histogram(jets_input_data[:,3], bins=100, range = [-3, 3]) 
        etagen = np.histogram(jets_gen_output_data[:,3], bins=100, range = [-3, 3])
        
        phiinp, bins = np.histogram(jets_input_data[:,4], bins=100, range = [-3, 3]) 
        phigen = np.histogram(jets_gen_output_data[:,4], bins=100, range = [-3, 3])
        
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
        
        emdg_m = wasserstein_distance(mgen,minp)
        emdg_pt = wasserstein_distance(ptgen,ptinp)
        emdg_e = wasserstein_distance(egen,einp)
        emdg_eta = wasserstein_distance(etagen,etainp)
        emdg_phi = wasserstein_distance(phigen,phiinp)
        emdg_sum = emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi
        
        print("EMD gauss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_m, emdg_pt, emdg_e, emdg_eta, emdg_phi, emdg_sum))
   
        emdg.append(emdg_sum)

        trial.report(emdg_sum,epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.min(emdg)

study = optuna.create_study(study_name='opt_tutorial',
                           storage='sqlite:///tutorial.db',
                           load_if_exists=True,
                           direction="maximize")

study.optimize(objective, n_trials=20,timeout=None)



trial = study.best_trial

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Best trial:")
print(" Value: ", trial.value)

print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("\nBest params: ")
for key, value in trial.params.items():
    print(" ",key,"=", value)
