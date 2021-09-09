#########################################################################################
# VAE 3D Sparse loss - trained on JEDI-net gluons' dataset
#########################################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import mplhep as mhep
plt.style.use(mhep.style.CMS)
import skhep.math as hep
import os
from functools import reduce
from matplotlib.colors import LogNorm
#from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import awkward as ak
import random
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

torch.autograd.set_detect_anomaly(True)

# Hyperparameters
# Input data specific params
num_particles = 30
jet_type = 'g'

# Training params
#N_epochs = 800
N_epochs = 1500
#N_epochs = 2000
#N_epochs = 300
#N_epochs = 11
batch_size = 100
learning_rate = 0.0001
#learning_rate = 0.001
saving_epoch = 20
#num_filter = 300
num_filter = 50

num_classes = 1
#latent_dim = 20
#latent_dim_seq = [20,30,50,75,100,150,200]
latent_dim_seq = [30]
#latent_dim_seq = [20]
#beta = 0.1
#beta = 5000.0
beta = 0.9998 # equivalent to beta=5000 in the old setup
#beta = 0.0001
#masking = True

# Model params
#model_name = '_test_generation_beta'+str(int(beta))+'_jetmassonly_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_awkwardcoffeacalc_moreepochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_zscore_moreepochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_twosteptraining_onlymassthenpt_moreepochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_zeropadloss_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_moreepochs_onlyjetmass_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_maryparamlv_mean0std1_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_onlymassthenpt_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_onlymassgeneration_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_multmassterm_1mbetatest_awkwardcoffeacalc_moreepochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_multmassterm_1mbetatest_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_withmasking_multmassterm_1mbetatest_awkwardcoffeacalc_moreepochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_withmasking_multmassterm_1mbetatest_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_withmasking_weightedsum_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_withmasking_crossentropy_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_massweight10_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluatemin'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_originaltest_fulldata_awkwardcoffeacalc_1500epochs_0p0to1p0_sparse_nnd_beta01_train_evaluatemin'+ str(num_particles) + 'p_jetpt_jetmass_'
#model_name = '_test_generation_originaltest_fulldata_awkwardcoffeacalc_moreepochs_0p0to1p0_sparse_nnd_beta0p9998_train_evaluatemin'+str(num_particles)+'p_jetpt_jetmass_'

#dir_name='dir'+model_name+'test'

# Regularizer for loss penalty
# Jet features loss weighting
gamma = 1.0
gamma_1 = 1.0
gamma_2 = 10.0
#gamma_2 = 1.0
n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

# Particle features loss weighting
alpha = 1.0

# Starting time
start_time = time.time()

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)
spdyellow = (234/255, 171/255, 0/255)

# Probability to keep a node in the dropout layer
drop_prob = 0.0

# Set patience for Early Stopping
patience = N_epochs
#patience = 20
#patience = 15

seed = 123456789
#seed = random.randint(0,(2**32)-1)
#print(seed)

####################################### LOAD DATA #######################################
train_dataset = torch.load('train_data_pxpypz_g_min30p.pt')
valid_dataset = torch.load('valid_data_pxpypz_g_min30p.pt')
test_dataset = torch.load('test_data_pxpypz_g_min30p.pt')

#train_dataset = train_dataset[:int((len(train_dataset))/10),:,:]
#valid_dataset = valid_dataset[:int((len(valid_dataset))/10),:,:]
#test_dataset = test_dataset[:int((len(test_dataset))/10),:,:]

print(train_dataset.shape,valid_dataset.shape,test_dataset.shape)

train_dataset = train_dataset.view(len(train_dataset),1,3,num_particles)
valid_dataset = valid_dataset.view(len(valid_dataset),1,3,num_particles)
test_dataset = test_dataset.view(len(test_dataset),1,3,num_particles)

train_dataset = train_dataset.cpu()
valid_dataset = valid_dataset.cpu()
test_dataset = test_dataset.cpu()

num_features = len(train_dataset[0,0])

tr_0_max = torch.max(train_dataset[:,0,0])
tr_1_max = torch.max(train_dataset[:,0,1])
tr_2_max = torch.max(train_dataset[:,0,2])

tr_0_min = torch.min(train_dataset[:,0,0])
tr_1_min = torch.min(train_dataset[:,0,1])
tr_2_min = torch.min(train_dataset[:,0,2])

train_dataset[:,0,0] = (train_dataset[:,0,0] - tr_0_min)/(tr_0_max - tr_0_min)
train_dataset[:,0,1] = (train_dataset[:,0,1] - tr_1_min)/(tr_1_max - tr_1_min)
train_dataset[:,0,2] = (train_dataset[:,0,2] - tr_2_min)/(tr_2_max - tr_2_min)

valid_dataset[:,0,0] = (valid_dataset[:,0,0] - tr_0_min)/(tr_0_max - tr_0_min)
valid_dataset[:,0,1] = (valid_dataset[:,0,1] - tr_1_min)/(tr_1_max - tr_1_min)
valid_dataset[:,0,2] = (valid_dataset[:,0,2] - tr_2_min)/(tr_2_max - tr_2_min)

test_dataset[:,0,0] = (test_dataset[:,0,0] - tr_0_min)/(tr_0_max - tr_0_min)
test_dataset[:,0,1] = (test_dataset[:,0,1] - tr_1_min)/(tr_1_max - tr_1_min)
test_dataset[:,0,2] = (test_dataset[:,0,2] - tr_2_min)/(tr_2_max - tr_2_min)

#tr_0_max = torch.std(train_dataset[:,0,0])
#tr_1_max = torch.std(train_dataset[:,0,1])
#tr_2_max = torch.std(train_dataset[:,0,2])

#tr_0_min = torch.mean(train_dataset[:,0,0])
#tr_1_min = torch.mean(train_dataset[:,0,1])
#tr_2_min = torch.mean(train_dataset[:,0,2])

#train_dataset[:,0,0] = (train_dataset[:,0,0] - tr_0_min)/(tr_0_max)
#train_dataset[:,0,1] = (train_dataset[:,0,1] - tr_1_min)/(tr_1_max)
#train_dataset[:,0,2] = (train_dataset[:,0,2] - tr_2_min)/(tr_2_max)

#valid_dataset[:,0,0] = (valid_dataset[:,0,0] - tr_0_min)/(tr_0_max)
#valid_dataset[:,0,1] = (valid_dataset[:,0,1] - tr_1_min)/(tr_1_max)
#valid_dataset[:,0,2] = (valid_dataset[:,0,2] - tr_2_min)/(tr_2_max)

#test_dataset[:,0,0] = (test_dataset[:,0,0] - tr_0_min)/(tr_0_max)
#test_dataset[:,0,1] = (test_dataset[:,0,1] - tr_1_min)/(tr_1_max)
#test_dataset[:,0,2] = (test_dataset[:,0,2] - tr_2_min)/(tr_2_max)

norm_part_px = torch.Tensor()
norm_part_py = torch.Tensor()
norm_part_pz = torch.Tensor()

gen_dataset = torch.zeros([test_dataset.shape[0], 1, num_features, num_particles])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#seq_num_filter = [16,20,32,50]
#seq_num_filter = [16,20,32,50,100,200,300]
#seq_num_filter = [300]

#for num_filter in seq_num_filter:
for latent_dim in latent_dim_seq:

    model_name = '_test_generation_originaltest_fulldata_'+str(latent_dim)+'latent_'+str(num_filter)+'filters_'+str(N_epochs)+'epochs_0p0to1p0_sparse_nnd_beta0p9998_train_evaluatemin'+str(num_particles)+'p_jetpt_jetmass_'
    
    dir_name='generation_beta0p9998_dir_second'+model_name+'test'
   
    os.system('mkdir /home/brenoorzari/jets/'+str(dir_name))

#    set_seed(seed)
    
    # Create iterable data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(dataset=gen_dataset, batch_size=batch_size, shuffle=False)
    
    output_tensor_emdt = torch.Tensor()
    output_tensor_emdg = torch.Tensor()
    
    ####################################### DEFINE MODEL ####################################
    # # Define models' architecture & helper functions
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()

            set_seed(seed)

    #        self.conv1 = nn.Conv2d(1, 16, kernel_size=(num_features,5), stride=(1), padding=(0))
    #        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0))
    #        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
            #
    #        self.fc1 = nn.Linear(1 * int(num_particles - 12) * 64, 1500)
    #        self.fc2 = nn.Linear(1500, 2 * latent_dim)
            #
    #        self.fc3 = nn.Linear(latent_dim, 1500)
    #        self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 64)
    #        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
    #        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0))
    #        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(num_features,5), stride=(1), padding=(0))
    
            self.conv1 = nn.Conv2d(1, 1*num_filter, kernel_size=(num_features,5), stride=(1), padding=(0))
            self.conv2 = nn.Conv2d(1*num_filter, 2*num_filter, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv3 = nn.Conv2d(2*num_filter, 4*num_filter, kernel_size=(1,5), stride=(1), padding=(0))
            #
            self.fc1 = nn.Linear(1 * int(num_particles - 12) * 4*num_filter, 1500)
            self.fc2 = nn.Linear(1500, 2 * latent_dim)
            #
            self.fc3 = nn.Linear(latent_dim, 1500)
            self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 4*num_filter)
            self.conv4 = nn.ConvTranspose2d(4*num_filter, 2*num_filter, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv5 = nn.ConvTranspose2d(2*num_filter, 1*num_filter, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv6 = nn.ConvTranspose2d(1*num_filter, 1, kernel_size=(num_features,5), stride=(1), padding=(0))
    
            #
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
            mean = out[:,:latent_dim]
            logvar = 1e-6 + (out[:,latent_dim:])
            return mean, logvar
    
        def decode(self, z):
    
            out = self.fc3(z)
            out = torch.relu(out)
            out = self.fc4(out)
            out = torch.relu(out)
            out = out.view(batch_size, 4*num_filter, 1, int(num_particles - 12)) # reshaping
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
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_decoded = model.decode(z)
    
        x_aux = torch.clone(x)
        x_decoded_aux = torch.clone(x_decoded)
    
        x_aux[:,0,0] = ((x_aux[:,0,0] * (tr_0_max - tr_0_min)) + tr_0_min)#/part_px_std
        x_aux[:,0,1] = ((x_aux[:,0,1] * (tr_1_max - tr_1_min)) + tr_1_min)#/part_py_std
        x_aux[:,0,2] = ((x_aux[:,0,2] * (tr_2_max - tr_2_min)) + tr_2_min)#/part_pz_std
    
        x_decoded_aux[:,0,0] = ((x_decoded_aux[:,0,0] * (tr_0_max - tr_0_min)) + tr_0_min)#/part_px_std
        x_decoded_aux[:,0,1] = ((x_decoded_aux[:,0,1] * (tr_1_max - tr_1_min)) + tr_1_min)#/part_py_std
        x_decoded_aux[:,0,2] = ((x_decoded_aux[:,0,2] * (tr_2_max - tr_2_min)) + tr_2_min)#/part_pz_std
    
        pdist = nn.PairwiseDistance(p=2) # Euclidean distance
        x_pos = torch.zeros(batch_size,num_features,num_particles).cuda()
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
    
        # Relative NND loss
    #    jet_pt_dist = torch.pow(pdist(jets_pt, jets_pt_reco),2)/(jets_pt*jets_pt) # [100] pt MSE on inp-outp
    #    jet_mass_dist = torch.pow(pdist(jets_mass, jets_mass_reco),2)/(jets_mass*jets_mass)  # [100] jet mass MSE on inp-outp
    
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
    #    ELBO = (reconstruction_loss) - (beta*KL_divergence)
        loss = - ELBO
    
#        print(-((1-beta)*reconstruction_loss),(beta*KL_divergence))
    
        return loss, KL_divergence, eucl, loss_rec_p, loss_rec_j, jet_pt_dist, jet_mass_dist, x_decoded
    
    ##### Training function per batch #####
    def train(model, batch_data_train, optimizer):
        """train_loss = 0.0
        train_KLD_loss = 0.0
        train_reco_loss = 0.0"""
        input_train = batch_data_train[:, :, :].cuda()
    #    output_train = model(input_train)
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
    
    ####################################### TRAINING #######################################
    # Initialize model and load it on GPU
    model = ConvNet()
    model = model.cuda()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    all_input = np.empty(shape=(0, 1, num_features, num_particles))
    all_output = np.empty(shape=(0, 1, num_features, num_particles))
    
    x_graph = []
    y_graph = []
    
    tr_y_rec = []
    tr_y_kl = []
    tr_y_loss = []
    
    # Individual loss components
    tr_y_loss_p = []
    tr_y_loss_j = []
    tr_y_loss_pt = []
    tr_y_loss_mass = []
    
    val_y_rec = []
    val_y_kl = []
    val_y_loss = []
    
    emdt_epoch = []
    emdg_epoch = []
    fifty_epoch = []
    
    min_loss, stale_epochs, min_emdt, min_emdg = 999999.0, 0, 9999999.0, 9999999.0
    
    for epoch in range(N_epochs):
    
        n=n+1
    
        x_graph.append(epoch)
        y_graph.append(epoch)
    
        tr_loss_aux = 0.0
        tr_kl_aux = 0.0
        tr_rec_aux = 0.0
        # Individual loss components
        tr_rec_p_aux = 0.0
        tr_rec_j_aux = 0.0
        tr_rec_pt_aux = 0.0
        tr_rec_mass_aux = 0.0
    
        val_loss_aux = 0.0
        val_kl_aux = 0.0
        val_rec_aux = 0.0
    
        for y, (jets_train) in enumerate(train_loader):
    
            if y == (len(train_loader) - 1):
                break
    
            # Run train function on batch data
            tr_inputs, tr_outputs, tr_loss, tr_kl, tr_eucl, tr_reco_p, tr_reco_j, tr_reco_pt, tr_rec_mass  = train(model, jets_train, optimizer)
            tr_loss_aux += tr_loss
            tr_kl_aux += tr_kl
            tr_rec_aux += tr_eucl
    
            # Individual loss components
            tr_rec_p_aux += tr_reco_p
            tr_rec_j_aux += tr_reco_j
            tr_rec_pt_aux += tr_reco_pt
            tr_rec_mass_aux += tr_rec_mass
    
            if (epoch==(N_epochs-1) or stale_epochs>patience):
                # Concat input and output per batch
                batch_input = tr_inputs.cpu().detach().numpy()
                batch_output = tr_outputs.cpu().detach().numpy()
                all_input = np.concatenate((all_input, batch_input), axis=0)
                all_output = np.concatenate((all_output, batch_output), axis=0)
    
        for w, (jets_valid) in enumerate(valid_loader):
    
            if w == (len(valid_loader) - 1):
                break
    
            # Run validate function on batch data
            val_loss, val_kl, val_eucl = validate(model, jets_valid)
            val_loss_aux += val_loss
            val_kl_aux += val_kl
            val_rec_aux += val_eucl
    
        tr_y_loss.append(tr_loss_aux.cpu().detach().item()/(len(train_loader) - 1))
        tr_y_kl.append(tr_kl_aux.cpu().detach().item()/(len(train_loader) - 1))
        tr_y_rec.append(tr_rec_aux.cpu().detach().item()/(len(train_loader) - 1))
    
        # Individual loss components
        tr_y_loss_p.append(tr_rec_p_aux.cpu().detach().item()/(len(train_loader) - 1))
        tr_y_loss_j.append(tr_rec_j_aux.cpu().detach().item()/(len(train_loader) - 1))
        tr_y_loss_pt.append(tr_rec_pt_aux.cpu().detach().item()/(len(train_loader) - 1))
        tr_y_loss_mass.append(tr_rec_mass_aux.cpu().detach().item()/(len(train_loader) - 1))
    
        val_y_loss.append(val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1))
        val_y_kl.append(val_kl_aux.cpu().detach().item()/(len(valid_loader) - 1))
        val_y_rec.append(val_rec_aux.cpu().detach().item()/(len(valid_loader) - 1))
    
        if stale_epochs > patience:
            print("Early stopped")
            break
    
        if val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1) < min_loss:
            min_loss = val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1)
            stale_epochs = 0
        else:
            stale_epochs += 1
            print('stale_epochs:', stale_epochs)
    
        print('Epoch: {} -- Train loss: {} -- Validation loss: {}'.format(epoch, tr_loss_aux.cpu().detach().item()/(len(train_loader)-1), val_loss_aux.cpu().detach().item()/(len(valid_loader)-1)))
    
        if((epoch+1)%saving_epoch==0 or stale_epochs>patience):
    
            #######################################################################################################
            int_time = time.time()
            print('The time to run the network is:', (int_time - start_time)/60.0, 'minutes')
            
            ######################## Training data ########################
            px_train = all_input[:,0,0,:]
            py_train = all_input[:,0,1,:]
            pz_train = all_input[:,0,2,:]
            
            px_reco_train = all_output[:,0,0,:]
            py_reco_train = all_output[:,0,1,:]
            pz_reco_train = all_output[:,0,2,:]
            
            if((epoch+1)==N_epochs or stale_epochs>patience):
                # Plot each component of the loss function
                plt.figure()
                plt.plot(x_graph, tr_y_kl, label = "Train KL Divergence")
                plt.plot(x_graph, tr_y_rec, label = 'Train Reconstruction Loss')
                plt.plot(x_graph, tr_y_loss, label = 'Train Total Loss')
                plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence")
                plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss')
                plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss')
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('A. U.')
                plt.title('Loss Function Components')
                plt.legend()
                plt.savefig('pxpypz_standardized_beta01_latent20' + str(model_name) + '.pdf')
                plt.clf()
            
            if((epoch+1)==N_epochs or stale_epochs>patience):        
                # Plot each depedent component of the loss function 
                plt.figure()
                plt.plot(y_graph, tr_y_loss_p, label = 'Train Reco - Particles Loss')
                plt.plot(y_graph, tr_y_loss_j, label = 'Train Reco - Jets Loss (a_Penalty)')
                plt.plot(y_graph, tr_y_loss_pt, label = 'Train Reco - Jets $p_T$')
                plt.plot(y_graph, tr_y_loss_mass, label = 'Train Reco - Jets Mass')
                plt.yscale('log')
                plt.xlabel('Epoch')
                #plt.ylabel('A. U.')
                plt.title('Dependent Components - NND')
                plt.legend()
                plt.savefig('pxpypz_standardized_loss_components_latent20' + str(model_name) + '.pdf')
                plt.clf()    
            
            ####################################### EVALUATION #######################################
            all_input_test = np.empty(shape=(0, 1, num_features, num_particles))
            all_output_test = np.empty(shape=(0, 1, num_features, num_particles))
            
            for i, (jets) in enumerate(test_loader):
    
                if i == (len(test_loader)-1):
                    break
                # run test function on batch data for testing
                test_inputs, test_outputs, ts_loss, ts_kl, ts_eucl = test_unseed_data(model, jets)
                batch_input_ts = test_inputs.cpu().detach().numpy()
                batch_output_ts = test_outputs.cpu().detach().numpy()
                all_input_test = np.concatenate((all_input_test, batch_input_ts), axis=0)
                all_output_test = np.concatenate((all_output_test, batch_output_ts), axis=0)
            
            ######################## Test data ########################
            
            px_test = all_input_test[:,0,0,:]
            py_test = all_input_test[:,0,1,:]
            pz_test = all_input_test[:,0,2,:] 
    
            px_reco_test = all_output_test[:,0,0,:]
            py_reco_test = all_output_test[:,0,1,:]
            pz_reco_test = all_output_test[:,0,2,:]
            
            ####################################### GENERATION #######################################
            gen_output = np.empty(shape=(0, 1, num_features, num_particles))
           
            for g, (jets) in enumerate(gen_loader):
    
                if g == (len(gen_loader) - 1):
                    break
                # generation
                z = torch.randn(batch_size, latent_dim).cuda()
                generated_output = model.decode(z)
                batch_gen_output = generated_output.cpu().detach().numpy()
                gen_output = np.concatenate((gen_output, batch_gen_output), axis=0)

            # Check arrays expected size
            px_gen = gen_output[:,0,0,:]
            py_gen = gen_output[:,0,1,:]
            pz_gen = gen_output[:,0,2,:]

            ############################################ Compute ############################################
            # Read data (input & output scaled). 
            px = torch.from_numpy(px_test)
            py = torch.from_numpy(py_test)
            pz = torch.from_numpy(pz_test)
            
            # Model output
            px_reco_0 = torch.from_numpy(px_reco_test)
            py_reco_0 = torch.from_numpy(py_reco_test)
            pz_reco_0 = torch.from_numpy(pz_reco_test)
            
            # Model generation 
            px_gen_0 = torch.from_numpy(px_gen)
            py_gen_0 = torch.from_numpy(py_gen)
            pz_gen_0 = torch.from_numpy(pz_gen)
    
            def inverse_standardize_t(X, tmin, tmax):
                mean = tmin
                std = tmax
                original_X = ((X * (std - mean)) + mean)
    #            original_X = ((X * std) + mean)
                return original_X
                    
            px_r = inverse_standardize_t(px, tr_0_min,tr_0_max)
            py_r = inverse_standardize_t(py, tr_1_min,tr_1_max)
            pz_r = inverse_standardize_t(pz, tr_2_min,tr_2_max)
            
            # Test data 
            px_reco_r_0 = inverse_standardize_t(px_reco_0, tr_0_min,tr_0_max)
            py_reco_r_0 = inverse_standardize_t(py_reco_0, tr_1_min,tr_1_max)
            pz_reco_r_0 = inverse_standardize_t(pz_reco_0, tr_2_min,tr_2_max)
            
            # Gen data 
            px_gen_r_0 = inverse_standardize_t(px_gen_0, tr_0_min,tr_0_max)
            py_gen_r_0 = inverse_standardize_t(py_gen_0, tr_1_min,tr_1_max)
            pz_gen_r_0 = inverse_standardize_t(pz_gen_0, tr_2_min,tr_2_max)
    
            n_jets = px_r.shape[0]
            ######################################################################################
            # Masking for input & output constraints
            
            # Input constraints
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
            
            inputs = torch.stack([px_r, py_r, pz_r], dim=1)
            masked_inputs = mask_zero_padding(inputs)
    
            # Test data
            outputs_0 = torch.stack([px_reco_r_0, py_reco_r_0, pz_reco_r_0], dim=1)
            masked_outputs_0 = mask_zero_padding(outputs_0) # Now, values that correspond to the min-pt should be zeroed.
    
            # Gen data
            gen_outputs_0 = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
            masked_gen_outputs_0 = mask_zero_padding(gen_outputs_0)
    
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
    
            pt_masked_inputs = mask_min_pt(masked_inputs) # Now, values that correspond to the min-pt should be zeroed.
    
            # Test data
    #        outputs_0 = torch.stack([px_reco_r_0, py_reco_r_0, pz_reco_r_0], dim=1)
            pt_masked_outputs_0 = mask_min_pt(masked_outputs_0) # Now, values that correspond to the min-pt should be zeroed.
            
            # Gen data
    #        gen_outputs_0 = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
            pt_masked_gen_outputs_0 = mask_min_pt(masked_gen_outputs_0)
            ######################################################################################
            # LVs from jets
            # Create a four Lorentz Vector from px,py,pz for each jet, set mass to zero
    #        def jet_from_pxpypzm(one_particle):
    #            px, py, pz, m = one_particle 
    #            jet = hep.vectors.LorentzVector()
    #            jet.setpxpypzm(px, py, pz, m)
    #            return jet 
            
    #        def jet_samples_from_particle_samples(jet_constituents_data):
    #            '''
    #            :param particles: N x 100 x 3 ( N .. number of events, 100 particles, 3 features (px, py, pz) )
    #            :return: N x 1 ( N events, each consisting of 1 jet )
    #            '''
    #            event_jets = []
    #            for jet in jet_constituents_data:
    #                particle_jets = [jet_from_pxpypzm(particle) for particle in jet] #j1p1-30
    #                event_jets.append(reduce(lambda x,y: x+y, particle_jets)) # sum all particle-jets to get event-jet
    #            return event_jets
            
            px_r_masked = pt_masked_inputs[:,0,:].detach().cpu().numpy()
            py_r_masked = pt_masked_inputs[:,1,:].detach().cpu().numpy()
            pz_r_masked = pt_masked_inputs[:,2,:].detach().cpu().numpy()
            mass = np.zeros((pz_r_masked.shape[0], num_particles))
            
            input_data = np.stack((px_r_masked, py_r_masked, pz_r_masked, mass), axis=2)
            
            # Test data
            px_reco_r_0_masked = pt_masked_outputs_0[:,0,:].detach().cpu().numpy()
            py_reco_r_0_masked = pt_masked_outputs_0[:,1,:].detach().cpu().numpy()
            pz_reco_r_0_masked = pt_masked_outputs_0[:,2,:].detach().cpu().numpy()
            mass_reco_0 = np.zeros((pz_reco_r_0_masked.shape[0], num_particles))
            
            # Test data
            output_data_0 = np.stack((px_reco_r_0_masked, py_reco_r_0_masked, pz_reco_r_0_masked, mass_reco_0), axis=2)
            
            # Gen data
            px_gen_r_0_masked = pt_masked_gen_outputs_0[:,0,:].detach().cpu().numpy()
            py_gen_r_0_masked = pt_masked_gen_outputs_0[:,1,:].detach().cpu().numpy()
            pz_gen_r_0_masked = pt_masked_gen_outputs_0[:,2,:].detach().cpu().numpy()
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
                return np.stack((part_pt, part_eta, part_phi, part_mass), axis=2)
    
            hadr_input_data = ptetaphim_particles(input_data)
            hadr_output_data = ptetaphim_particles(output_data_0)
            hadr_gen_output_data = ptetaphim_particles(gen_output_data_0)
    
            input_cart = np.empty(shape=(0,3))
            input_hadr = np.empty(shape=(0,3))
            output_cart = np.empty(shape=(0,3))
            output_hadr = np.empty(shape=(0,3))
            gen_cart = np.empty(shape=(0,3))
            gen_hadr = np.empty(shape=(0,3))
    
            if((epoch+1)==N_epochs or stale_epochs>patience):
    
                for d in range(len(input_data)):
                    input_cart = np.concatenate((input_cart,input_data[d,:,:3]),axis=0)
                    input_hadr = np.concatenate((input_hadr,hadr_input_data[d,:,:3]),axis=0)
                    output_cart = np.concatenate((output_cart,output_data_0[d,:,:3]),axis=0)
                    output_hadr = np.concatenate((output_hadr,hadr_output_data[d,:,:3]),axis=0)
                    gen_cart = np.concatenate((gen_cart,gen_output_data_0[d,:,:3]),axis=0)
                    gen_hadr = np.concatenate((gen_hadr,hadr_gen_output_data[d,:,:3]),axis=0)
        
                inppx, bins, _ = plt.hist(input_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outpx = plt.hist(output_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genpx = plt.hist(gen_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle px (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_px_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                inppy, bins, _ = plt.hist(input_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outpy = plt.hist(output_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genpy = plt.hist(gen_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle py (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_py_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                inppz, bins, _ = plt.hist(input_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outpz = plt.hist(output_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genpz = plt.hist(gen_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle pz (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_pz_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                inppt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle pt (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_pt_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                inplowpt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outlowpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genlowpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle pt (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_low_pt_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                inpeta, bins, _ = plt.hist(input_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outeta = plt.hist(output_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                geneta = plt.hist(gen_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle eta (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_eta_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                inpphi, bins, _ = plt.hist(input_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                outphi = plt.hist(output_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                genphi = plt.hist(gen_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('Particle phi (GeV)')
                plt.yscale('log')
                plt.legend(loc='upper right', prop={'size': 16})
                plt.savefig('part_phi_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
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
            jets_output_data = jet_features(hadr_output_data)
            jets_gen_output_data = jet_features(hadr_gen_output_data)
    
            minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
            mout = plt.hist(jets_output_data[:,0], bins=100, range=[0, 400], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
            mgen = plt.hist(jets_gen_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
            plt.clf()
            
            ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            ptout = plt.hist(jets_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
    
            ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            ptgen = plt.hist(jets_gen_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            eout = plt.hist(jets_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            egen= plt.hist(jets_gen_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            etaout = plt.hist(jets_output_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
    
            etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            etagen = plt.hist(jets_gen_output_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            phiout = plt.hist(jets_output_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
    
            phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
            phigen = plt.hist(jets_gen_output_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
            plt.clf()
            
            minp = (minp/minp.sum()) + 0.000000000001
            ptinp = (ptinp/ptinp.sum()) + 0.000000000001
            einp = (einp/einp.sum()) + 0.000000000001
            etainp = (etainp/etainp.sum()) + 0.000000000001
            phiinp = (phiinp/phiinp.sum()) + 0.000000000001
            
            mout = (mout[0]/mout[0].sum()) + 0.000000000001
            ptout = (ptout[0]/ptout[0].sum()) + 0.000000000001
            eout = (eout[0]/eout[0].sum()) + 0.000000000001
            etaout = (etaout[0]/etaout[0].sum()) + 0.000000000001
            phiout = (phiout[0]/phiout[0].sum()) + 0.000000000001
            
            mgen = (mgen[0]/mgen[0].sum()) + 0.000000000001
            ptgen = (ptgen[0]/ptgen[0].sum()) + 0.000000000001
            egen = (egen[0]/egen[0].sum()) + 0.000000000001
            etagen = (etagen[0]/etagen[0].sum()) + 0.000000000001
            phigen = (phigen[0]/phigen[0].sum()) + 0.000000000001
    
            emdt_m = wasserstein_distance(mout,minp)
            emdt_pt = wasserstein_distance(ptout,ptinp)
            emdt_e = wasserstein_distance(eout,einp)
            emdt_eta = wasserstein_distance(etaout,etainp)
            emdt_phi = wasserstein_distance(phiout,phiinp)
            emdt_sum = emdt_m + emdt_pt + emdt_e + emdt_eta + emdt_phi
    
            emdg_m = wasserstein_distance(mgen,minp)
            emdg_pt = wasserstein_distance(ptgen,ptinp)
            emdg_e = wasserstein_distance(egen,einp)
            emdg_eta = wasserstein_distance(etagen,etainp)
            emdg_phi = wasserstein_distance(phigen,phiinp)
            emdg_sum = emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi
            
            print("EMD test: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdt_m, emdt_pt, emdt_e, emdt_eta, emdt_phi, emdt_sum))
            print("EMD gauss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_m, emdg_pt, emdg_e, emdg_eta, emdg_phi, emdg_sum))
    
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*(epoch+1)))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_m))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_pt))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_e))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_eta))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_phi))
            output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_sum))
    
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*(epoch+1)))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_m))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_pt))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_e))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_eta))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_phi))
            output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_sum))
    
            fifty_epoch.append(epoch+1)
            emdt_epoch.append(emdt_sum)
            emdg_epoch.append(emdt_sum)
    
            if(emdt_sum <= min_emdt or stale_epochs>patience or (epoch+1)==N_epochs):
    
                min_emdt = emdt_sum
    
                minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                mout = plt.hist(jets_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet mass (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_mass_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                ptout = plt.hist(jets_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $p_T$ (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_pt_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                eout = plt.hist(jets_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('$jet energy$ (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_energy_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                etaout = plt.hist(jets_output_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $\eta$')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_eta_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
        
                phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                phiout = plt.hist(jets_output_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $\phi$')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_phi_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()

                torch.save(model.state_dict(), 'model_pxpypz_standardized_3DLoss_beta01_latent20'+ str(model_name) + '.pt')

                os.system('mv model_pxpypz_standardized_3DLoss_beta01_latent20'+str(model_name)+'.pt '+str(dir_name))

                print('############ The minimum emdt sum for ',latent_dim,' latent vector dimensions is ',min_emdt,' ############')

            if(emdg_sum <= min_emdg or stale_epochs>patience):
    
                min_emdg = emdg_sum
    
                minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                mgen = plt.hist(jets_gen_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet mass (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_gen_mass_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                ptgen = plt.hist(jets_gen_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $p_T$ (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_gen_pt_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                egen = plt.hist(jets_gen_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('$jet energy$ (GeV)')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_gen_energy_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                etagen = plt.hist(jets_gen_output_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $\eta$')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_gen_eta_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
                phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                phigen = plt.hist(jets_gen_output_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                plt.ylabel("Probability (a.u.)")
                plt.xlabel('jet $\phi$')
                plt.yscale('linear')
                plt.legend(loc='lower right', prop={'size': 16})
                plt.savefig('jet_gen_phi_GeV'+str(model_name)+'.pdf', format='pdf', bbox_inches='tight')
                plt.clf()
    
#                torch.save(model.state_dict(), 'model_pxpypz_standardized_3DLoss_beta01_latent20'+ str(model_name) + '.pt')

#                os.system('mv model_pxpypz_standardized_3DLoss_beta01_latent20'+str(model_name)+'.pt '+str(dir_name))

#                print('############ The minimum emdg sum for ',latent_dim,' latent vector dimensions is ',min_emdg,' ############')

            torch.save(output_tensor_emdt,'emdt'+str(model_name)+'.pt')
            torch.save(output_tensor_emdg,'emdg'+str(model_name)+'.pt')

            os.system('mv emdt'+str(model_name)+'.pt '+str(dir_name))
            os.system('mv emdg'+str(model_name)+'.pt '+str(dir_name))

    os.system('mv jet_mass_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_pt_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_energy_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_eta_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_phi_GeV'+str(model_name)+'.pdf '+str(dir_name))
    
    os.system('mv jet_gen_mass_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_gen_pt_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_gen_energy_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_gen_eta_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv jet_gen_phi_GeV'+str(model_name)+'.pdf '+str(dir_name))
    
    os.system('mv part_px_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_py_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_pz_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_pt_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_low_pt_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_eta_GeV'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv part_phi_GeV'+str(model_name)+'.pdf '+str(dir_name))
    
    os.system('mv pxpypz_standardized_beta01_latent20'+str(model_name)+'.pdf '+str(dir_name))
    os.system('mv pxpypz_standardized_loss_components_latent20'+str(model_name)+'.pdf '+str(dir_name))

#    os.system('mv emdt'+str(model_name)+'.pt '+str(dir_name))
#    os.system('mv emdg'+str(model_name)+'.pt '+str(dir_name))

#    os.system('mv model_pxpypz_standardized_3DLoss_beta01_latent20'+str(model_name)+'.pt '+str(dir_name))

end_time = time.time()
print("The total time is ",((end_time-start_time)/60.0)," minutes.")
