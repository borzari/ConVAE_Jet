import torch
from typing import Callable
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import mplhep as mhep
plt.style.use(mhep.style.CMS)
import skhep.math as hep
from scipy.stats import wasserstein_distance
from functools import reduce
from matplotlib.colors import LogNorm
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

start_time = time.time()

N_epochs = 100
batch_size = 100
learning_rate = 0.00001
num_flows_seq = [40]
num_ortho_vecs_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
num_householder_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
latent_dim = 30
bestmodel=99999.9
gstddev=1.0
gmean=0.0
flows = []
min_metric = []
beta_seq = ['998']
patience = 5
file_epoch = 1500
min_emd_per_flow = []

torch.cuda.set_device(0)

for beta in beta_seq:

    latent_data_train = torch.load('../beta0p'+beta+'latent_'+str(file_epoch)+'epochs_dim_values_for_training.pt')
    latent_data_valid = torch.load('../beta0p'+beta+'latent_'+str(file_epoch)+'epochs_dim_values_for_validation.pt')
    latent_data_train=latent_data_train.type(torch.FloatTensor)
    latent_data_valid=latent_data_valid.type(torch.FloatTensor)

    print(len(latent_data_train),len(latent_data_valid))

#    latent_data_train=latent_data_train[:int(len(latent_data_train)/10)]
#    latent_data_valid=latent_data_valid[:int(len(latent_data_valid)/10)]

    print(len(latent_data_train),len(latent_data_valid))

    train_loader = DataLoader(dataset=latent_data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=latent_data_valid, batch_size=batch_size, shuffle=True)

    for num_flows in num_flows_seq:
   
        for num_ortho_vecs in num_ortho_vecs_seq:

            for num_householder in num_householder_seq:

                class VariationalLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.distr = MultivariateNormal(torch.zeros(latent_dim).cuda(), torch.eye(latent_dim).cuda())
                
                    def forward(self, z0: Tensor, z: Tensor, sum_log_det_J: float) -> float:
                        target_density_log_prob = -self.distr.log_prob(z)
                        return (target_density_log_prob - sum_log_det_J).mean()
        
##############################################################################################################################################################################################
        
                class Sylvester(nn.Module):
                    """
                    Sylvester normalizing flow.
                    """
                
                    def __init__(self, num_ortho_vecs):
                
                        super(Sylvester, self).__init__()
                
                        self.num_ortho_vecs = num_ortho_vecs
               
                        self.h = nn.Tanh()
        
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
                
                        # Initialize log-det-jacobian to zero
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
                
                        # Normalizing flows
                        for k in range(self.num_flows_func):
                
                            flow_k = getattr(self, 'flow_' + str(k))
                            q_k = q_ortho[k]
                
                            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_k, b[:, :, :, k], sum_ldj=True)
                
                            z.append(z_k)
                            self.log_det_j += log_det_jacobian
                
                        return self.log_det_j, z[0], z[-1]
        
##############################################################################################################################################################################################
        
                model = HouseholderSylvester(num_householder)
                model = model.cuda()
                bound = VariationalLoss()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                loss_per_epoch = 0.0
                
                epoch = []
                loss = []
                mean_list = []
                stddev_list = []
                
                stale_epoch = 0
                min_emd = 999999.9
           
                # Start training
                for n in range(N_epochs):
                
                    znp = np.empty(shape=(0, latent_dim))
                    znp_valid = np.empty(shape=(0, latent_dim))
        
                    int_time = time.time()
                
                    # Training
                    for y, (jets_train) in enumerate(train_loader):
                    
                        torch.autograd.set_detect_anomaly(True)
                    
                        jets_train = jets_train.cuda()
                    
                        if y == (len(train_loader) - 1):
                            break
                    
                        log_det_j, z0, zk = model(jets_train)
                   
                        train_loss = bound(jets_train,zk.cuda(),log_det_j.cuda())
                        optimizer.zero_grad()
                        train_loss.backward(retain_graph=True)
                        # Adam optimization using the gradients from backprop
                        optimizer.step()
                    
                        if y == (len(train_loader) - 2):
                            loss_per_epoch = train_loss
                
                        znp = np.concatenate((znp,zk.cpu().detach().numpy()),axis=0)
            
                    model.eval()
            
                    stdgauss = np.random.normal(loc=0.000,scale=1.000,size=(len(znp[:,0]),latent_dim))
            
                    # Validating
                    for j, (jets_valid) in enumerate(valid_loader):
            
                        jets_valid = jets_valid.cuda()
            
                        if j == (len(valid_loader) - 1):
                            break
            
                        log_det_j_valid, z0_valid, zk_valid = model(jets_valid)
            
                        znp_valid = np.concatenate((znp_valid,zk_valid.cpu().detach().numpy()),axis=0)
            
                    mean_emd = 0.0
                
                    # Calculating EMD between 1D gaussians and flows output to quantify performance of flows network
                    for i in range(30):
                        inp, bins, _ = plt.hist(stdgauss[:,i], bins=100, range=[-4.0,4.0], density=True)
                        out, bins, _ = plt.hist(znp_valid[:,i], bins=100, range=[-4.0,4.0], density=True)
                        inp = (inp/inp.sum()) + 0.000000000001
                        out = (out/out.sum()) + 0.000000000001
                        emd = wasserstein_distance(out,inp)
                        mean_emd = mean_emd + emd
               
                    plt.clf()
        
                    epoch.append(n)
                    loss.append(loss_per_epoch.item())
                
                    mean = np.mean(np.mean(znp,axis=0))
                    stddev = np.mean(np.std(znp,axis=0))
            
                    metric=(abs(mean-gmean))+(abs(stddev-gstddev))
            
                    mean_list.append(mean)
                    stddev_list.append(stddev)
            
                    if(stale_epoch < patience):
                        if(mean_emd < min_emd):
                            min_emd = mean_emd
                            stale_epoch = 0
                        elif(mean_emd >= min_emd):
                            stale_epoch = stale_epoch + 1
            
                    elif(stale_epoch == patience):
                        torch.save(model.state_dict(), 'best_validated_model_sylvesterflow_nflows'+str(num_flows)+'_ortho'+str(num_ortho_vecs)+'_householder'+str(num_householder)+'_beta0p'+beta+'.pt')
                        print("%%%%%%%%%%%%%%%%%%%%%%%%%% Early stopped %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                        break
            
                    int_time2 = time.time()
                
                    print("Epoch: ",n," ---- Flows: ",num_flows," ---- Ortho: ",num_ortho_vecs," ---- Householder: ",num_householder," ---- Loss: ","{:.3f}".format(loss_per_epoch.item()))
                    print("Stale epoch: ",stale_epoch," ---- Min_emd: ",min_emd," ---- Mean_emd: ",mean_emd)
                    print("The mean of the means is {:.5f}, the mean of the stddevs is {:.5f}, and the metric is {:.5f}".format(mean,stddev,metric))
                    print("The time to run this epoch is {:.3f} minutes".format((int_time2-int_time)/60.0))
                    print("####################################### beta0p"+beta+" #######################################")
                
                plt.plot(epoch,loss)
                plt.savefig("sylvester_"+str(num_flows)+"flows_ortho"+str(num_ortho_vecs)+"_householder"+str(num_householder)+"_beta0p"+beta+"_loss.pdf", format="pdf")
                plt.clf()
                
                plt.plot(epoch,mean_list)
                plt.savefig("sylvester_"+str(num_flows)+"flows_ortho"+str(num_ortho_vecs)+"_householder"+str(num_householder)+"_beta0p"+beta+"_mean_mean.pdf", format="pdf")
                plt.clf()
                
                plt.plot(epoch,stddev_list)
                plt.savefig("sylvester_"+str(num_flows)+"flows_ortho"+str(num_ortho_vecs)+"_householder"+str(num_householder)+"_beta0p"+beta+"_mean_stddev.pdf", format="pdf")
                plt.clf()
            
                stdgauss = np.random.normal(loc=0.000,scale=1.000,size=len(znp[:,0]))
                
                print(stdgauss.shape)
                print(znp[:,0].shape)
                fig, axs = plt.subplots(5, 6, tight_layout=True)
                
                k = 0
                
                for i in range(5):
                    for j in range(6):
                        axs[i,j].hist(stdgauss, bins=100, range=[-3.75,3.75], density=True)
                        axs[i,j].hist(znp[:,k], bins=100, range=[-3.75,3.75], label=str(k+1), density=True)
                        axs[i,j].set_xticklabels(labels='',fontsize=3)
                        axs[i,j].set_yticklabels(labels='',fontsize=3)
                        axs[i,j].legend(loc='upper right', prop={'size': 6})
                        k=k+1
            
                fig.savefig('sylvester_histogram_nflows'+str(num_flows)+'_ortho'+str(num_ortho_vecs)+'_householder'+str(num_householder)+'_beta0p'+beta+'.pdf', format='pdf')
                plt.clf()
           
                min_emd_per_flow.append((num_flows,num_ortho_vecs,num_householder,min_emd))
                print(min_emd_per_flow)

    end_time = time.time()
    
    print("The time to run the network is ",(end_time-start_time)/60.0," minutes.")
