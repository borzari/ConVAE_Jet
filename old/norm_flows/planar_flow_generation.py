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
from functools import reduce
from matplotlib.colors import LogNorm
from torch.distributions import MultivariateNormal

start_time = time.time()

latent_data = torch.load('latent_dim_values_for_training.pt')
latent_data=latent_data.type(torch.FloatTensor)
N_epochs = 100
#N_epochs = 10
batch_size = 100
learning_rate = 0.0001
#num_flows_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
#num_flows_seq = [40]
num_flows_seq = [75,100,150,200]
#num_flows = 1
latent_dim = 30
train_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=True)

for num_flows in num_flows_seq:

    class TargetDistribution:
        def __init__(self, name: str):
            """Define target distribution. 
            Args:
                name: The name of the target density to use. 
                      Valid choices: ["U_1", "U_2", "U_3", "U_4", "ring"].
            """
            self.func = self.get_target_distribution(name)
    
        def __call__(self, z: Tensor) -> Tensor:
            return self.func(z)
    
        @staticmethod
        def get_target_distribution(name: str) -> Callable[[Tensor], Tensor]:
    
            if name == "U_1":
    
                def U_1(z):
                    u = 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2
                    u = u - torch.log(
                        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
                    )
                    return u
    
                return U_1
    
    class VariationalLoss(nn.Module):
        def __init__(self, distribution: TargetDistribution):
            super().__init__()
            self.distr = MultivariateNormal(torch.zeros(latent_dim).cuda(), torch.eye(latent_dim).cuda())
    
        def forward(self, z0: Tensor, z: Tensor, sum_log_det_J: float) -> float:
            target_density_log_prob = -self.distr.log_prob(z)
            return (target_density_log_prob - sum_log_det_J).mean()
    
    class PlanarTransform(nn.Module):
        """Implementation of the invertible transformation used in planar flow:
            f(z) = z + u * h(dot(w.T, z) + b)
        See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf. 
        """
    
        def __init__(self, dim: int = 2):
            """Initialise weights and bias.
            
            Args:
                dim: Dimensionality of the distribution to be estimated.
            """
            super().__init__()
            self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
            self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
    
        def forward(self, z: Tensor) -> Tensor:
            if torch.mm(self.u, self.w.T) < -1:
                self.get_u_hat()
    
#            return z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b)
            return z + self.u * nn.Hardtanh()(torch.mm(z, self.w.T) + self.b)
#            return z + self.u * (torch.mm(z, self.w.T) + self.b)

        def log_det_J(self, z: Tensor) -> Tensor:
            if torch.mm(self.u, self.w.T) < -1:
                self.get_u_hat()
            a = torch.mm(z, self.w.T) + self.b
#            psi = (1 - nn.Tanh()(a) ** 2) * self.w # Tanh
            aaux1 = ((a >= -1.0) | (a <= 1.0))
            aaux2 = (a > 1.0)
            aaux3 = (a < -1.0)
            psi = (self.w * aaux1) + 1.0*(aaux2) - 1.0*(aaux3) # Hardtanh
#            psi = self.w # Linear
            abs_det = (1 + torch.mm(self.u, psi.T)).abs()
            log_det = torch.log(1e-4 + abs_det)
    
            return log_det
    
        def get_u_hat(self) -> None:
            """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition 
            for invertibility of the transformation f(z). See Appendix A.1.
            """
            wtu = torch.mm(self.u, self.w.T)
            m_wtu = -1 + torch.log(1 + torch.exp(wtu))
            self.u.data = (
                self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
            )
    
    class PlanarFlow(nn.Module):
        def __init__(self, dim: int = 2, K: int = 6):
            """Make a planar flow by stacking planar transformations in sequence.
            Args:
                dim: Dimensionality of the distribution to be estimated.
                K: Number of transformations in the flow. 
            """
            super().__init__()
            self.layers = [PlanarTransform(dim) for _ in range(K)]
            self.model = nn.Sequential(*self.layers)
    
        def forward(self, z: Tensor) -> Tuple[Tensor, float]:
            log_det_J = 0
    
            for layer in self.layers:
                log_det_J += layer.log_det_J(z)
                z = layer(z)
    
            return z, log_det_J
    
    density = TargetDistribution("U_1")
    model = PlanarFlow(latent_dim,K=num_flows)
    model = model.cuda()
    bound = VariationalLoss(density)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_per_epoch = 0.0
    
    epoch = []
    loss = []
    mean_list = []
    stddev_list = []
    
    for n in range(N_epochs):
    
        znp = np.empty(shape=(0, latent_dim))
    
        int_time = time.time()
    
        for y, (jets_train) in enumerate(train_loader):
        
            torch.autograd.set_detect_anomaly(True)
        
            jets_train = jets_train.cuda()
        
            if y == (len(train_loader) - 1):
                break
        
            zk, log_det_j = model(jets_train)
       
            train_loss = bound(jets_train,zk.cuda(),log_det_j.cuda())
            optimizer.zero_grad()
            train_loss.backward()
            # Adam optimization using the gradients from backprop
            optimizer.step()
        
            if y == (len(train_loader) - 2):
                loss_per_epoch = train_loss
    
            znp = np.concatenate((znp,zk.cpu().detach().numpy()),axis=0)
    
        epoch.append(n)
        loss.append(loss_per_epoch.item())
    
        mean = np.mean(np.mean(znp,axis=0))
        stddev = np.mean(np.std(znp,axis=0))
    
        mean_list.append(mean)
        stddev_list.append(stddev)
    
        int_time2 = time.time()
    
        print("Epoch: ",n," ---- Flows: ",num_flows," ---- Loss: ","{:.3f}".format(loss_per_epoch.item()))
        print("The mean of the means is {:.5f}, and the mean of the stddevs is {:.5f}".format(mean,stddev))
        print("The time to run this epoch is {:.3f} minutes".format((int_time2-int_time)/60.0))
        print("##############################################################################")
    
    torch.save(model.state_dict(), 'model_planarflow_nflows'+str(num_flows)+'_hardtanh.pt')
#    torch.save(model.state_dict(), 'model_planarflow_nflows'+str(num_flows)+'_linear.pt')

    plt.plot(epoch,loss)
    plt.savefig("planar_"+str(num_flows)+"flows_hardtanh_loss.pdf", format="pdf")
#    plt.savefig("planar_"+str(num_flows)+"flows_linear_loss.pdf", format="pdf")
    plt.clf()
    
    plt.plot(epoch,mean_list)
    plt.savefig("planar_"+str(num_flows)+"flows_hardtanh_mean_mean.pdf", format="pdf")
#    plt.savefig("planar_"+str(num_flows)+"flows_linear_mean_mean.pdf", format="pdf")
    plt.clf()
    
    plt.plot(epoch,stddev_list)
    plt.savefig("planar_"+str(num_flows)+"flows_hardtanh_mean_stddev.pdf", format="pdf")
#    plt.savefig("planar_"+str(num_flows)+"flows_linear_mean_stddev.pdf", format="pdf")
    plt.clf()

    stdgauss = np.random.normal(loc=0.000,scale=1.000,size=len(znp[:,0]))
    
    print(stdgauss.shape)
    print(znp[:,0].shape)
    fig, axs = plt.subplots(5, 6, tight_layout=True)
    
    k = 0
    
    for i in range(5):
        for j in range(6):
            axs[i,j].hist(stdgauss, bins=100, range=[-3.75,3.75])
            axs[i,j].hist(znp[:,k], bins=100, range=[-3.75,3.75], label=str(k+1))
            axs[i,j].set_xticklabels(labels='',fontsize=3)
            axs[i,j].set_yticklabels(labels='',fontsize=3)
            axs[i,j].set_ylim([0,10000])
            axs[i,j].legend(loc='upper right', prop={'size': 6})
            print(k)
            k=k+1

    fig.savefig('latent_dim_histogram_nflows'+str(num_flows)+'_hardtanh.pdf', format='pdf')
#    fig.savefig('latent_dim_histogram_nflows40_linear.pdf', format='pdf')

end_time = time.time()

print("The time to run the network is ",(end_time-start_time)/60.0," minutes.")
