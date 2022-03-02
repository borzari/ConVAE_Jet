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
#import mplhep as mhep
#plt.style.use(mhep.style.CMS)
import skhep.math as hep
from scipy.stats import wasserstein_distance
from functools import reduce
from matplotlib.colors import LogNorm
from torch.distributions import MultivariateNormal

start_time = time.time()

latent_data = torch.load('latent_dim_values_for_training.pt')
latent_data=latent_data.type(torch.FloatTensor)
#ygauss = np.random.normal(loc=0.000,scale=1.000,size=len(latent_data[:,0]))
#ygauss = torch.randn(size=(len(latent_data[:,0]),latent_dim))
N_epochs = 100
#N_epochs = 10
batch_size = 100
learning_rate = 0.0001
#num_flows_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
num_flows_seq = [40]
#num_flows = 1
latent_dim = 30
ygauss = torch.randn(size=(int(1*len(latent_data[:,0])),latent_dim)).cuda()
#train_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=False)
gauss_loader = DataLoader(dataset=ygauss, batch_size=batch_size, shuffle=True)
#act_func = "tanh"
#act_func = "hardtanh"
act_func = "linear"

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

            if act_func == "tanh":
                return z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b), self.u, self.w.T, self.b
            elif act_func == "hardtanh":
                return z + self.u * nn.Hardtanh()(torch.mm(z, self.w.T) + self.b), self.u, self.w.T, self.b
            else:
                return z + self.u * (torch.mm(z, self.w.T) + self.b), self.u, self.w.T, self.b

        def log_det_J(self, z: Tensor) -> Tensor:
            if torch.mm(self.u, self.w.T) < -1:
                self.get_u_hat()
            a = torch.mm(z, self.w.T) + self.b
            psi = (1 - nn.Tanh()(a) ** 2) * self.w
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
            ul = []
            wtl = []
            bl = []

            for layer in self.layers:
                log_det_J += layer.log_det_J(z)
                z, u, wt, b = layer(z)
                ul.append(u)
                wtl.append(wt)
                bl.append(b)

            return z, log_det_J, ul, wtl, bl

    intermediates_cache = {}
    add_inverse_to_cache = True

    def _inverse(y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """
        if (y, 'x') in intermediates_cache:
            x = intermediates_cache.pop((y, 'x'))
            return x
        else:
            raise KeyError("PlanarFlow expected to find "
                           "key in intermediates cache but didn't")

    def add_intermediate_to_cache(intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        assert((y, name) not in intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        intermediates_cache[(y, name)] = intermediate

    density = TargetDistribution("U_1")
    model = PlanarFlow(latent_dim,K=num_flows)
    if act_func == "tanh":
        model.load_state_dict(torch.load('/home/brenoorzari/jets/normalizing_flows/model_planarflow_nflows40.pt'))
    elif act_func == "hardtanh":
        model.load_state_dict(torch.load('/home/brenoorzari/jets/normalizing_flows/model_planarflow_nflows40_hardtanh.pt'))
    else:
        model.load_state_dict(torch.load('/home/brenoorzari/jets/normalizing_flows/model_planarflow_nflows40_linear.pt'))
    model = model.cuda()
    bound = VariationalLoss(density)

    znp = np.empty(shape=(0, latent_dim))
    ynp = np.empty(shape=(0, latent_dim))
    outynp = np.empty(shape=(0, latent_dim))
    yl = []

#    for i, (jets_train) in enumerate(train_loader):
    for i, (gauss_data) in enumerate(gauss_loader):

        outyl = []

#        print(i)
        train_int = int(torch.randint(0,int((len(latent_data[:,0])-1)/100),(1,)))
#        jets_train = jets_train.cuda()
        jets_train = latent_data[int(batch_size*train_int):int(batch_size*(train_int+1))].cuda()

#        if i == (len(train_loader) - 1):
        if i == (len(ygauss) - 1):
            break

        y, log_det_j, u, wt, b = model(jets_train)

        ynp = np.concatenate((ynp,y.cpu().detach().numpy()),axis=0)

        for k in range(len(b)):

            jets_train = jets_train.view(batch_size,1,latent_dim)
            u[k] = u[k].repeat(batch_size,1,1)
            wt[k] = wt[k].repeat(batch_size,1,1)

            if k == 0:
                outy = jets_train + u[k] * nn.Hardtanh()(torch.bmm(jets_train, wt[k]) + b[k])
                outyl.append(outy)
            else:
                outy = outy + u[k] * nn.Hardtanh()(torch.bmm(outy, wt[k]) + b[k])
                outyl.append(outy.view(batch_size,latent_dim))

            u[k] = u[k][0]
            wt[k] = wt[k][0]
            jets_train = jets_train.view(batch_size,latent_dim)

        k = 0

        for k in range(len(b)):

            jets_train = jets_train.view(batch_size,1,latent_dim)
            u[num_flows-k-1] = u[num_flows-k-1].repeat(batch_size,1,1)
            wt[num_flows-k-1] = wt[num_flows-k-1].repeat(batch_size,1,1)

            argaux = wt[num_flows-k-1]
            argaux2 = torch.Tensor()

#            if k != len(b) - 1:
#                argaux2 = torch.bmm(outyl[num_flows-k-2].view(batch_size,1,latent_dim),argaux) + b[num_flows-k-1]
#            else:
#                argaux2 = torch.bmm(jets_train,argaux) + b[num_flows-k-1]

            gauss_data = gauss_data.view(batch_size,1,latent_dim)
            argaux2 = torch.bmm(gauss_data,argaux) + b[num_flows-k-1]
            gauss_data = gauss_data.view(batch_size,latent_dim)

            invaux = torch.inverse(torch.bmm(wt[num_flows-k-1],u[num_flows-k-1])+torch.eye(latent_dim).cuda())
            invaux2 = u[num_flows-k-1]*b[num_flows-k-1]
            invaux3 = gauss_data.view(batch_size,latent_dim,1)-torch.transpose(invaux2,1,2)
#            invaux3 = y.view(batch_size,latent_dim,1)-torch.transpose(invaux2,1,2)
            invy = torch.bmm(torch.transpose(invaux,1,2),invaux3)
            invy = invy.view(batch_size,latent_dim)

            argaux2 = (argaux2.view(batch_size,1)).repeat(1,latent_dim)
            argaux3 = (argaux2 >= 1.0)
            argaux4 = (argaux2 <= -1.0)
            argaux5 = ((argaux2 < 1.0) + (argaux2 > -1.0))

            u[num_flows-k-1] = u[num_flows-k-1].view(batch_size,latent_dim)

#            y = ((y-u[num_flows-k-1])*argaux3) + ((y+u[num_flows-k-1])*argaux4) + ((invy)*argaux5)
#            gauss_data = ((gauss_data-u[num_flows-k-1])*argaux3) + ((gauss_data+u[num_flows-k-1])*argaux4) + ((invy)*argaux5)
            gauss_data = invy
#            y = invy

            u[num_flows-k-1] = u[num_flows-k-1][0]
            wt[num_flows-k-1] = wt[num_flows-k-1][0]

#        znp = np.concatenate((znp,y.cpu().detach().numpy()),axis=0)
        znp = np.concatenate((znp,gauss_data.cpu().detach().numpy()),axis=0)

        if i%int(len(ygauss)/(100*100))==0:
            print(int((i/int(len(ygauss)/(100*100))))+1,"% / 100 %")

    stdgauss = np.random.normal(loc=0.000,scale=1.000,size=len(ynp[:,0]))

    fig, axs = plt.subplots(5, 6, tight_layout=True)
    k = 0
    mean = 0.0

    for i in range(5):
        for j in range(6):
            inp, bins, _ = axs[i,j].hist(latent_data[:,k].numpy(), bins=100, range=[-0.75,0.75], density=True)
            out, bins, _ = axs[i,j].hist(znp[:,k], bins=100, range=[-0.75,0.75], label=str(k+1), density=True)
            inp = (inp/inp.sum()) + 0.000000000001
            out = (out/out.sum()) + 0.000000000001
            emd = wasserstein_distance(out,inp)
            mean = mean + emd
            axs[i,j].set_xticklabels(labels='',fontsize=3)
            axs[i,j].set_yticklabels(labels='',fontsize=3)
#            axs[i,j].set_ylim([0,10000])
            axs[i,j].set_ylim([0,7])
            axs[i,j].legend(loc='upper right', prop={'size': 6})
            k=k+1

    print("The z emd mean is ",mean/30.0)

    if act_func == "tanh":
        fig.savefig('inverted_latent_dim_histogram_nflows40.pdf', format='pdf')
    elif act_func == "hardtanh":
        fig.savefig('inverted_latent_dim_histogram_nflows40_hardtanh_output_ygauss.pdf', format='pdf')
    else:
        fig.savefig('inverted_latent_dim_histogram_nflows40_linear_ygauss.pdf', format='pdf')
    fig.clf()

#    fig1, axs1 = plt.subplots(5, 6, tight_layout=True)
#    k1 = 0
#    emdy = []

#    for i1 in range(5):
#        for j1 in range(6):
#            inp1, bins1, _ = axs1[i1,j1].hist(stdgauss, bins=100, range=[-3.75,3.75], density=True)
#            out1, bins1, _ = axs1[i1,j1].hist(ynp[:,k1], bins=100, range=[-3.75,3.75], label=str(k1+1), density=True)
#            inp1 = (inp1/inp1.sum()) + 0.000000000001
#            out1 = (out1/out1.sum()) + 0.000000000001
#            emd1 = wasserstein_distance(out1,inp1)
#            emdy.append(emd1)
#            axs1[i1,j1].set_xticklabels(labels='',fontsize=3)
#            axs1[i1,j1].set_yticklabels(labels='',fontsize=3)
#            axs1[i1,j1].set_ylim([0,0.6])
#            axs1[i1,j1].legend(loc='upper right', prop={'size': 6})
#            k1=k1+1

#    print("The y emd mean is ",np.mean(np.array(emdy))," and the y emd stddev is ",np.std(np.array(emdy)))

#    if act_func == "tanh":
#        fig1.savefig('inverted_gauss_latent_dim_histogram_nflows40.pdf', format='pdf')
#    elif act_func == "hardtanh":
#        fig1.savefig('inverted_gauss_latent_dim_histogram_nflows40_hardtanh.pdf', format='pdf')
#    else:
#        fig1.savefig('inverted_gauss_latent_dim_histogram_nflows40_linear.pdf', format='pdf')
#    fig1.clf()

    fig2, axs2 = plt.subplots(5, 6, tight_layout=True)
    k2 = 0

    for i2 in range(5):
        for j2 in range(6):
            inp2, bins2, _ = axs2[i2,j2].hist(stdgauss, bins=100, range=[-3.75,3.75], density=True)
            out2, bins2, _ = axs2[i2,j2].hist(latent_data[:,k2].numpy(), bins=100, range=[-3.75,3.75], label=str(k2+1), density=True)
            axs2[i2,j2].set_xticklabels(labels='',fontsize=3)
            axs2[i2,j2].set_yticklabels(labels='',fontsize=3)
            axs2[i2,j2].set_ylim([0,4.0])
            axs2[i2,j2].legend(loc='upper right', prop={'size': 6})
            k2=k2+1

    if act_func == "tanh":
        fig2.savefig('inverted_inputvsgauss_latent_dim_histogram_nflows40.pdf', format='pdf')
    elif act_func == "hardtanh":
        fig2.savefig('inverted_inputvsgauss_latent_dim_histogram_nflows40_hardtanh.pdf', format='pdf')
    else:
        fig2.savefig('inverted_inputvsgauss_latent_dim_histogram_nflows40_linear.pdf', format='pdf')
    fig2.clf()

end_time = time.time()

print("The time to run the network is ",(end_time-start_time)/60.0," minutes.")
