import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch.distributions import MultivariateNormal

latent_data = torch.load('latent_dim_values_for_training.pt')
latent_data = latent_data.type(torch.FloatTensor)
latent_dim = 30
#num_flows = 1
N_epochs = 100
batch_size = 100
learning_rate = 0.00001
num_flows_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
train_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=True)


def VariationalLoss(z, sum_log_det_J):
    distr = MultivariateNormal(torch.zeros(latent_dim).cuda(), torch.eye(latent_dim).cuda())
    target_density_log_prob = -distr.log_prob(z)
    return (target_density_log_prob - sum_log_det_J).mean()

class MADE(nn.Module):
    """
    See Also:
        Germain et al. (2015, Feb 12) MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509
    """

    # Don't use ReLU, so that neurons don't get nullified.
    # This makes sure that the autoregressive test can verified
    def __init__(self, in_features, hidden_features):

        super().__init__()
        self.layers = SequentialMasked(
            LinearMasked(in_features, hidden_features, in_features),
            nn.ELU(),
            LinearMasked(hidden_features, hidden_features, in_features),
            nn.ELU(),
            LinearMasked(hidden_features, in_features, in_features),
            nn.Sigmoid(),
        )
        self.layers.set_mask_last_layer()

    def forward(self, x):
        return self.layers(x)

class LinearMasked(nn.Module):
    """
    Masked Linear layers used in Made.

    See Also:
        Germain et al. (2015, Feb 12) MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509

    """

    def __init__(self, in_features, out_features, num_input_features, bias=True):
        """

        Parameters
        ----------
        in_features : int
        out_features : int
        num_input_features : int
            Number of features of the models input X.
            These are needed for all masked layers.
        bias : bool
        """
        super(LinearMasked, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.num_input_features = num_input_features

        # Make sure that d-values are assigned to m
        # d = 1, 2, ... D-1
        d = set(range(1, num_input_features))
        c = 0
        while True:
            c += 1
            if c > 10:
                break
            # m function of the paper. Every hidden node, gets a number between 1 and D-1
            self.m = torch.randint(1, num_input_features, size=(out_features,)).type(
                torch.int32
            )
            if len(d - set(self.m.numpy())) == 0:
                break

        self.register_buffer(
            "mask", torch.ones_like(self.linear.weight).type(torch.uint8)
        )

    def set_mask(self, m_previous_layer):
        """
        Sets mask matrix of the current layer.

        Parameters
        ----------
        m_previous_layer : tensor
            m values for previous layer layer.
            The first layers should be incremental except for the last value,
            as the model does not make a prediction P(x_D+1 | x_<D + 1).
            The last prediction is P(x_D| x_<D)
        """
        self.mask[...] = (m_previous_layer[:, None] <= self.m[None, :]).T

    def forward(self, x):
        if self.linear.bias is None:
            b = 0
        else:
            b = self.linear.bias

        return F.linear(x, self.linear.weight * self.mask, b)

class SequentialMasked(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

        input_set = False
        for i in range(len(args)):
            layer = self.__getitem__(i)
            if not isinstance(layer, LinearMasked):
                continue
            if not input_set:
                layer = set_mask_input_layer(layer)
                m_previous_layer = layer.m
                input_set = True
            else:
                layer.set_mask(m_previous_layer)
                m_previous_layer = layer.m

    def set_mask_last_layer(self):
        reversed_layers = filter(
            lambda l: isinstance(l, LinearMasked), reversed(self._modules.values())
        )

        # Get last masked layer
        layer = next(reversed_layers)
        prev_layer = next(reversed_layers)
        set_mask_output_layer(layer, prev_layer.m)


def set_mask_output_layer(layer, m_previous_layer):
    # Output layer has different m-values.
    # The connection is shifted one value to the right.
    layer.m = torch.arange(0, layer.num_input_features)
    layer.set_mask(m_previous_layer)
    return layer


def set_mask_input_layer(layer):
    m_input_layer = torch.arange(1, layer.num_input_features + 1)
    m_input_layer[-1] = 1e9
    layer.set_mask(m_input_layer)
    return layer

class AutoRegressiveNN(MADE):
    def __init__(self, in_features, hidden_features, context_features):
        super().__init__(in_features, hidden_features)
        self.context = nn.Linear(context_features, in_features)
        # remove MADE output layer
        del self.layers[len(self.layers) - 1]

    def forward(self, z, h):
        return self.layers(z) + self.context(h)

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, size, context_size, auto_regressive_hidden):
        super().__init__()
        self.context_size = context_size
        self.s_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        self.m_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )

    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, z, h=None):
        if h is None:
            h = torch.randn(self.context_size).cuda()

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = (self.s_t(z, h)).cuda()
        sigma_t = torch.sigmoid(s_t).cuda()
        m_t = (self.m_t(z, h)).cuda()

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t, sigma_t

class BasicFlow(nn.Module):
    def __init__(self, dim, n_flows):
        super().__init__()
        self.layers = [IAF(latent_dim,10,16) for _ in range(n_flows)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        log_det_J = 0

        for layer in self.layers:
            z, sigma_t = layer(z)
            log_det_J += layer.determine_log_det_jac(sigma_t)

        return z, log_det_J

for num_flows in num_flows_seq:

    flow = BasicFlow(dim=latent_dim, n_flows=num_flows).cuda()
    optim = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    
    epoch = []
    loss_list = []
    mean_list = []
    stddev_list = []
    
    for i in range(N_epochs):

        znp = np.empty(shape=(0, latent_dim))
        int_time = time.time()

        for y, (jets_train) in enumerate(train_loader):

            jets_train = jets_train.cuda()
            zk, ldj = flow(jets_train)
    
            loss = VariationalLoss(zk,ldj.cuda())
            loss.backward()
            optim.step()
            optim.zero_grad()
            znp = np.concatenate((znp,zk.cpu().detach().numpy()),axis=0)
    
        epoch.append(i)
        loss_list.append(loss.item())
    
        mean = np.mean(np.mean(znp,axis=0))
        stddev = np.mean(np.std(znp,axis=0))
    
        mean_list.append(mean)
        stddev_list.append(stddev)
    
        int_time2 = time.time()
    
        print("Epoch: ",i," ---- Flows: ",num_flows," ---- Loss: ","{:.3f}".format(loss.item()))
        print("The mean of the means is {:.5f}, and the mean of the stddevs is {:.5f}".format(mean,stddev))
        print("The time to run this epoch is {:.3f} minutes".format((int_time2-int_time)/60.0))
        print("##############################################################################")

    plt.plot(epoch,loss_list)
    plt.savefig("iaf_"+str(num_flows)+"flows_hardtanh_loss.pdf", format="pdf")
    plt.clf()

    plt.plot(epoch,mean_list)
    plt.savefig("iaf_"+str(num_flows)+"flows_hardtanh_mean_mean.pdf", format="pdf")
    plt.clf()

    plt.plot(epoch,stddev_list)
    plt.savefig("iaf_"+str(num_flows)+"flows_hardtanh_mean_stddev.pdf", format="pdf")
    plt.clf()
