import abc
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
#from pytorch_generative import nn as pg_nn
#from pytorch_generative.models import base

latent_data = torch.load('latent_dim_values_for_training.pt')
latent_data = latent_data.type(torch.FloatTensor)
latent_dim = 30
#num_flows = 1
N_epochs = 100
batch_size = 100
learning_rate = 0.00001
#num_flows_seq = [1,2,3,5,7,10,15,20,25,30,40,50]
num_flows_seq = [3]
train_loader = DataLoader(dataset=latent_data, batch_size=batch_size, shuffle=True)


def VariationalLoss(z, sum_log_det_J):
    distr = MultivariateNormal(torch.zeros(latent_dim).cuda(), torch.eye(latent_dim).cuda())
    target_density_log_prob = -distr.log_prob(z)
    return (target_density_log_prob - sum_log_det_J).mean()

def _default_sample_fn(logits):
    return distributions.Bernoulli(logits=logits).sample()

class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.
    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, *args, **kwargs):
        if getattr(self, "_c", None) is None and len(args[0].shape) == 4:
            _, self._c, self._h, self._w = args[0].shape
        return super().__call__(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class AutoregressiveModel(GenerativeModel):
    """The base class for Autoregressive generative models. """

    def __init__(self, sample_fn=None):
        """Initializes a new AutoregressiveModel instance.
        Args:
            sample_fn: A fn(logits)->sample which takes sufficient statistics of a
                distribution as input and returns a sample from that distribution.
                Defaults to the Bernoulli distribution.
        """
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    def _get_conditioned_on(self, n_samples, conditioned_on):
        assert (
            n_samples is not None or conditioned_on is not None
        ), 'Must provided one, and only one, of "n_samples" or "conditioned_on"'
        if conditioned_on is None:
            shape = (n_samples, self._c, self._h, self._w)
            conditioned_on = (torch.ones(shape) * -1).to(self.device)
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    # TODO(eugenhotaj): This function does not handle subpixel sampling correctly.
    def sample(self, n_samples=None, conditioned_on=None):
        """Generates new samples from the model.
        Args:
            n_samples: The number of samples to generate. Should only be provided when
                `conditioned_on is None`.
            conditioned_on: A batch of partial samples to condition the generation on.
                Only dimensions with values < 0 are sampled while dimensions with
                values >= 0 are left unchanged. If 'None', an unconditional sample is
                generated.
        """
        with torch.no_grad():
            conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
            n, c, h, w = conditioned_on.shape
            for row in range(h):
                for col in range(w):
                    out = self.forward(conditioned_on)[:, :, row, col]
                    out = self._sample_fn(out).view(n, c)
                    conditioned_on[:, :, row, col] = torch.where(
                        conditioned_on[:, :, row, col] < 0,
                        out,
                        conditioned_on[:, :, row, col],
                    )
            return conditioned_on

class CausalConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.
    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.
    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=False  [1 1 0],    mask_center=True  [1 0 0],
                           [0 0 0]]                      [0 0 0]
    In [1], they refer to the left masks as 'type A' and right as 'type B'.
    NOTE: This layer does *not* implement autoregressive channel masking.
    """

    def __init__(self, mask_center, *args, **kwargs):
        """Initializes a new CausalConv2d instance.
        Args:
            mask_center: Whether to mask the center pixel of the convolution filters.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.zeros((i, o, h, w))
        mask.data[:, :, : h // 2, :] = 1
        mask.data[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class CausalResidualBlock(nn.Module):
    """A residual block masked to respect the autoregressive property."""

    def __init__(self, n_channels):
        """Initializes a new CausalResidualBlock instance.
        Args:
            n_channels: The number of input (and output) channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.ReLU(),
            CausalConv2d(
                mask_center=False,
                in_channels=n_channels // 2,
                out_channels=n_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class PixelCNN(AutoregressiveModel):
    """The PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_residual=15,
        residual_channels=128,
        head_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelCNN instance.
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_residual: The number of residual blocks.
            residual_channels: The number of channels to use in the residual layers.
            head_channels: The number of channels to use in the two 1x1 convolutional
                layers at the head of the network.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=2 * residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._causal_layers = nn.ModuleList(
            [
                CausalResidualBlock(n_channels=2 * residual_channels)
                for _ in range(n_residual)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * residual_channels,
                out_channels=head_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        x = self._input(x)
        for layer in self._causal_layers:
            x = x + layer(x)
        return self._head(x)

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, size):
        super().__init__()
        self.s_t = PixelCNN(
            in_channels=1,
            out_channels=1,
            n_residual=15,
            residual_channels=16,
            head_channels=16
        )
        self.m_t = PixelCNN(
            in_channels=1,
            out_channels=1,
            n_residual=15,
            residual_channels=16,
            head_channels=16
        )

    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, z):

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = (self.s_t(z)).cuda()
        sigma_t = torch.sigmoid(s_t).cuda()
        m_t = (self.m_t(z)).cuda()

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t, sigma_t

class BasicFlow(nn.Module):
    def __init__(self, dim, n_flows):
        super().__init__()
        self.layers = [IAF(latent_dim) for _ in range(n_flows)]
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

            jets_train = jets_train.view(batch_size,1,1,latent_dim).cuda()
            zk, ldj = flow(jets_train)
    
            loss = VariationalLoss(zk,ldj.cuda())
            loss.backward()
            optim.step()
            optim.zero_grad()
            zk = zk.view(batch_size,latent_dim)
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
