import torch
import gpytorch
from config import device
from torch.distributions import TransformedDistribution, AffineTransform

class AffineTransformedDistribution(TransformedDistribution):
    r"""
    Implements an affine transformation of a probability distribution p(x)

    x_transformed = mean + std * x , x \sim p(x)

    Args:
        base_dist: (torch.distributions.Distribution) probability distribution to transform
        normalization_mean: (np.ndarray) additive factor to add to x
        normalization_std: (np.ndarray) multiplicative factor for scaling x
    """

    def __init__(self, base_dist, normalization_mean, normalization_std):
        self.loc_tensor = torch.tensor(normalization_mean).float().reshape((1,)).to(device)
        self.scale_tensor = torch.tensor(normalization_std).float().reshape((1,)).to(device)
        normalization_transform = AffineTransform(loc=self.loc_tensor, scale=self.scale_tensor)
        super().__init__(base_dist, normalization_transform)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)

    @property
    def stddev(self):
        return torch.exp(torch.log(self.base_dist.stddev) + torch.log(self.scale_tensor))

    @property
    def variance(self):
        return torch.exp(torch.log(self.base_dist.variance) + 2 * torch.log(self.scale_tensor))



class LearnedGPRegressionModel(gpytorch.models.ExactGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, mean_module=None, covar_module=None):
        super(LearnedGPRegressionModel, self).__init__(train_x, train_y, likelihood)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module

        self.covar_module = covar_module

        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean
        self.likelihood = likelihood

    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x

        # feed through mean module
        if self.learned_mean is not None:
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def prior(self, x):
        self.train()
        return self.__call__(x)

    def posterior(self, x):
        self.eval()
        return self.__call__(x)

    def kl(self, x):
        return torch.distributions.kl.kl_divergence(self.posterior(x), self.prior(x))

    def pred_dist(self, x):
        self.eval()
        return self.likelihood(self.__call__(x))

    def pred_ll(self, x, y):
        pred_dist = self.pred_dist(x)
        return pred_dist.log_prob(y)
