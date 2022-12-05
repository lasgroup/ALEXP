import torch
import gpytorch
from environment.feature_map import *


class KernelFunction(gpytorch.kernels.Kernel):
    is_stationary = False
    """
    This is the base class for kernels.
    """

    def __init__(self, feature_map: FeatureMap, eta: Optional[np.ndarray] = None, **kwargs) -> None:

        # set parameters common among all kernel
        super().__init__(**kwargs)
        self.feature_map = feature_map

        # define variables that will be set later
        if eta is not None:
            assert eta.shape == (self.num_groups,), 'eta must be of same size as number of groups'
            self.eta = eta
        else:
            self.eta = np.ones(feature_map.num_groups) # by default, all features are active

    @property
    def groups(self) -> List[List[int]]:
        return self.feature_map.groups

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def active_groups(self) -> List[int]:
        return np.where(self.eta != 0.0)[0]

    @cached_property
    def feature_size(self) -> int:
        return self.feature_map.size

    @property
    def num_dim_x(self) -> int:
        return self.feature_map.num_dim_x

    def map_eta_to_beta(self, eta: np.ndarray):
        assert eta.shape == (self.num_groups,)
        eta_to_beta = np.ones(self.feature_map.size)
        for e, group in zip(eta, self.groups):
            eta_to_beta[group] *= e
        return eta_to_beta

    def get_feature(self, x):
        return self.feature_map.get_feature(x)


    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, **params):
        phi1 = self.get_feature(x=x1.numpy().squeeze())

        if x2 == None:
            phi2 = self.get_feature(x=x1.numpy().squeeze())
        else:
            phi2 = self.get_feature(x=x2.numpy().squeeze())

        for j in range(self.num_groups):
            phi1[:, j * 2 * self.group_dim: (j + 1) * 2 * self.group_dim] *= np.sqrt(self.eta[j])
            phi2[:, j * 2 * self.group_dim: (j + 1) * 2 * self.group_dim] *= np.sqrt(self.eta[j])

        return torch.from_numpy(np.dot(phi1, phi2.transpose()))
