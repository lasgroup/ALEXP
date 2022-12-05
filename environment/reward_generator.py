import numpy as np
from tueplots import bundles

from typing import Optional, Dict, Any, List
from environment.reward import Reward
from environment.domain import ContinuousDomain, DiscreteDomain, Domain
from environment.kernel import KernelFunction


class MetaEnvironment:

    def __init__(self, random_state=None):
        self._rds = np.random if random_state is None else random_state

    def sample_env_param(self):
        raise NotImplementedError

    def sample_env_params(self, num_envs):
        return [self.sample_env_param() for _ in range(num_envs)]

    def sample_envs(self, num_envs):
        pass


class MetaBenchmarkEnvironment(MetaEnvironment):
    env_class = None

    def sample_env(self):
        return self.env_class(**self.sample_env_param(), random_state=self._rds)

    def sample_envs(self, num_envs):
        param_list = self.sample_env_params(num_envs)
        return [self.env_class(**params,
                               random_state=np.random.RandomState(self._rds.randint(0, 10**6)))
                for params in param_list]

    def generate_uniform_meta_train_data(self, num_tasks: int, num_points_per_task: int,
                                         random_state: Optional[np.random.RandomState] = None):
        if random_state is None:
            random_state = self._rds
        envs = self.sample_envs(num_tasks)
        meta_data = []
        for env in envs:
            if isinstance(env.domain, ContinuousDomain):
                x = random_state.uniform(env.domain.l, env.domain.u,
                                      size=(num_points_per_task, env.domain.d))
            elif isinstance(env.domain, DiscreteDomain):
                x = random_state.choice(env.domain.points, num_points_per_task, replace=True)
            else:
                raise AssertionError
            y = env.f(x) + env.noise_std * random_state.normal(0, 1, num_points_per_task)
            meta_data.append((x, y))
        return meta_data

    def generate_uniform_meta_valid_data(self, num_tasks, num_points_context, num_points_test):
        meta_data = self.generate_uniform_meta_train_data(num_tasks, num_points_context+num_points_test)
        meta_valid_data = [(x[:num_points_context], y[:num_points_context],
                            x[num_points_context:], y[num_points_context:]) for x, y in meta_data]
        return meta_valid_data

    @property
    def domain(self):
        return self.env_class.domain

    @property
    def normalization_stats(self):
        meta_data = self.generate_uniform_meta_train_data(20, 1000 * self.domain.d**2)

        if isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            x_points = np.concatenate([y for x, y in meta_data], axis=0)
        y_concat = np.concatenate([y for x, y in meta_data], axis=0)
        y_min, y_max = np.min(y_concat), np.max(y_concat)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats


class KernelGroupSparseMetaEnvironment(MetaBenchmarkEnvironment):
    env_class = Reward

    def __init__(self, kernel: KernelFunction, domain: Domain,
                 sparsity: Optional[int] = None, eta: Optional[np.ndarray] = None,
                 beta_min: float = 1e-2, noise_std: float = 0.01,
                 random_state: Optional[np.random.RandomState] = None):
        super().__init__()
        self._set_rds(random_state)
        self._noise_std = noise_std

        assert kernel.num_dim_x == domain.d, 'Kernel`s domain must have the same number of dims as the domain'
        self._domain = domain
        self.kernel = kernel

        assert beta_min > 0
        self._beta_min = beta_min

        if eta is not None:
            assert eta.shape == (kernel.num_groups,)
            self.eta = eta
            self.active_groups = np.where(eta != 0)[0]
        elif sparsity is not None:
            assert sparsity <= kernel.num_groups, ('number of non-zero groups (sparsity) must be smaller ',
                                                   'than the number of groups')
            self.active_groups = self._rds.choice(kernel.num_groups, sparsity, replace=False)
            self.eta = np.array([int(i in self.active_groups) for i in range(kernel.num_groups)])
        else:
            raise ValueError('Either sparsity or eta must be provided as an argument.')

    def _set_rds(self, random_state: Optional[np.random.RandomState] = None):
        if random_state is None:
            self._rds = np.random
        else:
            self._rds = random_state

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def static_env_param_dict(self) -> Dict[str, Any]:
        return {'kernel': self.kernel, 'domain': self._domain,
                'noise_std': self._noise_std}

    def sample_env_params(self, num_envs: int) -> List[Dict[str, Any]]:
        # make sure that beta is bounded away from zero
        beta = self._rds.uniform(self._beta_min, 1, (num_envs, self.kernel.feature_size))
        # sample sign
        beta *= (self._rds.random(size=(num_envs, self.kernel.feature_size)) > 0.5).astype(np.float)
        # introduce sparsity
        # groups_covered = []
        env_params = []
        for s in range(num_envs):
            # make mask
            eta_mask = np.zeros(self.kernel.num_groups)
            eta_mask[self.active_groups] = 1
            eta_to_beta_mask = self.kernel.map_eta_to_beta(eta_mask)
            beta_masked = beta[s] = eta_to_beta_mask[None, :] * beta[s]
            # make env
            env_param_dict = {'beta': beta_masked.squeeze()}
            env_params.append({**env_param_dict, **self.static_env_param_dict})

        return env_params

