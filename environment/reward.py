from typing import Tuple

from environment.domain import ContinuousDomain, DiscreteDomain, Domain
from environment.solver import DualAnnealingSolver
from environment.kernel import KernelFunction
from environment.feature_map import *


class Environment:
    domain = None

    def __init__(self):
        self.tmax = None
        self._x0 = None
        self._t = 0

    @property
    def name(self):
        return f"{type(self).__module__}.{type(self).__name__}"

    def evaluate(self, x):
        raise NotImplementedError


class BenchmarkEnvironment(Environment):
    has_constraint = None

    def __init__(self, noise_std=0.0, noise_std_constr=0.0, random_state=None):
        super().__init__()

        self._rds = np.random if random_state is None else random_state
        self.noise_std = noise_std
        self.noise_std_constr = noise_std_constr

    def f(self, x):
        """
        Function to be implemented by actual benchmark.
        """
        raise NotImplementedError

    def q_constraint(self, x):
        """ constraint function"""
        raise NotImplementedError

    def evaluate(self, x, x_bp=None):
        if x.ndim == 1:
            x = x.reshape((1, -1))
        assert x.shape == (1, self.domain.d, )
        self._t += 1
        evaluation = {'x': x, 't': self._t}
        evaluation['y_exact'] = np.asscalar(self.f(x))
        evaluation['y_min'] = self.min_value

        evaluation['y_std'] = self.noise_std
        evaluation['y'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        if self.has_constraint:
            evaluation['q_excact'] = np.asscalar(self.q_constraint(x))
            evaluation['q_std'] = self.noise_std_constr
            evaluation['q'] = evaluation['q_excact'] + self.noise_std_constr * self._rds.normal(0, 1)

        if x_bp is not None:
            evaluation['x_bp'] = x_bp
            evaluation['y_exact_bp'] = np.asscalar(self.f(x_bp.reshape(1, -1)))
            evaluation['y_bp'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        return evaluation

    def _determine_minimum(self, max_iter_per_d2=500):
        if isinstance(self.domain, ContinuousDomain):
            solver = DualAnnealingSolver(self.domain, random_state=self._rds,
                                         max_iter=max_iter_per_d2 * self.domain.d**2)
            solution = solver.minimize(lambda x: self.f(x))
            return solution[1]
        elif isinstance(self.domain, DiscreteDomain):
            return np.min(self.f(self.domain.points))

    @property
    def normalization_stats(self):
        if isinstance(self.domain, ContinuousDomain):
            x_points = np.random.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
        elif isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            raise NotImplementedError
        ys = self.f(x_points)
        y_min, y_max = np.min(ys), np.max(ys)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats

    @property
    def normalization_stats_constr(self):
        assert self.has_constraint
        if isinstance(self.domain, ContinuousDomain):
            x_points = np.random.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
        elif isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            raise NotImplementedError
        ys = self.q_constraint(x_points)
        y_min, y_max = np.min(ys), np.max(ys)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats

    def sample_domain_uniform(self, num_samples: int):
        assert num_samples >= 0, 'number of samples must be positive'
        if isinstance(self.domain, ContinuousDomain):
            x =  self._rds.uniform(self.domain.l, self.domain.u, size=num_samples)
        elif isinstance(self.domain, DiscreteDomain):
            x = self._rds.choice(self.domain.points, replace=True, size=num_samples)
        else:
            raise ValueError('Can only sample from ContinuousDomain or DiscreteDomain')
        return x.reshape((num_samples, self.domain.d))


class Reward(BenchmarkEnvironment):
    has_constraint = False

    def __init__(self, kernel: KernelFunction, domain: Domain, beta: Optional[np.ndarray] = None,
                 rkhs_norm: float = 10, noise_std: float = 0.01,  beta_min: float = 1e-2,
                 random_state: Optional[np.random.RandomState] = None):
        super().__init__(noise_std=noise_std, random_state=random_state)

        assert kernel.num_dim_x == domain.d, 'Kernel`s domain must have the same number of dims as the domain'
        self.domain = domain

        # set config variables
        self.num_groups = kernel.num_groups
        self.kernel = kernel
        self.B = rkhs_norm
        self.active_groups = kernel.active_groups

        # mask = np.zeros(self.kernel.feature_size)
        # for i in self.kernel.active_groups:
        #     mask[self.kernel.groups[i]] = 1.0
        # self.eta = mask

        assert beta_min > 0
        self._beta_min = beta_min

        # if beta is not given, sample it
        if beta is None:
            self.beta = self._sample_beta()
        else:
            self.beta = beta
        assert self.beta.shape == (self.kernel.feature_size, )

    @cached_property
    def min_value(self):
        return self._determine_minimum()

    def _sample_beta(self) -> np.ndarray:
        # DO NOT CALL THIS METHOD WITHOUT A PREDEFINED BETA
        raise NotImplementedError

        # # to make sure that beta is bounded away from zero, replace 0 with beta_min
        # beta = self._rds.uniform(self._beta_min, 1, size=(self.kernel.feature_size, ))
        # # sample sign
        # beta *= (self._rds.random(size=(self.kernel.feature_size,)) > 0.5).astype(np.float)
        # beta *= self.eta
        # beta = beta / np.linalg.norm(beta) * self.B
        # return beta

    def f(self, x: np.ndarray) -> np.ndarray:
        psi = self.kernel.get_feature(x)
        return np.dot(psi, self.beta)

    def generate_samples(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        assert num_samples > 0, 'number of samples must be positive integer'
        x = self.sample_domain_uniform(num_samples)
        y = self.f(x)

        if self.noise_std > 0.0:
            y += self._rds.normal(scale=self.noise_std, size=y.shape)
        y = y.reshape((-1, 1))
        assert x.shape[0] == y.shape[0]
        return x, y

