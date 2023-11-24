from environment.domain import DiscreteDomain, ContinuousDomain
from environment.solver import FiniteDomainSolver, EvolutionarySolver, DoubleSolverWrapper
from environment.feature_map import FeatureMap
from algorithms.regression_oracle import LassoOracle
from typing import Optional

import numpy as np

class AcquisitionAlgorithm:
    """
    Algorithm which is defined through an acquisition function.
    """

    def __init__(self, model, domain, x0=None, solver=None, random_state=None):
        super().__init__()

        self.model = model
        self.domain = domain
        self.t = 0
        self._x0 = x0
        self._rds = np.random if random_state is None else random_state
        self.solver = self._get_solver(domain=self.domain) if solver is None else solver

    def acquisition(self, x):
        raise NotImplementedError

    def add_data(self, X, y):
        self.model.add_data(X, y)

    def next(self):
        if self.t == 0:
            if self._x0 is not None:
                x = self._x0
            else:
                x = self.domain.default_x0
        else:
            x, _ = self.solver.minimize(lambda x: self.acquisition(x))
        self.t += 1
        return x

    def explore(self):
        if isinstance(self.domain, ContinuousDomain):
            x = self._rds.uniform(self.domain.l, self.domain.u, size=1)
        elif isinstance(self.domain, DiscreteDomain):
            # print(self.domain.points.shape)
            # print(self._rds.choice(self.domain.points.shape[0]))
            x = self.domain.points[self._rds.choice(self.domain.points.shape[0])]#, replace=True, size=1)
        else:
            raise ValueError('Can only sample from ContinuousDomain or DiscreteDomain')
        return x.reshape((1, self.domain.d))

    def best_predicted(self):
        x_bp, _ = self.solver.minimize(lambda x: self.model.predict_mean_std(x)[0])
        return x_bp

    def _get_solver(self, domain):
        if isinstance(domain, DiscreteDomain):
            return FiniteDomainSolver(domain)
        elif isinstance(domain, ContinuousDomain):
            return DoubleSolverWrapper(solver=EvolutionarySolver(domain, num_particles_per_d2=500,
                                                                 survival_rate=0.98,
                                                                 max_iter_per_d=300, random_state=self._rds),
                                       atol=1e-3, max_repeats=4, double_effort_at_rerun=True,
                                       throw_precision_error=False)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['solver']
        return self_dict

class UCB(AcquisitionAlgorithm):
    def __init__(self, model, domain, beta, **kwargs):
        super().__init__(model, domain, **kwargs)
        self.beta = beta

    def acquisition(self, x):
        pred_mean, pred_std = self.model.predict_mean_std(x)
        return pred_mean - self.beta * pred_std  # since we minimize f - we want to minimize the LCB

class Greedy(AcquisitionAlgorithm):
    def acquisition(self, x):
        pred_mean, pred_std = self.model.predict_mean_std(x)
        return pred_mean

class ESTC:
    def __init__(self, domain, feature_map: FeatureMap,
                 T0: int,
                 likelihood_std: float = 0.01,
                 banditalg='UCB',
                 lambda_coef: Optional[float] = None,
                 delta: float = 0.2,
                 betaUCB: Optional[float] = 2,
                 model_select: bool = False,
                 random_state=None):

        self._rds = np.random if random_state is None else random_state
        self.feature_map = feature_map
        self.domain = domain
        self.t = 0
        self.T0 = T0
        self.betaUCB = betaUCB
        self.model_select = model_select
        self.solver = self._get_solver(domain=self.domain)
        self.policy = banditalg
        self.lasso_oracle = LassoOracle(feature_map=self.feature_map, lambda_coef=lambda_coef,delta=delta,
                                        likelihood_std=likelihood_std, domain=domain, random_state=self._rds)

    def next(self):
        if self.t < self.T0:
            return self.explore()
        else:
            if self.policy == 'UCB':
                x, _ = self.solver.minimize(lambda z: self.ucb(z))
            elif self.policy == 'Greedy':
                x, _ = self.solver.minimize(lambda z: self.greedy(z))
            else:
                raise NotImplementedError
            return x
        # just take action

    def explore(self):
        if isinstance(self.domain, ContinuousDomain):
            x = self._rds.uniform(self.domain.l, self.domain.u, size=1)
        elif isinstance(self.domain, DiscreteDomain):
            x = self.domain.points[self._rds.choice(self.domain.points.shape[0])]  # , replace=True, size=1)
        else:
            raise ValueError('Can only sample from ContinuousDomain or DiscreteDomain')
        return x.reshape((1, self.domain.d))

    def add_data(self, x,y):
        self.lasso_oracle.add_data(x, y)
        # first update your lasso solver/reward estimator
        self.t += 1
        if self.t == self.T0:
            self.lasso_oracle.fit_lasso()
            # this one 1) estimates theta hat... 2) model selects

    def ucb(self, x):
        if self.model_select:
            pred_mean, pred_std = self.lasso_oracle.predict(x)
            return pred_mean - self.betaUCB * pred_std
        else:
            raise NotImplementedError

    def greedy(self, x):
        if self.model_select:
            pred_mean, _ = self.lasso_oracle.predict(x)
            return pred_mean
        else:
            return self.lasso_oracle.celer_predict(x)

    def _get_solver(self, domain):
        if isinstance(domain, DiscreteDomain):
            return FiniteDomainSolver(domain)
        elif isinstance(domain, ContinuousDomain):
            return DoubleSolverWrapper(solver=EvolutionarySolver(domain, num_particles_per_d2=500,
                                                                 survival_rate=0.98,
                                                                 max_iter_per_d=300, random_state=self._rds),
                                       atol=1e-3, max_repeats=4, double_effort_at_rerun=True,
                                       throw_precision_error=False)

    def best_predicted(self):
        if self.t < self.T0:
            argmin = np.argmin(self.lasso_oracle.y_data)
            return self.lasso_oracle.X_data[argmin]
        else:
            if self.model_select:
                x_bp, _ = self.solver.minimize(lambda x: self.lasso_oracle.predict(x)[0])
            else:
                x_bp, _ = self.solver.minimize(lambda x: self.lasso_oracle.celer_predict(x))
        return x_bp
