import numpy as np
import gpytorch
from typing import Optional, Dict, List, Tuple
import torch
from algorithms.utils import _handle_input_dimensionality
from environment.feature_map import FeatureMap, FilteredFeatureMap
from environment.domain import Domain
from algorithms.gp_utils import LearnedGPRegressionModel, AffineTransformedDistribution
from config import device

class RegressionOracle:
    def __init__(self, domain: Domain, feature_map: FeatureMap,
                 eta: Optional[np.ndarray] = None,
                 likelihood_std: float = 0.05,
                 normalize_data: bool = False,
                 normalization_stats: Optional[Dict] = None,
                 random_state: Optional[np.random.RandomState] = None):

        self.normalize_data = normalize_data
        self.input_dim = None
        self._rds = random_state if random_state is not None else np.random
        torch.manual_seed(self._rds.randint(0, 10 ** 7))

        assert domain.d == feature_map.num_dim_x, 'Feature map`s input dims must be the same as domain`s dims'


        self.likelihood_std = likelihood_std
        self.feature_map = feature_map
        self.domain = domain
        self.input_dim = domain.d
        self.output_dim = 1
        self._normalization_stats = normalization_stats


        self.active_groups = np.where(eta!= 0)[0]

        self.feature_map = FilteredFeatureMap(feature_map=self.feature_map, eta=eta)


        """  ------ Setup model ------ """
        self.num_groups = feature_map.num_groups

        self.covar_module = gpytorch.kernels.LinearKernel().to(device)
        self.mean_module = gpytorch.means.ZeroMean().to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.likelihood.noise = self.likelihood_std**2

        """ ------- normalization stats & data setup ------- """
        self._set_normalization_stats(self._normalization_stats)
        self.reset_to_prior()

    def _prior(self, x):
        mean_x = self.mean_module(x).squeeze()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_to_prior(self) -> None:
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def _reset_posterior(self):
        x_context = self._get_feature(self.X_data).to(device)
        y_context = torch.from_numpy(self.y_data).float().to(device)
        self.gp = LearnedGPRegressionModel(x_context, y_context, self.likelihood,
                                           learned_kernel=None, learned_mean=None,
                                           covar_module=self.covar_module, mean_module=self.mean_module)
        self.gp.eval()
        self.likelihood.eval()

    def add_data(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 1 or X.ndim == 2

        # handle input dimensionality
        X, y = self._handle_input_dim(X, y)

        # normalize data
        if self.normalize_data:
            X, y = self._normalize_data(X, y)
        y = y.flatten()

        if self._num_train_points == 0 and y.shape[0] == 1:
            # for some reason gpytorch can't deal with one data point
            # thus store first point double and remove later
            self.X_data = np.concatenate([self.X_data, X])
            self.y_data = np.concatenate([self.y_data, y])
        if self._num_train_points == 1 and self.X_data.shape[0] == 2:
            # remove duplicate datapoint
            self.X_data = self.X_data[:1, :]
            self.y_data = self.y_data[:1]

        self.X_data = np.concatenate([self.X_data, X])
        self.y_data = np.concatenate([self.y_data, y])

        self._num_train_points += y.shape[0]

        assert self.X_data.shape[0] == self.y_data.shape[0]
        assert self._num_train_points == 1 or self.X_data.shape[0] == self._num_train_points

        self._reset_posterior()

    def confidence_intervals(self, test_x: np.ndarray, confidence: float = 0.9,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            pred_dist = self.predict(test_x, return_density=True, **kwargs)
            pred_dist = self._vectorize_pred_dist(pred_dist)
            test_x = self._get_feature(test_x)
            if type(test_x) == torch.Tensor:
                test_x = test_x.numpy()
            alpha = (1 - confidence) / 2
            ucb = pred_dist.icdf(torch.ones(test_x.shape[0]) * (1 - alpha))
            lcb = pred_dist.icdf(torch.ones(test_x.shape[0]) * alpha)
            return lcb.numpy(), ucb.numpy()

    def predict(self, test_x: np.ndarray, return_density: bool = False, **kwargs):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        if type(test_x) == torch.Tensor:
            test_x = test_x.numpy()
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            if type(test_x_normalized) == np.ndarray:
                test_x_tensor = torch.from_numpy(test_x_normalized).float().to(device)
            else:
                test_x_tensor = test_x_normalized.float().to(device)

            post_f = self.gp(self._get_feature(test_x_tensor))
            pred_dist = post_f
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std


    def predict_mean_std(self, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict(test_x, return_density=False)

    def state_dict(self):
        state_dict = {
            'model': self.gp.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.gp.load_state_dict(state_dict['model'])

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def _get_feature(self, x):
        if type(x) == torch.Tensor:
            x = x.numpy()
        features = self.feature_map.get_feature(x)
        return torch.from_numpy(features).float()


    def eval(self, test_x, test_y, **kwargs):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_y: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        # convert to tensors
        test_x, test_y = _handle_input_dimensionality(test_x, test_y)
        test_t_tensor = torch.from_numpy(test_y).contiguous().float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(test_x, return_density=True, **kwargs)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_t_tensor)
            calibr_error_chi2 = _calib_error_chi2(pred_dist_vect, test_t_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item(), calibr_error_chi2

#==============

    def _reset_data(self):
        self.X_data = torch.empty(size=(0, self.input_dim), dtype=torch.float64)
        self.y_data = torch.empty(size=(0,), dtype=torch.float64)
        self._num_train_points = 0

    def _handle_input_dim(self, X, y):
        if X.ndim == 1:
            assert X.shape[-1] == self.input_dim
            X = X.reshape((-1, self.input_dim))

        if isinstance(y, float) or y.ndim == 0:
            y = np.array(y)
            y = y.reshape((1,))
        elif y.ndim == 1:
            pass
        elif y.ndim == 2 and y.shape[-1] == 1:
            y = y.reshape((y.shape[0],))
        else:
            raise AssertionError('y must not have more than 1 dim')
        return X, y

    def _set_normalization_stats(self, normalization_stats_dict=None):
        if normalization_stats_dict is None:
            self.x_mean, self.y_mean = np.zeros(self.input_dim), np.zeros(1)
            self.x_std, self.y_std = np.ones(self.input_dim), np.ones(1)
        else:
            self.x_mean = normalization_stats_dict['x_mean'].reshape((self.input_dim,))
            self.y_mean = normalization_stats_dict['y_mean'].squeeze()
            self.x_std = normalization_stats_dict['x_std'].reshape((self.input_dim,))
            self.y_std = normalization_stats_dict['y_std'].squeeze()

    def _calib_error(self, pred_dist_vectorized, test_t_tensor):
        return _calib_error(pred_dist_vectorized, test_t_tensor)

    def _compute_normalization_stats(self, X, Y):
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0) + 1e-8, np.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean) / self.y_std
            return X_normalized, Y_normalized

    def _unnormalize_pred(self, pred_mean, pred_std):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        if self.normalize_data:
            assert pred_mean.ndim == pred_std.ndim == 2 and pred_mean.shape[1] == pred_std.shape[1] == self.output_dim
            if isinstance(pred_mean, torch.Tensor) and isinstance(pred_std, torch.Tensor):
                y_mean_tensor, y_std_tensor = torch.tensor(self.y_mean).float(), torch.tensor(self.y_std).float()
                pred_mean = pred_mean.mul(y_std_tensor[None, :]) + y_mean_tensor[None, :]
                pred_std = pred_std.mul(y_std_tensor[None, :])
            else:
                pred_mean = pred_mean.multiply(self.y_std[None, :]) + self.y_mean[None, :]
                pred_std = pred_std.multiply(self.y_std[None, :])

        return pred_mean, pred_std

    def _initial_data_handling(self, train_x, train_t):
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim, self.output_dim = train_x.shape[-1], train_t.shape[-1]
        self.n_train_samples = train_x.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x = torch.from_numpy(train_x_normalized).contiguous().float().to(device)
        self.train_t = torch.from_numpy(train_t_normalized).contiguous().float().to(device)

        return self.train_x, self.train_t

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)



def _calib_error(pred_dist_vectorized, test_t_tensor):
    cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)

    if test_t_tensor.shape[0] == 1:
        test_t_tensor = test_t_tensor.flatten()
        cdf_vals = cdf_vals.flatten()

    num_points = test_t_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 1.0, 20)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points

    calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels) ** 2))
    return calib_rmse


def _calib_error_chi2(pred_dist_vectorized, test_t_tensor):
    import scipy.stats
    z2 = (((pred_dist_vectorized.mean - test_t_tensor) / pred_dist_vectorized.stddev) ** 2).detach().numpy()
    f = lambda p: np.mean(z2 < scipy.stats.chi2.ppf(p, 1))
    conf_levels = np.linspace(0.05, 1, 20)
    accs = np.array([f(p) for p in conf_levels])
    return np.sqrt(np.mean((accs - conf_levels) ** 2))


