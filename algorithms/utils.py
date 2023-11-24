import numpy as np

def _handle_input_dimensionality(x, y=None):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)

    assert x.ndim == 2

    if y is not None:
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 2

        return x, y
    else:
        return x
    #
    # def _calib_error(pred_dist_vectorized, test_t_tensor):
    #     cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)
    #
    #     if test_t_tensor.shape[0] == 1:
    #         test_t_tensor = test_t_tensor.flatten()
    #         cdf_vals = cdf_vals.flatten()
    #
    #     num_points = test_t_tensor.shape[0]
    #     conf_levels = torch.linspace(0.05, 1.0, 20)
    #     emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points
    #
    #     calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels) ** 2))
    #     return calib_rmse
    #
    # def _calib_error_chi2(pred_dist_vectorized, test_t_tensor):
    #     import scipy.stats
    #     z2 = (((pred_dist_vectorized.mean - test_t_tensor) / pred_dist_vectorized.stddev) ** 2).detach().numpy()
    #     f = lambda p: np.mean(z2 < scipy.stats.chi2.ppf(p, 1))
    #     conf_levels = np.linspace(0.05, 1, 20)
    #     accs = np.array([f(p) for p in conf_levels])
    #     return np.sqrt(np.mean((accs - conf_levels) ** 2))
