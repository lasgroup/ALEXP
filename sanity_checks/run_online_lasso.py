import numpy as np
from environment.feature_map import CombOfLegendreMaps
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain
from algorithms.regression_oracle import RegressionOracle, LassoOracle

from environment.reward_generator import KernelGroupSparseMetaEnvironment
from matplotlib import pyplot as plt
from plotting.plot_specs import line_color, shade_color, label
import itertools



feature_map = CombOfLegendreMaps(num_dim_x=1, sparsity=2, max_degree=4)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=1)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))

true_model = np.random.randint(0, true_kernel.num_groups)
# print(true_model)
eta = np.array([int(i == true_model) for i in range(true_kernel.num_groups)])
meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=0.001)
env = meta_env.sample_envs(1)[0]




# plot data and true function
x_plot = np.linspace(env.domain.l, env.domain.u, 200)
f = env.f(x_plot)
plt.plot(x_plot, f, label='true f', color = line_color['oracle'])



gp_oracle = LassoOracle(feature_map=feature_map,lambda_coef=0.009,
                           likelihood_std=0.1, domain=domain)
full_gp = RegressionOracle(feature_map=feature_map, domain=domain,likelihood_std=0.1, eta = np.ones(true_kernel.num_groups), )
x, y = env.generate_samples(num_samples=1)
gp_oracle.add_data(x, y)

pred_error = []
pred_error_full = []
est_error = []

for i in range(50):
    x, y = env.generate_samples(num_samples=1)
    gp_oracle.add_data(x, y)
    full_gp.add_data(x,y)
    gp_oracle.fit_lasso()
    # print(np.where(gp_oracle.eta>0))
    # phi_x = feature_map.get_feature(x_plot)
    # pred_mean = np.dot(phi_x, gp_oracle.theta_hat.T)
    pred_mean, _ = gp_oracle.predict(x_plot)
    # pred_mean = gp_oracle.celer_predict(x_plot)
    if i > 0 and i % 10==0:
        plt.plot(x_plot, pred_mean, label='Step'+str(i))
    pred_error.append(np.sqrt(np.linalg.norm(pred_mean - f, ord=2)))
    est_error.append(np.sqrt(np.linalg.norm(gp_oracle.theta_hat - env.beta, ord=2)))

    pred_mean, _ = full_gp.predict(x_plot)
    pred_error_full.append(np.sqrt(np.linalg.norm(pred_mean - f, ord = 2)))
    # plt.plot(x_plot, pred_mean, label='LASSO', color = '#4f7992')
window = np.ones(5)/5
smooth_error = np.convolve(pred_error, window, mode='valid')
smooth_error_full = np.convolve(pred_error_full, window, mode='valid')

plt.legend()
plt.show()

plt.figure()
plt.plot(smooth_error, color = line_color['ours'], label = label['ours'])
plt.plot(smooth_error_full, color = line_color['full'], label = label['full'])
plt.legend()
plt.show()