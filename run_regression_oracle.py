import numpy as np
from environment.feature_map import LegendreMap
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain
from algorithms.regression_oracle import RegressionOracle

from environment.reward_generator import Reward
from matplotlib import pyplot as plt
from config import color_dict

feature_map = LegendreMap(num_dim_x=1, degree=8)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=2)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
env = Reward(true_kernel, domain=domain, noise_std=0.1)

x_train, y_train = env.generate_samples(num_samples=10)

gp_mll = RegressionOracle(feature_map=feature_map, eta=np.ones(true_kernel.num_groups),
                                likelihood_std=0.1, domain=domain)
gp_mll.add_data(x_train, y_train)

# plot data and true function
x_plot = np.linspace(env.domain.l, env.domain.u, 200)
f = env.f(x_plot)
plt.plot(x_plot, f, label='true f', color = color_dict['night'])
plt.scatter(x_train, y_train, label='train data', color = color_dict['morning'])

# plot predictions
pred_mean, pred_std = gp_mll.predict(x_plot)
pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()
plt.plot(x_plot, pred_mean, label='prediction', color = color_dict['blood'])

lcb, ucb = gp_mll.confidence_intervals(x_plot)
plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2, color = color_dict['blood'])

plt.legend()
plt.show()
