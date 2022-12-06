import numpy as np
from environment.feature_map import LegendreMap, FilteredFeatureMap, UnionOfFeatureMaps, ProductOfMaps
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain
from algorithms.regression_oracle import RegressionOracle

from environment.reward_generator import Reward
from matplotlib import pyplot as plt
from config import line_color, shade_color
import itertools


sparsity = 2
degree = 15

# feature_map = LegendreMap(num_dim_x=1, degree=degree)
feature_map1 = LegendreMap(num_dim_x=1, degree=6)
feature_map2 = LegendreMap(num_dim_x=1, degree=3)
feature_map = ProductOfMaps(feature_map1, feature_map2)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=sparsity)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
env = Reward(true_kernel, domain=domain, noise_std=0.1)

x_train, y_train = env.generate_samples(num_samples=10)



model_inds = [list(i) for i in itertools.combinations(range(feature_map.size), sparsity)]
feature_maps = []
for inds in model_inds:
    eta_model = np.zeros(feature_map.size)
    eta_model[inds] = 1
    feature_maps.append(FilteredFeatureMap(feature_map=feature_map, eta=eta_model))
stacked_feature_maps = UnionOfFeatureMaps(feature_maps)


gp_full = RegressionOracle(feature_map=stacked_feature_maps, eta=np.ones(stacked_feature_maps.num_groups),
                           likelihood_std=0.1, domain=domain)
gp_full.add_data(x_train, y_train)

# plot data and true function
x_plot = np.linspace(env.domain.l, env.domain.u, 200)
f = env.f(x_plot)
plt.plot(x_plot, f, label='true f', color = line_color['oracle'])
plt.scatter(x_train, y_train, label='train data', color = shade_color['oracle'])

# plot predictions
pred_mean, pred_std = gp_full.predict(x_plot)
pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()
plt.plot(x_plot, pred_mean, label='naive prediction', color = line_color['full'])

lcb, ucb = gp_full.confidence_intervals(x_plot)
plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2, color = shade_color['full'])


gp_oracle = RegressionOracle(feature_map=feature_map, eta=env.eta,
                           likelihood_std=0.1, domain=domain)
gp_oracle.add_data(x_train, y_train)

# plot predictions
pred_mean, pred_std = gp_oracle.predict(x_plot)
pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()
plt.plot(x_plot, pred_mean, label='oracle prediction', color = line_color['select'])

lcb, ucb = gp_oracle.confidence_intervals(x_plot)
plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2, color = shade_color['select'])


plt.legend()
plt.show()
