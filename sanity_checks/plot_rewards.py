from environment.domain import ContinuousDomain
from environment.kernel import KernelFunction
from environment.reward_generator import KernelGroupSparseMetaEnvironment
import numpy as np
from tueplots import bundles
from plotting.plot_specs import color_dict, linestyles, line_color
from environment.feature_map import LegendreMap, ProductOfMaps, CombOfLegendreMaps
from config import PLOT_DIR
import os

# feature_map = LegendreMap(num_dim_x=1, degree=20)
feature_map1 = CombOfLegendreMaps(num_dim_x=1, sparsity=3, max_degree=10)
# feature_map2 = LegendreMap(num_dim_x=1, degree=3)
# feature_map = ProductOfMaps(feature_map1, feature_map2)
true_kernel = KernelFunction(feature_map=feature_map1, sparsity=1)
domain = ContinuousDomain(l=-np.ones(feature_map1.num_dim_x), u=np.ones(feature_map1.num_dim_x))

true_model = np.random.randint(0, true_kernel.num_groups)
print(feature_map1.combinations[true_model])
eta = np.array([int(i == true_model) for i in range(true_kernel.num_groups)])
meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=0.001)
envs = meta_env.sample_envs(5)
counter = 0
from matplotlib import pyplot as plt
for (env, color) in zip(envs, line_color.keys()):
    x, y = env.generate_samples(num_samples=200)
    idx = np.argsort(x, axis=0)[:, 0]
    x, y = x[idx], y[idx]
    plt.plot(x, y, color = line_color[color], linestyle = linestyles[counter], linewidth = 2)
    counter += 1

plot_name = f'/LegendrePolies_{3}_{10}.pdf'
plt.rcParams.update(bundles.neurips2021(nrows=1))
print(os.path.join(PLOT_DIR, plot_name))
plt.savefig(PLOT_DIR+plot_name)
plt.show()
