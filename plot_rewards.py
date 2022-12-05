from environment.domain import ContinuousDomain
from environment.kernel import KernelFunction
from environment.reward_generator import KernelGroupSparseMetaEnvironment
import numpy as np
from tueplots import bundles
from config import color_dict

from environment.feature_map import LegendreMap
feature_map = LegendreMap(num_dim_x=1, degree=20)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=6)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))

active_groups = np.random.choice(21, 6, replace=False)
eta = np.array([int(i in active_groups) for i in range(true_kernel.num_groups)])

meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=0.001)
envs = meta_env.sample_envs(5)
from matplotlib import pyplot as plt
for (env, color) in zip(envs, color_dict.keys()):
    x, y = env.generate_samples(num_samples=200)
    idx = np.argsort(x, axis=0)[:, 0]
    x, y = x[idx], y[idx]
    plt.plot(x, y, color = color_dict[color])


plt.rcParams.update(bundles.icml2022())
plt.show()
