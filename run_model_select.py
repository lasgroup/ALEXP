import numpy as np

from environment.reward_generator import KernelGroupSparseMetaEnvironment
from environment.feature_map import PolynomialMap, PeriodicMap, LinearMap, LegendreMap, UnionOfFeatureMaps, RFFMap, FilteredFeatureMap
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain
from algorithms.acquisition import UCB
from algorithms.regression_oracle import RegressionOracle
from algorithms.model_selection import MultWeights
import itertools

from matplotlib import pyplot as plt
from tueplots import bundles

from config import color_dict, line_color, shade_color, regret_label, linestyles

sparsity = 2
degree = 15
num_features = degree + 1
# length_scales = [1,0.5,0.2,0.1,0.05]
feature_map = LegendreMap(num_dim_x=1, degree=degree)
# feature_map = RFFMap(num_dim_x=1, lengthscale=length_scales[2], feature_dim=degree)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=sparsity)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
reward_gen = KernelGroupSparseMetaEnvironment(kernel=true_kernel, domain=domain, sparsity=sparsity, noise_std=0.001)
eta = reward_gen.eta
active_groups = reward_gen.active_groups
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for i, color in enumerate(list(color_dict.keys())[0:5]):
    plt.rcParams.update(bundles.icml2022())
    f = reward_gen.sample_envs(num_envs=1)[0]
    if feature_map.num_dim_x == 1:
        x, y = f.generate_samples(num_samples=500)
        idx = np.argsort(x, axis=0)[:, 0]
        x, y = x[idx], y[idx]
        axes[0].plot(x, y, linestyles[i], color=color_dict[color])
        axes[0].set_xlabel(r'$x$')
        axes[0].set_ylabel(r'$f_i(x)$')
        plt.rcParams.update(bundles.icml2022())
    else:
        raise NotImplementedError

reward = reward_gen.sample_envs(num_envs=1)[0]
# print('True Features', active_groups)

print('Shape of eta', eta.shape)
runs = 1
T = 100

# model select
cum_regrets = []
simple_regrets = []

model_inds = [list(i) for i in itertools.combinations(range(degree+1), sparsity)]
# print('choices:', model_inds)

for run in range(runs):
    evals = []
    mw = MultWeights(num_features, model_inds, domain, feature_map, T, likelihood_std=0.01)

    for t in range(T):
        x, x_bp = mw.select()
        evaluation = reward.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        mw.update(evaluation)

    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)
    simple_regrets.append(simple_regret)
    cum_regrets.append(cum_regret)

simple_regrets = np.array(simple_regrets)
cum_regrets = np.array(cum_regrets)
axes[1].plot(np.mean(simple_regrets, axis=0), color=line_color['select'], label=regret_label['select'])
axes[1].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['select'])
axes[2].plot(np.mean(cum_regrets, axis=0), color=line_color['select'], label=regret_label['select'])
axes[2].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['select'])


# Oracle
cum_regrets = []
simple_regrets = []
for run in range(runs):
    evals = []
    oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta=eta)
    algo = UCB(oracle, reward_gen.domain, beta=2.0)

    for t in range(T):
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = reward.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        algo.add_data(evaluation['x'], evaluation['y'])

    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)
    simple_regrets.append(simple_regret)
    cum_regrets.append(cum_regret)

simple_regrets = np.array(simple_regrets)
cum_regrets = np.array(cum_regrets)
axes[1].plot(np.mean(simple_regrets, axis=0), color=line_color['oracle'], label=regret_label['oracle'])
axes[1].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])
axes[2].plot(np.mean(cum_regrets, axis=0), color=line_color['oracle'], label=regret_label['oracle'])
axes[2].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])

# All
cum_regrets = []
simple_regrets = []

feature_maps = []
for inds in model_inds:
    eta_model = np.zeros(num_features)
    eta_model[inds] = 1
    feature_maps.append(FilteredFeatureMap(feature_map=feature_map, eta=eta_model))
stacked_feature_maps = UnionOfFeatureMaps(feature_maps)

for run in range(runs):
    evals = []
    full = RegressionOracle(domain=domain, feature_map=stacked_feature_maps, likelihood_std=0.01, eta=np.ones(stacked_feature_maps.num_groups))
    algo = UCB(full, reward_gen.domain, beta=2.0)

    for t in range(T):
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = reward.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        algo.add_data(evaluation['x'], evaluation['y'])

    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)
    simple_regrets.append(simple_regret)
    cum_regrets.append(cum_regret)

simple_regrets = np.array(simple_regrets)
cum_regrets = np.array(cum_regrets)
axes[1].plot(np.mean(simple_regrets, axis=0), color=line_color['full'], label=regret_label['full'])
axes[1].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['full'])
axes[2].plot(np.mean(cum_regrets, axis=0), color=line_color['full'], label=regret_label['full'])
axes[2].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['full'])

axes[1].set_ylabel('simple regret')
axes[1].set_yscale('log')
axes[1].set_xlabel('t')
axes[2].set_ylabel('cumulative inference regret')
axes[2].set_xlabel('t')
axes[1].legend()
axes[2].legend()
plt.tight_layout()
plt.show()