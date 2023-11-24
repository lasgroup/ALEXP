import numpy as np

from environment.reward_generator import KernelGroupSparseMetaEnvironment
from environment.feature_map import CombOfLegendreMaps, UnionOfFeatureMaps, FilteredFeatureMap
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain, DiscreteDomain
from algorithms.acquisition import UCB, ESTC
from algorithms.regression_oracle import RegressionOracle
from algorithms.model_selection import LEXP
import itertools

from matplotlib import pyplot as plt
from tueplots import bundles

from plotting.plot_specs import line_color, shade_color, label

#sample the environment
feature_map = CombOfLegendreMaps(num_dim_x=1, sparsity=2, max_degree=3)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=1)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))

subsample_domain = np.linspace(domain.l, domain.u, 2000)
domain = DiscreteDomain(subsample_domain)

true_model = np.random.randint(0, true_kernel.num_groups)
eta = np.array([int(i == true_model) for i in range(true_kernel.num_groups)])
meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=0.001)
env = meta_env.sample_envs(1)[0]


fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

runs = 3
T = 50
likelihood_std = 0.01
lambda_coef = 0.009

# Oracle
cum_regrets = []
simple_regrets = []
for run in range(runs):
    evals = []
    oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta=eta)
    algo = UCB(oracle, env.domain, beta=2.0)

    for t in range(T):
        print(t)
        if t ==0:
            x = algo.explore()
            x_bp = x
        else:
            x = algo.next()
            x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
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
axes[0].plot(np.mean(simple_regrets, axis=0), color=line_color['oracle'], label=label['oracle'])
axes[0].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - (1/np.sqrt(runs))*np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + (1/np.sqrt(runs))*np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])
axes[1].plot(np.mean(cum_regrets, axis=0), color=line_color['oracle'], label=label['oracle'])
axes[1].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - (1/np.sqrt(runs))*np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + (1/np.sqrt(runs))*np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])

# All
cum_regrets = []
simple_regrets = []

for run in range(runs):
    evals = []
    full = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta=np.ones(feature_map.num_groups))
    algo = UCB(full, env.domain, beta=2.0)

    for t in range(T):
        print(t)
        if t ==0:
            x = algo.explore()
            x_bp = x
        else:
            x = algo.next()
            x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
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
axes[0].plot(np.mean(simple_regrets, axis=0), color=line_color['full'], label=label['full'])
axes[0].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - (1/np.sqrt(runs))*np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + (1/np.sqrt(runs))*np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['full'])
axes[1].plot(np.mean(cum_regrets, axis=0), color=line_color['full'], label=label['full'])
axes[1].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - (1/np.sqrt(runs))*np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + (1/np.sqrt(runs))*np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['full'])


# #ETS
cum_regrets = []
simple_regrets = []
for run in range(runs):
    evals = []
    oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta=eta)
    algo = ESTC(domain=domain,feature_map=feature_map, likelihood_std=0.01, banditalg='UCB', lambda_coef=0.009, model_select=True, T0 = 10)

    for t in range(T):
        print(t)
        if t ==0:
            x = algo.explore()
            x_bp = x
        else:
            x = algo.next()
            x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
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
axes[0].plot(np.mean(simple_regrets, axis=0), color=line_color['ets'], label=label['ets'])
axes[0].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - (1/np.sqrt(runs))*np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + (1/np.sqrt(runs))*np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['ets'])
axes[1].plot(np.mean(cum_regrets, axis=0), color=line_color['ets'], label=label['ets'])
axes[1].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - (1/np.sqrt(runs))*np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + (1/np.sqrt(runs))*np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['ets'])

# ETC
cum_regrets = []
simple_regrets = []
for run in range(runs):
    evals = []
    algo = ESTC(domain=domain,feature_map=feature_map, likelihood_std=0.01, banditalg='Greedy', lambda_coef=0.009, model_select=False, T0 = 10)

    for t in range(T):
        print(t)
        if t ==0:
            x = algo.explore()
            x_bp = x
        else:
            x = algo.next()
            x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
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
axes[0].plot(np.mean(simple_regrets, axis=0), color=line_color['etc'], label=label['etc'])
axes[0].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - (1/np.sqrt(runs))*np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + (1/np.sqrt(runs))*np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['etc'])
axes[1].plot(np.mean(cum_regrets, axis=0), color=line_color['etc'], label=label['etc'])
axes[1].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - (1/np.sqrt(runs))*np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + (1/np.sqrt(runs))*np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['etc'])

axes[0].set_ylabel('simple regret')
axes[0].set_yscale('log')
axes[0].set_xlabel('t')
axes[1].set_ylabel('cumulative inference regret')
axes[1].set_xlabel('t')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()