import numpy as np
from matplotlib import pyplot as plt

from algorithms.model_selection import LEXP, Corral
from environment.domain import ContinuousDomain
from environment.feature_map import CombOfLegendreMaps
from environment.kernel import KernelFunction
from environment.reward_generator import KernelGroupSparseMetaEnvironment
from plotting.plot_specs import line_color, shade_color, label

#sample the environment
feature_map = CombOfLegendreMaps(num_dim_x=1, sparsity=3, max_degree=5)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=1)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
true_model = np.random.randint(0, true_kernel.num_groups)
print(true_model)
eta = np.array([int(i == true_model) for i in range(true_kernel.num_groups)])
meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=0.001)
env = meta_env.sample_envs(1)[0]

runs = 1
T = 15
likelihood_std = 0.01
lambda_coef = 0.009

# model select
cum_regrets = []
simple_regrets = []

for run in range(runs):
    evals = []
    models = []
    # algo = LEXP(domain = domain, feature_map = feature_map, T = T, likelihood_std =  likelihood_std,
    #              banditalg = 'UCB', lambda_coef = lambda_coef, theta_oracle=env.beta)
    algo = Corral(domain = domain, feature_map = feature_map, T=T, likelihood_std = likelihood_std)

    for t in range(T):
        print(t)
        x, x_bp, _ = algo.select()
        evaluation = env.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        models.append(algo.chosen_model)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        algo.update(evaluation)

        # algo.visualize_probs(true_model)

    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    evals_stacked['chosen_models'] = models
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)
    simple_regrets.append(simple_regret)
    cum_regrets.append(cum_regret)

simple_regrets = np.array(simple_regrets)
cum_regrets = np.array(cum_regrets)
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].plot(np.mean(simple_regrets, axis=0), color=line_color['ours'], label=label['ours'])
axes[0].fill_between(np.arange(T), np.mean(simple_regrets, axis=0) - np.std(simple_regrets, axis=0),
                     np.mean(simple_regrets, axis=0) + np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['ours'])
axes[1].plot(np.mean(cum_regrets, axis=0), color=line_color['ours'], label=label['ours'])
axes[1].fill_between(np.arange(T), np.mean(cum_regrets, axis=0) - np.std(cum_regrets, axis=0),
                     np.mean(cum_regrets, axis=0) + np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['ours'])

plt.show()