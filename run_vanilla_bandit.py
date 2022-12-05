import numpy as np

from environment.reward_generator import KernelGroupSparseMetaEnvironment
from environment.feature_map import PolynomialMap, PeriodicMap, LinearMap, LegendreMap, UnionOfFeatureMaps, RFFMap
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain
from algorithms.acquisition import UCB
from algorithms.regression_oracle import RegressionOracle
import itertools

from matplotlib import pyplot as plt
from tueplots import bundles

from config import color_dict, line_color, shade_color, regret_label, linestyles

sparsity = 2
degree = 6
#length_scales = [1,0.5,0.2,0.1,0.05]
feature_map = LegendreMap(num_dim_x=1, degree=degree)
# feature_map = RFFMap(num_dim_x=1, lengthscale=length_scales[2], feature_dim=degree)
true_kernel = KernelFunction(feature_map=feature_map, sparsity=sparsity)
domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
reward_gen = KernelGroupSparseMetaEnvironment(kernel=true_kernel, domain=domain, sparsity=sparsity, noise_std=0.001)
eta = reward_gen.eta
active_groups = reward_gen.active_groups
fig, axes = plt.subplots(ncols=3, figsize=(12,4))

for i, color in enumerate(list(color_dict.keys())[0:5]):
    plt.rcParams.update(bundles.icml2022())
    f = reward_gen.sample_envs(num_envs=1)[0]
    if feature_map.num_dim_x == 1:
        x, y = f.generate_samples(num_samples=500)
        idx = np.argsort(x, axis=0)[:, 0]
        x, y = x[idx], y[idx]
        axes[0].plot(x, y, linestyles[i], color = color_dict[color] )
        axes[0].set_xlabel(r'$x$')
        axes[0].set_ylabel(r'$f_i(x)$')
        plt.rcParams.update(bundles.icml2022())
    else:
        raise NotImplementedError

reward = reward_gen.sample_envs(num_envs=1)[0]
print('True Features', active_groups)


runs = 2
T = 50



#model select
cum_regrets = []
simple_regrets = []

# model_inds = [list(i) for i in itertools.combinations(range(degree+1), sparsity)]
# print('choices:', model_inds)
# meta_train_data = reward_gen.generate_uniform_meta_train_data(num_tasks=2, num_points_per_task=2)

for run in range(runs):
    # algos = []
    # models = []
    evals = []
    # for model_ind in model_inds:
    #     eta_model = np.zeros(eta.shape)
    #     eta_model[model_ind] = 1
    #     model = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta = eta_model)
    #     models.append(model)
    # algos = [UCB(model, reward_gen.domain, beta=2.0) for model in models]
    # weights = np.ones(len(models))
    # etaMW = np.sqrt(np.log(len(models))/T)

    oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=0.01, eta=eta)
    algo = UCB(oracle, reward_gen.domain, beta=2.0)
    
    for t in range(T):
        # probs = weights/np.sum(weights)
        # i = np.random.choice(range(len(models)), 1,p=probs)
        # algo = algos[i[0]]
        # print('At T=', t, 'Choosed feaures:',model_inds[i[0]])
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = reward.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        # losses = []
        # for i, algo in enumerate(algos):
            # update algo with new rewards
        algo.add_data(evaluation['x'], evaluation['y'])
            #calculate the loss for new algo
            # losses.append((algo.acquisition(x)-evaluation['y'])**2)

        # update weights
        # losses = np.array(losses).squeeze()
        # weights = np.multiply(weights, (1-etaMW*losses/np.max(losses)))

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
axes[1].plot(np.mean(simple_regrets, axis=0), color=line_color['oracle'], label = regret_label['oracle'])
axes[1].fill_between(np.arange(T), np.mean(simple_regrets, axis=0)-np.std(simple_regrets, axis=0),np.mean(simple_regrets, axis=0)+np.std(simple_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])
axes[2].plot(np.mean(cum_regrets, axis=0), color=line_color['oracle'], label =regret_label['oracle'])
axes[2].fill_between(np.arange(T), np.mean(cum_regrets, axis=0)-np.std(cum_regrets, axis=0), np.mean(cum_regrets, axis=0)+np.std(cum_regrets, axis=0), alpha=0.4,
                     color=shade_color['oracle'])

plt.show()