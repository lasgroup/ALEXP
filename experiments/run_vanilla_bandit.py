import numpy as np
import argparse
from environment.feature_map import CombOfLegendreMaps
from environment.kernel import KernelFunction
from environment.domain import ContinuousDomain, DiscreteDomain
from algorithms.acquisition import UCB
from algorithms.regression_oracle import RegressionOracle
from experiments.utils import NumpyArrayEncoder
import time
import os
import json
from environment.reward_generator import KernelGroupSparseMetaEnvironment

def main(args):
    t_start = time.time()
    rds = np.random.RandomState(args.seed)
    max_degree = args.sparsity + args.sparse_max_diff

    feature_map = CombOfLegendreMaps(num_dim_x=1, sparsity=args.sparsity, max_degree=max_degree)
    true_kernel = KernelFunction(feature_map=feature_map, sparsity=1)
    domain = ContinuousDomain(l=-np.ones(feature_map.num_dim_x), u=np.ones(feature_map.num_dim_x))
    if args.discrete_domain:
        subsample_domain = np.linspace(domain.l, domain.u, args.mesh_size)
        domain = DiscreteDomain(subsample_domain)

    true_model = rds.randint(0, true_kernel.num_groups)
    eta = np.array([int(i == true_model) for i in range(true_kernel.num_groups)])
    meta_env = KernelGroupSparseMetaEnvironment(true_kernel, domain=domain, eta=eta, noise_std=args.obsv_noise,random_state=rds)
    env = meta_env.sample_envs(1)[0]

    evals = []

    if args.mode =='full':
        oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=args.likelihood_std, eta=np.ones(feature_map.num_groups), random_state=rds)
    elif args.mode =='oracle':
        oracle = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=args.likelihood_std, eta=eta, random_state=rds)
    else:
        raise ValueError('Only full or oracle is accepted.')

    algo = UCB(oracle, env.domain, beta=args.betaUCB, random_state=rds)

    for t in range(args.horizon):
        print(t)
        if t ==0:
            x = algo.explore()
            x_bp = x
        else:
            x = algo.next()
            x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        algo.add_data(evaluation['x'], evaluation['y'])


    t_after = time.time()
    print('time(min)', (t_after-t_start)/60)

    """ save results """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}

    results_dict = {
        'evals': evals_stacked,
        'params': args.__dict__,
        'duration_total': t_after - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        exp_hash = str(abs(json.dumps(results_dict['params'], sort_keys=True).__hash__()))
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla UCB')
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    # model arguments
    parser.add_argument('--sparsity', type=int, default=2)
    parser.add_argument('--sparse_max_diff', type=int, default=1)
    parser.add_argument('--obsv_noise', type=float, default=0.001, help = 'noise in generating data')
    parser.add_argument('--horizon', type=int, default=10, help='number of steps in the online optimizer')
    parser.add_argument('--discrete_domain', type = bool, default=True, help ='should we approximate the domain with a discrete mesh?')
    parser.add_argument('--mesh_size', type = int, default=2000, help ='tunes how fine grained the infinite mesh is')


    parser.add_argument('--mode', type=str, default='full', help='full or oracle')
    parser.add_argument('--likelihood_std', type=float, default=0.01, help = 'the regularization for the GP solvers')
    parser.add_argument('--betaUCB', type=float, default=2, help='balance UCB exploration vs exploitation')
    args = parser.parse_args()
    main(args)