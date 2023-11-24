from experiments.utils import generate_base_command, generate_run_commands, hash_dict, collect_hparam_results
from config import RESULT_DIR
import run_probs as script_to_run
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'LASSO':['likelihood_std','lambda_coef', 'delta', 'oracle_or_lasso'],
    'LEXP':['gammaEXP', 'etaEXP', 'adaptive'],
    'Feature_Maps': ['sparsity', 'sparse_max_diff'],
    'Exp_setup': [ 'horizon', 'discrete_domain', 'mesh_size']
}

hparams, _ = collect_hparam_results(exp_name='hyper_params', verbose=True, search_depth=0, T = 100, sparsity= 3, total = 10)

default_configs = {
    'delta': 0.1,
    'likelihood_std': 0.01,
    'horizon': 100,
    'lambda_coef': 0.009,
    'sparsity': 3,
    'sparse_max_diff': 7,
    'gammaEXP': hparams['ALEXP']['gammaEXP'],
    'etaEXP':  hparams['ALEXP']['etaEXP'],
    'oracle_or_lasso': 'lasso',
    'discrete_domain': True,
    'mesh_size': 2000,
    'adaptive': True
}

search_ranges = {
    # 'lambda_coef':['uniform',[0.005, 0.01]],
    # 'gammaEXP': ['loguniform', [-4, -1]],
    # 'etaEXP': ['loguniform', [0, 2]],
}

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}

def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10**rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    else:
        raise NotImplementedError

def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_haparam < 101
    init_seeds = list(rds.randint(0, 10**6, size=(101,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    # exp_path = os.path.join(exp_base_path, '%s_%s'%(args.feature_map, args.model))
    exp_path = exp_base_path


    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_haparam', 'exp_name', 'num_cpus']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                # print(flag)
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]
                print(flags)

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)
        # print(flags)

        for j in range(args.num_seeds_per_haparam):
            seed = init_seeds[j]
            cmd = generate_base_command(script_to_run, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='euler', promt=True,  duration='3:59:00') #euler or local_async

if __name__ == '__main__':
    # experiments
    parser = argparse.ArgumentParser(description='Tuning Exp')
    parser.add_argument('--exp_name', type=str,  default='alexp_virtual_discrete') #required=True,
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_haparam', type=int, default=20)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=734, help='random number generator seed')
    parser.add_argument('--seed_env', type=int, default=834, help='random number generator seed')
    parser.add_argument('--fixed_oracle', type = bool, default=False, help = 'if true, sets the oracle model the same across runs, else, randomizes it')

    # model arguments
    parser.add_argument('--sparsity', type=int, default=2)
    parser.add_argument('--sparse_max_diff', type=int, default=1)
    parser.add_argument('--obsv_noise', type=float, default = 0.001, help='noise in generating data')
    parser.add_argument('--horizon', type=int, help='number of steps in the online optimizer')
    parser.add_argument('--discrete_domain', type = bool,  help ='should we approximate the domain with a discrete mesh?')
    parser.add_argument('--mesh_size', type = int, help ='tunes how fine grained the infinite mesh is')


    parser.add_argument('--gammaEXP', type=float, default = 1, help = 'tunes the rate of exploration for LEXP')
    parser.add_argument('--etaEXP', type=float,default = 1, help = 'tunes the learning rate of LEXP ')
    parser.add_argument('--adaptive', type=bool, default=False, help='adaptively change gamma and eta?')

    parser.add_argument('--lambda_coef', type=float, default=0.009,
                        help='The regularization of hte lasso estimator is multiplied by this coefficient. settign to 1 gives the theoretical rate')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='the lasso bounds are valid with probability greater than 1-delta.')
    parser.add_argument('--likelihood_std', type=float, default=0.01, help='the regularization for the GP solvers')
    parser.add_argument('--oracle_or_lasso', type=str, default='oracle', help = 'runs LEXP with oracle reward estimator')

    args = parser.parse_args()
    print(args.num_seeds_per_haparam)
    main(args)


