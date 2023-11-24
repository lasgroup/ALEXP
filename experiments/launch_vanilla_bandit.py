from experiments.utils import generate_base_command, generate_run_commands, hash_dict
from config import RESULT_DIR
import run_vanilla_bandit as script_to_run
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'LASSO':['likelihood_std','betaUCB', 'mode'],
    'Feature_Maps': ['sparsity', 'sparse_max_diff'],
    'Exp_setup': [ 'horizon','discrete_domain', 'mesh_size']
}

default_configs = {
    'likelihood_std': 0.5,
    'horizon': 100,
    'betaUCB': 2,
    'sparsity': 2,
    'sparse_max_diff': 8,
    'mode': 'full',
    'discrete_domain': True,
    'mesh_size': 2000
}

search_ranges = {
    'sparse_max_diff': ['range',[7, 11]]
    # 'mode': 'oracle'
}

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}

# def sample_flag(sample_spec, rds=None):
#     if rds is None:
#         rds = np.random
#     assert len(sample_spec) == 2
#
#     sample_type, range = sample_spec
#     if sample_type == 'loguniform':
#         assert len(range) == 2
#         return 10**rds.uniform(*range)
#     elif sample_type == 'uniform':
#         assert len(range) == 2
#         return rds.uniform(*range)
#     elif sample_type == 'choice':
#         return rds.choice(range)
#     else:
#         raise NotImplementedError

def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_haparam < 101
    init_seeds = list(rds.randint(0, 10**6, size=(101,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    # exp_path = os.path.join(exp_base_path, '%s_%s'%(args.feature_map, args.model))
    exp_path = exp_base_path


    command_list = []
    # for _ in range(args.num_hparam_samples):
    # transfer flags from the args
    flags = copy.deepcopy(args.__dict__)
    [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_haparam', 'exp_name', 'num_cpus']]

    # randomly sample flags
    for diff in [7,8,9,10,11]:
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = diff
            else:
                flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)
        # print(flags)

        for j in range(args.num_seeds_per_haparam):
            seed = init_seeds[j]
            cmd = generate_base_command(script_to_run, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='euler', promt=True, dry = False) #euler or local_async

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCB Exps')
    parser.add_argument('--exp_name', type=str,  default='vanilla_ucb_discrete') #required=True,
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_haparam', type=int, default=10)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=734, help='random number generator seed')

    # model arguments
    parser.add_argument('--sparsity', type=int, default=2)
    parser.add_argument('--sparse_max_diff', type=int, default=1)
    parser.add_argument('--obsv_noise', type=float, default=0.001, help='noise in generating data')
    parser.add_argument('--horizon', type=int, default=9, help='number of steps in the online optimizer')
    parser.add_argument('--discrete_domain', type = bool, default=True, help ='should we approximate the domain with a discrete mesh?')
    parser.add_argument('--mesh_size', type = int, default=2000, help ='tunes how fine grained the infinite mesh is')


    parser.add_argument('--mode', type=str, default='full', help='full or oracle')
    parser.add_argument('--likelihood_std', type=float, default=0.01, help = 'the regularization for the GP solvers')
    parser.add_argument('--betaUCB', type=float, default=2, help='balance UCB exploration vs exploitation')


    args = parser.parse_args()
    print(args.num_seeds_per_haparam)
    main(args)


