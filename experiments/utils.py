import sys
import os
import json
import numpy as np
import glob
import pandas as pd
from typing import List, Optional
from itertools import cycle
from config import BASE_DIR, RESULT_DIR

""" Gather exp results """

def collect_exp_results(exp_name, verbose=True, search_depth: int = 2):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0

    exp_dicts = []
    param_names = set()
    file_regex = '/' + ''.join(['*/' for _ in range(search_depth)]) + '*.json'
    for results_file in glob.glob(exp_dir + file_regex):

        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                exp_dicts.append({**exp_dict['evals'], **exp_dict['params'],** {'duration_total' : exp_dict['duration_total']}})
                param_names = param_names.union(set(exp_dict['params'].keys()))
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            exp_name, len(exp_dicts), no_results_counter))

    return pd.DataFrame(data=exp_dicts), list(param_names)

def collect_hparam_results(exp_name, verbose=True, search_depth: int = 2, T=100, sparsity=3, total = 10):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0

    exp_dicts = []
    param_names = set()
    # file_regex = '/' + ''.join(['*/' for _ in range(search_depth)]) + '*.json'
    file_regex = f'/configs_{sparsity}_{total}_{T}.json'
    print(file_regex)
    for results_file in glob.glob(exp_dir + file_regex):

        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            exp_name, len(exp_dicts), no_results_counter))

    return exp_dict, list(param_names)

""" Async executer """
import multiprocessing

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass


""" Command generators """

def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd

def cmd_exec_fn(cmd):
    import os
    os.system(cmd)

def generate_run_commands(command_list: List[str], output_file_list: Optional[List[str]] = None,
                          num_cpus: int = 1, num_gpus: int = 0,
                          dry: bool = False, mem: int = 2 * 1028, duration: str = '3:59:00',
                          mode: str = 'local', promt: bool = True) -> None:

    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'sbatch ' + \
                   f'--time={duration} ' + \
                   f'--mem-per-cpu={mem} ' + \
                   f'-n {num_cpus} '

        if num_gpus > 0:
            bsub_cmd += f'-G {num_gpus} '

        assert output_file_list is None or len(command_list) == len(output_file_list)
        if output_file_list is None:
            for cmd in command_list:
                cluster_cmds.append(bsub_cmd + f'--wrap="{cmd}"')
        else:
            for cmd, output_file in zip(command_list, output_file_list):
                cluster_cmds.append(bsub_cmd + f'--output={output_file} --wrap="{cmd}"')

        if dry:
            for cmd in cluster_cmds:
                print(cmd)
        else:
            if promt:
                answer = input(f"about to launch {len(command_list)} jobs with {num_cpus} cores each. proceed? [yes/no]")
            else:
                answer = 'yes'
            if answer == 'yes':
                for cmd in cluster_cmds:
                    print(cmd)
                    os.system(cmd)

    elif mode == 'local':
        if promt:
            answer = input(f"about to run {len(command_list)} jobs in a loop. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(f"about to launch {len(command_list)} commands in {num_cpus} local processes. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                cmd_exec_fun = lambda cmd: os.system(cmd)
                exec.run(cmd_exec_fun, command_list)
    else:
        raise NotImplementedError

""" Hashing and Encoding dicts to JSON """

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def hash_dict(d):
    return str(abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))