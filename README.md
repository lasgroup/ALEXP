# ALEXP: Anytime Model Selection for Linear Bandits

<p align="center">
<img src="https://github.com/lasgroup/ALEXP/blob/main/Thumbnail.jpg?raw=true" width="400">
</p>
This repository contains the code to our paper <a href=https://arxiv.org/abs/2307.12897> Anytime Model Selection for Linear Bandits</a>, NeurIPS 2023.
We propose ALEXP, an algorithm for simultanous online optimization and feature/model selection. ALEXP is a method of probabilistic aggregation for online model selection and maintains an adaptive probability distribution based on which it iteratively samples features/models. The sampled model is then used to estimate the objective function, as a proxy for the optimization objective.
For a brief overview you can watch the <a href=https://recorder-v3.slideslive.com/?share=89684&s=186c9e74-31fe-4cf1-a3c1-1d99b41b0644> NeurIPS teaser video</a> or read the <a href=https://pkassraie.github.io/assets/talks/rss_2023.pdf>slides of our RSS talk</a>.

## Contents and Setup

This is a torch-based repository, `requirements.txt` lists the needed packages/used versions.

### Algorithms
The repository contains the implementation of a few model selection algorithms. The classes are implemented as follows:
- Adaptive Algorithms: ALEXP and <a href=https://arxiv.org/abs/1612.06246>CORRAL</a> (based on an <a href=https://github.com/pacchiano/modelselection/blob/main/algorithmsmodsel.py>earlier implementation</a> by <a href=https://www.aldopacchiano.ai/>Aldo</a>), located in `algorithms/model_selection.py`.
- Non-adaptive algorithms: ETC/ETS (Explore then Commit/Explore the Select), located in `algorithms/acquisition.py`.
Naturally, we have also implemented a bandit optimization algorithms, mainly GP-UCB and Greedy, located in `algorithms/acquisition.py`.

### Dataset
In the paper, we present experiments based on synthetic data, where for every run of the algorithm, we sample a new random environment.
This repository also includes the code to generate random environments.
The folder `Environment` contains the classes for Domain, Reward and Kernel functions, which together make up the environment. 
To sample an environment, you will need to create an instance of `MetaEnvironment` and run the method `sample_envs()`. For more details, see `environment/reward_generator.py`.
Alternatively, you can write a data-loader that maps any given dataset into the domain and reward classes defined in the repository. 

### Running experiments

The main file for running and testing ALEXP is `experiments/run_lexp_hparam.py`. The same script may be used for running CORRAL or ETC/ETS by changing the input arguments. Similarly, you can try our different base bandit solvers (among UCB or Greedy). To see the training dynamics of your algorithm, you can alternatively run `run_probs.py` which runs ALEXP, keeps track of the selected models and saves the data used for creating plots such as Figure 4 in the paper.

**Launching large-scale experiments:** The launcher scripts in `experiments` allow you to run the different instances of the experiments (e.g. different hyper-parameters or different random seeds) in parallel on a cluster. The bash commands are detailed in `experiments/utils.py`, you should modify them based on your cluster job scheduling system.

**Tests:** The folder `sanity_checks` includes the scripts for some simple test to verify the correctness of individual modules in the code. For instance, `test_vanilla_bandits.py` checks the bandit optimization algorithm, given oracle knowledge of the model, or `run_online_lasso.py` checks the quality of our online lasso regression oracle.

**Saving the results:** The experiment scripts, i.e. `run_*.py` or `launcher_*.py` all automatically save results in json files containing dictionaries (often called `results_dict`) with all relevant parameters of the experiments and the action-reward pairs, runtime, etc. File names are automatically hashed based on experiment parameters and the time of running the experiment and saved in a `results` folder. To read the results, you can use the `collect_exp_results()` method, detailed in `experiments/utils.py` which searches through the `results` folder and creates a dataframe based on the aforementioned dictionaries. You can then filter this dataframe to omit the result of some experiments.

**Plotting:** The folder `plotting` includes some scripts to plot the results, for instance `plot_regret.py` should give you an idea to how to read the results and plot them.

The pipeline for launching large-scale experiments, storing and reading the result files is based on the work of Jonas Rothfuss (e.g. in <a href=https://github.com/jonasrothfuss/meta_learning_pacoh>this repository</a>).

## Reference and Contact
You can contact <a href=https://pkassraie.github.io/>Parnian</a> if you have questions.
If you find the code useful, please cite our paper:

```
@inproceedings{kassraie2023anytime,
  title={Anytime Model Selection in Linear Bandits},
  author={Kassraie, Parnian and  Emmenegger, Nicolas and Krause, Andreas and Pacchiano, Aldo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
