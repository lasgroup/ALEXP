import tueplots.figsizes
from experiments.utils import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plotting.plot_specs import *
import seaborn as sns



df_full, _ = collect_exp_results(exp_name='vanilla_ucb_discrete', verbose=True, search_depth=1)

# set the experiment setting you would like to see:
T = 100
sparsity = 2
sparse_max_diff = 8

df_full = df_full.loc[df_full['horizon'] == T]
fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
plot_name = 'vanilla_UCB_{}T_{}s'.format(T, sparsity)

for mode in ['oracle', 'full']:
    df_mode = df_full.loc[df_full['mode']==mode]
    configurations = [config for config in zip(df_mode['sparsity'], df_mode['sparse_max_diff']) if config[0]+config[1] ==10]
    #
    configurations = list(set(configurations))
    clrs = sns.color_palette('husl', n_colors=len(configurations))
    for i, config in enumerate(configurations):
        if config[0]+config[1] == 10:
            df = df_mode.loc[df_mode['sparse_max_diff'] == config[1]]
            df = df.loc[df['sparsity'] == config[0]]
            curve_name = label[mode] + r' $s= $' + '{}'.format(config[0]) + r'$-p = $' + '{}'.format(config[1]+config[0])
                # for index, run in df_full.iterrows():
            y_exact = np.array([np.squeeze(np.array(a)) for a in df['y_exact']])
            y_min = np.array([np.squeeze(np.array(a)) for a in df['y_min']])
            y_exact_bp = np.array([np.squeeze(np.array(a)) for a in df['y_exact_bp']])
            regret = y_exact - y_min
            regret_bp = y_exact_bp- y_min

            simple_regret = np.minimum.accumulate(regret, axis=-1)
            cum_regret = np.cumsum(regret, axis=-1)
            cum_regret_bp = np.cumsum(regret_bp, axis=-1)


            axes[0].plot(np.mean(simple_regret, axis=0), color = clrs[i], label = curve_name, linestyle = line_styles[mode])
            axes[1].plot(np.mean(cum_regret, axis = 0), color =clrs[i], label = curve_name, linestyle = line_styles[mode])
            axes[1].fill_between(np.arange(T), np.mean(cum_regret, axis=0)-0.5*np.std(cum_regret, axis=0),np.mean(cum_regret, axis=0)+0.5*np.std(cum_regret, axis=0), alpha=0.2,
                                                    color=clrs[i])

axes[0].set_xlabel(r'$t$')
axes[0].set_title('Simple Regret')
axes[1].set_xlabel(r'$t$')
axes[1].set_title('Cumulative Regret')
axes[1].legend(loc = 'upper center', bbox_to_anchor=(1.2, 1), fontsize = 12, ncol = 1)



plt.tight_layout()
# plt.savefig(f'/local/pkassraie/modelselect/plots/{plot_name}.pdf')
plt.show()