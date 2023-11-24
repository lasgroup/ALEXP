import tueplots.figsizes
from experiments.utils import collect_hparam_results, collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plotting.plot_specs import *
from config import PLOT_DIR


T = 100
sparsity = 3
sparse_max_diff = 7

fig, axes = plt.subplots(ncols=3, figsize=(8.5, 3))
hparams, _ = collect_hparam_results(exp_name='hyper_params', verbose=True, search_depth=0)

df_full, _ = collect_exp_results(exp_name='alexp_probs_discrete', verbose=True, search_depth=1)
df_full = df_full.loc[df_full['horizon'] == T]
df_full = df_full.loc[df_full['sparsity'] == sparsity]
df_full = df_full.loc[df_full['sparse_max_diff'] == sparse_max_diff]
df_full = df_full.loc[df_full['etaEXP'] == hparams['ALEXP']['etaEXP'] ]
df_full = df_full.loc[df_full['gammaEXP'] == hparams['ALEXP']['gammaEXP']]

true_model = np.array(df_full['true_model'])[0]

probs_mid = []
probs_last = []
models = []
true_model_probs = []
true_model_percs = []
perc_covered = []
for index, run in df_full.iterrows():
    probs = np.array(run['probs'])
    probs_mid.append(probs[20])
    total_models = np.shape(np.array(run['probs']))[1]
    probs_last.append(probs[-1])

    true_model_probs.append(probs[:, true_model])

    perc_covered_run = []
    true_model_perc = []
    models = run['chosen_models']
    for i in range(T):
        unique_elements, counts = np.unique(models[0:i+1], return_counts=True)
        perc_covered_run.append(len(unique_elements)/total_models)
        ind = np.where(unique_elements == true_model)[0]
        if len(ind) > 0:
            true_model_perc.append(counts[ind[0]] / (i + 1))
        else:
            true_model_perc.append(0)
    perc_covered.append(perc_covered_run)
    true_model_percs.append(true_model_perc)


## Show mid and last histograms
xlab = np.arange(1, total_models + 1)
axes[0].bar(xlab, np.mean(probs_mid, axis = 0), color=line_color['alexp'], linewidth = 3,ecolor = shade_color['alexp'],capsize=10 )
for i, v in enumerate(np.mean(probs_mid, axis = 0)):
    # determine position of text
    x_pos = axes[0].patches[i].get_x() + axes[0].patches[i].get_width() / 2-0.005
    y_pos = axes[0].patches[i].get_height()-0.01
    # add text to plot
    if i == true_model[0]:
        axes[0].text(x_pos, y_pos, '\u2605', ha='center', va='bottom', color = color_dict['black'] ,fontsize = 10)

axes[0].set_xlabel('(a)', fontsize=14)
axes[0].text(0.65, 0.95, r'$q_{t=20}$', transform=axes[0].transAxes, va='top', ha='left', fontsize = 18, color = color_dict['gray'])


## probability of picking the true model increase
axes[2].plot(np.mean(true_model_probs, axis = 0).squeeze(),linewidth = 3,color=line_color['alexp'] )
axes[2].fill_between(np.arange(T), np.mean(true_model_probs, axis = 0).squeeze()-1/np.sqrt(20) * np.std(true_model_probs, axis=0).squeeze(),
                     np.mean(true_model_probs, axis=0).squeeze() + 1 / np.sqrt(20) * np.std(true_model_probs, axis=0).squeeze(), alpha = 0.2, color = shade_color['alexp'])
axes[2].set_xlabel('(c)', fontsize=14)
axes[2].text(0.05, 0.95, r'$q_{t,j^\star}$', transform=axes[2].transAxes, va='top', ha='left', fontsize = 18,  color = color_dict['gray'])

axes[1].plot(np.mean(perc_covered, axis = 0),linewidth = 3,color=line_color['alexp'] )
axes[1].fill_between(np.arange(T), np.mean(perc_covered, axis = 0)-1/np.sqrt(20) * np.std(perc_covered, axis=0),
                     np.mean(perc_covered, axis=0) + 1 / np.sqrt(20) * np.std(perc_covered, axis=0), alpha = 0.2, color = shade_color['alexp'])

axes[1].set_xlabel('(b)', fontsize=14)
axes[1].text(0.05, 0.95, r'$\frac{M_t}{M}$', transform=axes[1].transAxes, va='top', ha='left', fontsize = 18, color = color_dict['gray'])


axes[0].set_title('MS Distribution', fontsize = 14)
axes[2].set_title('Prob. of Oracle Agent', fontsize = 14)
axes[1].set_title('Ratio of Visited Agents', fontsize = 14)


axes[2].bar(np.NaN, np.NaN, color=line_color['alexp'], label=label['alexp'])
axes[2].legend( loc = 'lower right', fontsize = 14)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()

plot_name = f'/prob_curves_alexp_{sparsity}_{sparsity+sparse_max_diff}.pdf'
plt.savefig(PLOT_DIR+plot_name)

plt.show()