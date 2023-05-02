from dag_loader import DagSampler
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURE_FOLDER, POLICY2COLOR, POLICY2LABEL
import os
import random
import ipdb
from result_getter import ResultGetter
sns.set()

OVERWRITE_ALL = True


def plot_results_vary_nodes(
        nnodes_list: list,
        ngraphs: int,
        sampler: DagSampler,
        other_params: dict,
        algorithms: set,
        overwrite=False
):
    random.seed(98625472)
    os.makedirs('figures', exist_ok=True)

    rg = ResultGetter(
        algorithms,
        nnodes_list,
        sampler,
        other_params_list=[other_params],
        ngraphs=ngraphs,
    )
    res_df = rg.get_results(overwrite=overwrite)
    mean_generalized_cost = res_df.groupby(level=['alg', 'nnodes'])['generalized_cost'].mean()
    std_generalized_cost = res_df.groupby(level=['alg', 'nnodes'])['generalized_cost'].std()
    mean_time = res_df.groupby(level=['alg', 'nnodes'])['time'].mean()
    std_time = res_df.groupby(level=['alg', 'nnodes'])['time'].std()
    
    algorithms = sorted(algorithms)

    plt.clf()
    for alg in algorithms:
        plt.errorbar(nnodes_list, mean_time[mean_time.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_time[std_time.index.get_level_values('alg') == alg], capsize=5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time (log scale)')
    plt.legend()
    plt.yscale('log')
    plt.xticks(nnodes_list)
    plt.tight_layout()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_time_log.png'.format(other_params['figname'])))

    plt.clf()
    for alg in algorithms:
        plt.errorbar(nnodes_list, mean_generalized_cost[mean_generalized_cost.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_generalized_cost[std_generalized_cost.index.get_level_values('alg') == alg], capsize=5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Generalized interventional cost (log scale)')
    plt.legend()
    plt.yscale('log')
    plt.xticks(nnodes_list)
    plt.tight_layout()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_generalized_cost_log.png'.format(other_params['figname'])))

    plt.clf()
    for alg in algorithms:
        plt.errorbar(nnodes_list, mean_time[mean_time.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_time[std_time.index.get_level_values('alg') == alg], capsize=5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time')
    plt.legend()
    plt.xticks(nnodes_list)
    plt.tight_layout()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_time.png'.format(other_params['figname'])))

    plt.clf()
    for alg in algorithms:
        plt.errorbar(nnodes_list, mean_generalized_cost[mean_generalized_cost.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_generalized_cost[std_generalized_cost.index.get_level_values('alg') == alg], capsize=5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Generalized interventional cost')
    plt.legend()
    plt.xticks(nnodes_list)
    plt.tight_layout()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_generalized_cost.png'.format(other_params['figname'])))

