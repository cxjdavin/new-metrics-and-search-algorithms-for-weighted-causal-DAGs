from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'random',
    'dct',
    'coloring',
    'opt_single',
    'separator_k1',
    'weighted_separator_k1'
]
nnodes_list = [10, 15, 20, 25]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.SHANMUGAM,
    dict(density=.1, figname="exp1_type2_alpha1_beta1", p=0.1, weight_type=2, alpha=1, beta=1),
    algorithms=algs,
    overwrite=False
)


