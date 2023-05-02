from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'random',
    'dct',
    'coloring',
    'separator_k1',
    'weighted_separator_k1'
]
nnodes_list = [100, 200, 300, 400, 500]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS,
    dict(degree=4, e_min=2, e_max=5, figname="exp3_type1_alpha0_beta1", weight_type=1, alpha=0, beta=1),
    algorithms=algs,
    overwrite=False
)


