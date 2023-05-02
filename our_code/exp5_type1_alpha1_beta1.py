from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'weighted_separator_k1',
    'weighted_separator_k3',
    'weighted_separator_k5'
]
nnodes_list = [100, 200, 300, 400, 500]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS,
    dict(degree=40, e_min=20, e_max=50, figname="exp5_type1_alpha1_beta1", weight_type=1, alpha=1, beta=1),
    algorithms=algs,
    overwrite=False
)


