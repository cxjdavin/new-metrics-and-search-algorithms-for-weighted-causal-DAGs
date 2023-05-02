from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'weighted_separator_k1',
    'weighted_separator_k3',
    'weighted_separator_k5'
]
nnodes_list = list(range(10, 101, 5))
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.SHANMUGAM,
    dict(density=.1, figname="exp4_type2_alpha1_beta1", p=0.1, weight_type=2, alpha=1, beta=1),
    algorithms=algs,
    overwrite=False
)

