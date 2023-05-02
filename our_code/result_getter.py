import itertools as itr
from dag_loader import DagLoader
from alg_runner import AlgRunner
import pandas as pd
import math

from verify import *

class ResultGetter:
    def __init__(self, algs, nnodes_list, sampler, other_params_list, ngraphs=100, comparable_edges=True):
        self.algs = algs
        self.nnodes_list = nnodes_list
        self.other_params_list = other_params_list
        self.sampler = sampler
        self.ngraphs = ngraphs
        self.dag_loaders = [
            DagLoader(nnodes, self.ngraphs, self.sampler, other_params, comparable_edges=comparable_edges)
            for nnodes, other_params in itr.product(self.nnodes_list, self.other_params_list)
        ]

    def get_results(self, overwrite=False):
        results = []

        for alg in self.algs:
            for dl in self.dag_loaders:
                ar = AlgRunner(alg, dl)
                generalized_cost_list, times_list = ar.get_alg_results(overwrite=overwrite)

                # Compute alpha * wt{nu}_1(G^*) + beta * ceil(nu_1(G^*)/k) as lower bound benchmark
                k = 1
                if 'separator' in alg:
                    k = int(alg[-1])
                benchmark_list = []
                for dag, w, alpha, beta in list(zip(*dl.get_dags())):
                    tilde_nu = len(atomic_verification(dag.to_nx(), w))
                    nu = len(atomic_verification(dag.to_nx()))
                    benchmark_list.append(alpha * tilde_nu + beta * math.ceil(nu/k))

                for gc, time, benchmark in zip(generalized_cost_list, times_list, benchmark_list):
                    results.append(dict(
                        alg=alg,
                        nnodes=dl.nnodes,
                        **dl.other_params,
                        generalized_cost=gc,
                        time=time,
                        benchmark=benchmark
                    ))

        res_df = pd.DataFrame(results)
        res_df = res_df.set_index(list(set(res_df.columns) - {'generalized_cost', 'time', 'benchmark'}))
        return res_df


