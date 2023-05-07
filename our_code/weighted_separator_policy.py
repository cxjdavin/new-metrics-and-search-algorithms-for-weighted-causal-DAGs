from causaldag import DAG
import random
import networkx as nx
import numpy as np

from collections import defaultdict
import math
import sys
sys.path.insert(0, './PADS')
import LexBFS

'''
Verify that the peo computed is valid
For any node v, all neighbors that appear AFTER v forms a clique (i.e. pairwise adjacent)
'''
def verify_peo(adj_list, actual_to_peo, peo_to_actual):
    assert len(adj_list) == len(actual_to_peo)
    assert len(adj_list) == len(peo_to_actual)
    try:
        n = len(adj_list)
        for i in range(n):
            v = peo_to_actual[i]
            later_neighbors = [u for u in adj_list[v] if actual_to_peo[u] > i]
            for u in later_neighbors:
                for w in later_neighbors:
                    assert u == w or u in adj_list[w]
    except Exception as err:
        print('verification error:', adj_list, actual_to_peo, peo_to_actual)
        assert False

'''
Compute perfect elimination ordering using PADS
Source: https://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt
'''
def peo(adj_list, nodes):
    n = len(nodes)

    G = dict()
    for v in nodes:
        G[v] = adj_list[v]
    lexbfs_output = list(LexBFS.LexBFS(G))

    # Reverse computed ordering to get actual perfect elimination ordering
    output = lexbfs_output[::-1]
    
    actual_to_peo = dict()
    peo_to_actual = dict()
    for i in range(n):
        peo_to_actual[i] = output[i]
        actual_to_peo[output[i]] = i

    # Sanity check: verify PADS's peo output
    # Can comment out for computational speedup
    #verify_peo(adj_list, actual_to_peo, peo_to_actual)
    
    return actual_to_peo, peo_to_actual

'''
Given a connected chordal graph on n nodes, compute the 1/2-clique graph separator
FAST CHORDAL SEPARATOR algorithm of [GRE84]
Reference: [GRE84] A Separator Theorem for Chordal Graphs
'''
def compute_clique_graph_separator(adj_list, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for v in nodes:
        G.add_edges_from([(v,u) for u in adj_list[v]])
    assert nx.is_connected(G)
    assert nx.is_chordal(G)

    n = len(nodes)

    # Compute perfect elimination ordering via lex bfs
    actual_to_peo, peo_to_actual = peo(adj_list, nodes)

    w = [1] * n
    total_weight = sum(w)

    # Compute separator
    peo_i = 0
    while w[peo_i] <= total_weight/2:
        # w[i] is the weight of the connected component of {v_0, ..., v_i} that contains v_i
        # v_k <- lowest numbered neighbor of v_i with k > i
        k = None
        for j in adj_list[peo_to_actual[peo_i]]:
            if actual_to_peo[j] > peo_i and (k is None or actual_to_peo[j] < actual_to_peo[k]):
                k = j
        if k is not None:
            w[actual_to_peo[k]] += w[peo_i]
        peo_i += 1

    # i is the minimum such that some component of {v_0, ..., v_i} weighs more than total+weight/2
    # C <- v_i plus all of v_{i+1}, ..., v_n that are adjacent to v_i
    C = [peo_to_actual[peo_i]]
    for j in adj_list[peo_to_actual[peo_i]]:
        if actual_to_peo[j] > peo_i:
            C.append(j)
    return C

'''
CliqueIntervention subroutine
Refactored out from [CSB22]'s separator_policy
'''
def clique_intervention(clique_separator_nodes, k):
    assert len(clique_separator_nodes) > 0
    I = []
    if k == 1 or len(clique_separator_nodes) == 1:
        I = [set([v]) for v in clique_separator_nodes]
    else:
        # Setup parameters. Note that [SKDV15] use n and x+1 instead of h and L
        h = len(clique_separator_nodes)
        k_prime = min(k, h/2)
        a = math.ceil(h/k_prime)
        assert a >= 2
        L = math.ceil(math.log(h,a))
        assert pow(a,L-1) < h and h <= pow(a,L)

        # Execute labelling scheme
        S = defaultdict(set)
        for d in range(1, L+1):
            a_d = pow(a,d)
            r_d = h % a_d
            p_d = h // a_d
            a_dminus1 = pow(a,d-1)
            r_dminus1 = h % a_dminus1 # Unused
            p_dminus1 = h // a_dminus1
            assert h == p_d * a_d + r_d
            assert h == p_dminus1 * a_dminus1 + r_dminus1
            for i in range(1, h+1):
                node = clique_separator_nodes[i-1]
                if i <= p_d * a_d:
                    val = (i % a_d) // a_dminus1
                else:
                    val = (i - p_d * a_d) // math.ceil(r_d / a)
                if i > a_dminus1 * p_dminus1:
                    val += 1
                S[(d,val)].add(node)
        I = list(S.values())
    return I

def total_deg_in_interventional_graph(undirected_nx, oriented_arcs, vertices):
    G = nx.Graph(undirected_nx)
    G.remove_edges_from(oriented_arcs)
    return sum([G.degree[v] for v in vertices])

def node_induced_subgraph_is_clique(undirected_nx, vertices):
    n = len(vertices)
    return undirected_nx.subgraph(vertices).size() == n*(n-1)/2

'''
Modified from [CSB22]'s separator_policy

Generalized cost: alpha * w(I) + beta * |I|,
where I is the bounded size verifying set output by algorithm
and each intervention involves at most k vertices

The separator_policy algorithm of [CSB22] is the special case where alpha = 0 and beta = 1
'''
def weighted_separator_policy(dag: DAG, k: int, weights: list, alpha: float, beta: float, verbose: bool = False) -> set:
    intervened_nodes = set()
    current_cpdag = dag.cpdag()

    while current_cpdag.num_arcs != dag.num_arcs:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")

        skel_EG = nx.Graph()
        skel_EG.add_nodes_from(current_cpdag.nodes)
        skel_EG.add_edges_from(current_cpdag.edges)
        skel_EG.add_edges_from(current_cpdag.arcs)
        
        # Cannot directly use G = undirected_portions.to_nx() because it does not first add the nodes
        # We need to first add nodes because we want to check if the clique nodes have incident edges
        # See https://causaldag.readthedocs.io/en/latest/_modules/causaldag/classes/pdag.html#PDAG 
        undirected_portions = current_cpdag.copy()
        undirected_portions.remove_all_arcs()
        G = nx.Graph()
        G.add_nodes_from(undirected_portions.nodes)
        G.add_edges_from(undirected_portions.edges)

        # Compute 1/2-clique separator for each connected component of size >= 2
        for cc_nodes in nx.connected_components(G):
            if len(cc_nodes) == 1:
                continue
            cc = G.subgraph(cc_nodes).copy()
            
            # Map indices of subgraph into 0..n-1
            map_indices = dict()
            unmap_indices = dict()
            for v in cc.nodes():
                map_indices[v] = len(map_indices)
                unmap_indices[map_indices[v]] = v

            # Extract adj_list and nodes of subgraph
            nodes = []
            adj_list = dict()
            for v, nbr_dict in cc.adjacency():
                nodes.append(map_indices[v])
                adj_list[map_indices[v]] = [map_indices[x] for x in list(nbr_dict.keys())]

            #
            # Compute clique separator for this connected component
            #
            clique_separator_nodes = [unmap_indices[v] for v in compute_clique_graph_separator(adj_list, nodes)]

            # Isolate heaviest vertex
            heaviest_vertex = None
            for csn in clique_separator_nodes:
                if heaviest_vertex is None or weights[csn] > weights[heaviest_vertex]:
                    heaviest_vertex = csn
            assert heaviest_vertex is not None
            clique_separator_nodes.remove(heaviest_vertex)

            # Compute bounded size intervention set S of V(K_H) \ {v_H}, then intervene on S
            if len(clique_separator_nodes) > 0:
                S = clique_intervention(clique_separator_nodes, k)
                for intervention in S:
                    assert len(intervention) <= k

                    # If all incident edges already oriented, skip this intervention
                    if total_deg_in_interventional_graph(G, current_cpdag.arcs, intervention) > 0:
                        # Intervene on selected node(s) and update the CPDAG
                        intervention = frozenset(intervention)
                        intervened_nodes.add(intervention)
                        current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)

            #
            # Handle connected components dangling from heaviest_vertex, if necessary
            #
            cc.remove_edges_from(current_cpdag.arcs)
            assert nx.is_chordal(cc)

            # If heaviest_vertex is already a singleton, we're done and there is no "dangling" to handle
            heaviest_vertex_neighbors = set([x for x in cc.neighbors(heaviest_vertex)])
            if len(heaviest_vertex_neighbors) > 0:
                assert cc.degree[heaviest_vertex] > 0
                for H_nodes in nx.connected_components(cc):
                    # Only look at the chain component containing the heaviest vertex
                    if heaviest_vertex in H_nodes:
                        involved_nodes = H_nodes.difference({heaviest_vertex})
                        H = cc.subgraph(involved_nodes).copy()
                        assert nx.is_chordal(H)

                        # Note that H does NOT include the heaviest vertex v and may be disjoint due to removal of v
                        # Loop over each chain component H_1, ... , H_t in H
                        cost_for_not_intervening_on_heaviest = 0
                        for H_i_nodes in nx.connected_components(H):
                            # Narrow down to the subgraph that involves neighbors of heaviest vertex
                            N_i = H.subgraph(H_i_nodes.intersection(heaviest_vertex_neighbors))
                            assert nx.is_connected(N_i)
                            assert nx.is_chordal(N_i)

                            best = 0
                            for maximal_clique in nx.chordal_graph_cliques(N_i):
                                mc = [c for c in maximal_clique]
                                mc_weight = sum(np.array(weights)[mc])
                                current = alpha * mc_weight + beta * len(mc)
                                best = max(best, current)
                            cost_for_not_intervening_on_heaviest += best
                        
                        # Decide whether to intervene on heaviest or within each dangling H_i
                        cost_for_intervening_on_heaviest = alpha * weights[heaviest_vertex] + beta
                        if cost_for_intervening_on_heaviest <= cost_for_not_intervening_on_heaviest:
                            # Intervene on heaviest
                            assert total_deg_in_interventional_graph(G, current_cpdag.arcs, {heaviest_vertex}) > 0
                            intervention = frozenset({heaviest_vertex})
                            intervened_nodes.add(intervention)
                            current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)
                        else:
                            # Loop over each chain component H_1, ... , H_t in H
                            for H_i_nodes in nx.connected_components(H):
                                V_prime = H_i_nodes.intersection(heaviest_vertex_neighbors)

                                while not node_induced_subgraph_is_clique(skel_EG, V_prime) or len(V_prime) > k:
                                    current_H_i = H.subgraph(V_prime).copy()
                                    current_H_i.remove_edges_from(current_cpdag.arcs)
                                    assert nx.is_connected(current_H_i)
                                    assert nx.is_chordal(current_H_i)

                                    #
                                    # Compute clique separator for H_i[V']
                                    #

                                    # Map indices of subgraph into 0..v_prime_n-1
                                    V_prime_map_indices = dict()
                                    V_prime_unmap_indices = dict()
                                    for v in current_H_i.nodes():
                                        V_prime_map_indices[v] = len(V_prime_map_indices)
                                        V_prime_unmap_indices[V_prime_map_indices[v]] = v

                                    V_prime_nodes = []
                                    V_prime_adj_list = dict()
                                    for v, nbr_dict in current_H_i.adjacency():
                                        V_prime_nodes.append(V_prime_map_indices[v])
                                        V_prime_adj_list[V_prime_map_indices[v]] = [V_prime_map_indices[x] for x in list(nbr_dict.keys())]

                                    # Compute clique separator for this connected component
                                    V_prime_clique_separator_nodes = [V_prime_unmap_indices[v] for v in compute_clique_graph_separator(V_prime_adj_list, V_prime_nodes)]

                                    #
                                    # Arbitrarily partition into sets involving <= k vertices each, then intervene on them
                                    #
                                    S = [V_prime_clique_separator_nodes[k*i:k*i+k] for i in range(len(V_prime_clique_separator_nodes))]
                                    for intervention in S:
                                        assert len(intervention) <= k

                                        # If all incident edges already oriented, skip this intervention
                                        if total_deg_in_interventional_graph(G, current_cpdag.arcs, intervention) > 0:
                                            # Intervene on selected node(s) and update the CPDAG
                                            intervention = frozenset(intervention)
                                            intervened_nodes.add(intervention)
                                            current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)

                                    #
                                    # Identify source set S_source within S
                                    # The source of the clique has no incoming arcs from all other vertices in the clique
                                    #
                                    S_source = None
                                    for S_i in S:
                                        for candidate in S_i:
                                            found_source = True
                                            for u,v in current_cpdag.arcs:
                                                if u in V_prime_clique_separator_nodes and v == candidate:
                                                    found_source = False
                                                    break
                                            if found_source:
                                                S_source = S_i
                                                break
                                    assert S_source is not None

                                    #
                                    # Determine if there is a connected component Q that only has arcs into S_source. If it does, find it
                                    # There is at most one such connected component Q
                                    #
                                    Q = None
                                    current_H_i.remove_edges_from(current_cpdag.arcs)
                                    for Q_nodes in nx.connected_components(current_H_i):
                                        arcs_into_S_source = False
                                        arcs_from_S_source = False
                                        for u,v in current_cpdag.arcs:
                                            if u in S_source and v in S_source:
                                                continue
                                            if u in Q_nodes and v in S_source:
                                                arcs_into_S_source = True
                                            if u in S_source and v in Q_nodes:
                                                arcs_from_S_source = True
                                        if arcs_into_S_source and not arcs_from_S_source:
                                            assert Q is None
                                            Q = set(Q_nodes)

                                    #
                                    # Update V_prime
                                    #
                                    if Q is not None:
                                        V_prime = Q
                                    else:
                                        V_prime = S_source

                                #
                                # Intervene on the clique V_prime using CliqueIntervention
                                #
                                assert node_induced_subgraph_is_clique(skel_EG, V_prime)
                                assert len(V_prime) <= k
                                for intervention in clique_intervention(list(V_prime), k):
                                    assert len(intervention) <= k
                
                                    # If all incident edges already oriented, skip this intervention
                                    if total_deg_in_interventional_graph(G, current_cpdag.arcs, intervention) > 0:
                                        # Intervene on selected node(s) and update the CPDAG
                                        intervention = frozenset(intervention)
                                        intervened_nodes.add(intervention)
                                        current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)
    return intervened_nodes
