import simple_wta
import networkx as nx
import numpy as np

def survival_value(prob: simple_wta.WTAProblem, S: list, j: int):
        return prob.v[j]*np.prod((1-prob.p[S[j],j]))

def improvement_graph(prob: simple_wta.WTAProblem, S: list, T: list):
    (n,m) = prob.p.shape
    G = nx.DiGraph()
    edges = []
    for r in range(n):
        for l in range(n):
            if T[r] != T[l]:
                j = T[l]
                qr = 1-prob.p[r,j]
                ql = 1-prob.p[l,j]
                vjp = survival_value(prob,S,j)
                crl = vjp*(qr/ql-1)
                edges += [(r,l,crl)]
    G.add_weighted_edges_from(edges)
    return G

def multiexchange(cycle,S,T):
    s0 = cycle[0]
    t0 = T[s0]
    for i in range(len(cycle)-2):
        r = cycle[i]
        l = cycle[i+1]
        old_t = T[r]
        S[old_t].remove(r)
        new_t = T[l]
        T[r] = new_t
        S[new_t].append(r)
    r = cycle[-2]
    S[T[r]].remove(r)
    T[r] = t0
    S[t0].append(r)
    return S,T

def vlsn(prob: simple_wta.WTAProblem, maxiters=100):
    # obtain feasible initial solution to problem
    (n,m) = prob.p.shape
    T = simple_wta.greedy(prob) # list of target assignments
    S = [[i for i in range(n) if T[i] == j] for j in range(m)] # list of weapon partitions by target
    # construct the improvement graph G
    G = improvement_graph(prob,S,T)
    iters = 0
    while len(G.nodes) > 0 and iters < maxiters:
        iters += 1
        # obtain negative cost subset-disjoint cycle W in G
        try:
            cycle = nx.find_negative_cycle(G,0)
            # perform multiexchange according to cycle
            multiexchange(cycle,S,T) # does update of S and T
            # update G
            G = improvement_graph(prob,S,T)
        except nx.NetworkXError as e:
            print(e)
            break
    return T,S
