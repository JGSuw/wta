import simple_wta
import networkx as nx
import numpy as np

def survival_probability(prob: simple_wta.WTAProblem, S: list, j: int):
        return np.prod((1-prob.p[S[j],j]))

def survival_value(prob: simple_wta.WTAProblem, S: list, j: int):
        return prob.v[j]*survival_probability(prob,S,j)

def cycle_improvement_graph(prob: simple_wta.WTAProblem, S: list, T: list):
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

def path_improvement_graph(prob, S: list, T: list):
    (n,m) = prob.p.shape
    q = np.array([survival_probability(prob,S,j) for j in range(m)])
    G = nx.DiGraph()
    edges = []
    for r in range(n):
         for l in range(n):
            if r != l:
                delta = -prob.v[T[l]]*q[T[l]]*prob.p[r,T[l]]-prob.v[T[r]]*q[T[r]]*(1-1/(1-prob.p[r,T[r]]))
                edges += [(r,l,delta)]
    G.add_weighted_edges_from(edges)
    return G

def cycle_detection(G: nx.DiGraph, source):
    N = len(G.nodes)
    dist = np.full((N,),np.inf)
    dist[source] = 0.
    path = np.full((N,), -1, dtype=int)
    for i in range(1,N):
        tail = -1
        for e in G.edges:
            cost = G.edges[e]["weight"]
            if dist[e[0]]+cost < dist[e[1]]:
                dist[e[1]] = dist[e[0]]+cost
                path[e[1]] = e[0]
                tail = e[1]
    if tail == -1:
        return []
    else:
        head = tail
        for i in range(1,N):
            head = path[head]
        cycle = []
        node = head
        while True:
            cycle += [node]
            if node == head and len(cycle) > 1:
                break
            node = path[node]
        cycle.reverse()
        return cycle

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
    G = cycle_improvement_graph(prob,S,T)
    iters = 0
    while len(G.nodes) > 0 and iters < maxiters:
        iters += 1
        # obtain negative cost subset-disjoint cycle W in G
        try:
            cycle = nx.find_negative_cycle(G,0)
            # perform multiexchange according to cycle
            multiexchange(cycle,S,T) # does update of S and T
            # update G
            G = cycle_improvement_graph(prob,S,T)
        except nx.NetworkXError as e:
            print(e)
            break
    return T,S
