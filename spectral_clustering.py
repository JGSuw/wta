from tkinter.font import names
import numpy as np
from simple_wta import WTAProblem, random_wta_factory
from sklearn.cluster import SpectralClustering
import heapq

# hacking on some spectral clustering ideas
# key idea is to apply spectral clustering based on affinity data.

# affinity should capture:
    # how important it is that two weapons work together 
    # how easy it is for the weapons to communicate with each other

# the former may be partly captured by how similar the PK vectors are for two weapons (an inner product)
# the latter involves structure which has not yet been added to the WTA problem

# for now, focus on only the former case (fully connected graph with no communication affinity).

def weapon_affinity(prob: WTAProblem, i: int, j: int):
    return np.sum(prob.p[i,:]*prob.v*prob.p[j,:]) 
    # return np.sum(prob.p[i,:]*prob.p[j,:]) 

def weapon_adjacency(prob: WTAProblem):
    (n,m) = np.shape(prob.p)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i,j] = weapon_affinity(prob,i,j)
    return A

# not using laplacian (yet)
# def laplacian(prob: WTAProblem):
#     (n,m) = np.shape(prob.p)
#     N = range(n)
#     iter = (weapon_affinity(prob,i,j) for i in N for j in N)
#     L = -adjacency(prob)
#     for i in range(n):
#         L[i,i] = n-1 # fully connected graph
#     return L

def weapon_clustering(prob: WTAProblem, n_clusters: int):
    N = range(np.shape(prob.p)[0])
    A = weapon_adjacency(prob)
    clustering = SpectralClustering(n_clusters = n_clusters, affinity="precomputed").fit(A)
    return clustering.labels_

# a simple greedy heuristic for target clustering
def target_clustering(prob: WTAProblem, weapon_labels: np.array):
    (n,m) = prob.p.shape
    A = prob.p*prob.v
    heap = [(-A[i,j],i,j) for i in range(n) for j in range(m)]
    heapq.heapify(heap)
    I = list(range(n))
    J = list(range(m))
    m = np.zeros(m)
    while J and I:
        val,i,j = heapq.heappop(heap)
        if i in I and j in J:
            m[j] = weapon_labels[i]
            I.remove(i)
            J.remove(j)
    return m

# takes a large WTA problem, and breaks it into n_clusters smaller WTA problems
def reduce_problem(prob: WTAProblem, n_clusters: int):
    (n,m) = prob.p.shape
    weapon_labels = weapon_clustering(prob,n_clusters)
    target_labels = target_clustering(prob,weapon_labels)
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(n_clusters)]
    return problems, coalitions, missions

# a sanity check for comparing the above to random clusters
def random_reduction(prob: WTAProblem, n_clusters: int):
    (n,m) = prob.p.shape
    weapon_labels = np.random.choice(n_clusters, size=n)
    target_labels = np.random.choice(n_clusters, size=m)
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(n_clusters)]
    return problems, coalitions, missions
