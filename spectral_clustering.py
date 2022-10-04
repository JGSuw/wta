from tkinter.font import names
import numpy as np
from simple_wta import WTAProblem, random_wta_factory
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import scipy.sparse
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


def foo_adjacency(prob: WTAProblem):
    (n,m) = np.shape(prob.p)
    (n,m) = np.shape(prob.p)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                u_norm = np.sqrt(np.sum(prob.v*prob.p[i,:]**2))
                w_norm = np.sqrt(np.sum(prob.v*prob.p[j,:]**2))
                A[i,j] = np.sum(prob.p[i,:]*prob.v*prob.p[j,:])/(u_norm*w_norm)
    return A

def weapon_clustering(prob: WTAProblem, n_clusters: int, rng=np.random):
    N = np.shape(prob.p)[0]
    A = weapon_adjacency(prob)
    # clustering = SpectralClustering(n_clusters = n_clusters, affinity="precomputed", random_state=rng,assign_labels='cluster_qr').fit(A)
    # clustering = SpectralClustering(n_clusters = n_clusters, affinity="precomputed", random_state=rng,assign_labels='kmeans').fit(A)
    clustering = SpectralClustering(n_clusters = n_clusters, affinity="nearest_neighbors", random_state=rng,assign_labels='cluster_qr',n_neighbors=int(N/n_clusters)).fit(np.sqrt(prob.v)*prob.p)
    # clustering = SpectralClustering(n_clusters = n_clusters, affinity="rbf", random_state=rng,assign_labels='cluster_qr').fit(prob.p*np.sqrt(prob.v))
    # A = foo_adjacency(prob)
    # clustering = SpectralClustering(n_clusters = n_clusters, affinity="precomputed", random_state=rng,assign_labels='cluster_qr').fit(A)
    return clustering.labels_


def foo_clustering(prob: WTAProblem, n_clusters: int, rng=np.random):
    # okay going to fuck with some SHIT here
    A = np.arccos(foo_adjacency(prob))
    A_nn = np.zeros(A.shape)
    for i in range(A.shape[0]):
        idx = np.argsort(A[i,:])[::-1]
        for j in range(n_clusters):
            A_nn[i,idx[j]] = A[i,idx[j]]

    # now make A_nn symmetryc
    A_nn = (A_nn+A_nn.T)/2

    # now construct sparse nearest neighbors matrix
    rows = []
    cols = []
    data = []
    for i in range(A_nn.shape[0]):
        idx = np.argwhere(A_nn[i,:]).flatten()
        for j in idx:
            rows.append(i)
            cols.append(j)
            data.append(A_nn[i,j])
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(cols)
    A_sparse = scipy.sparse.csr_array((data,(np.array(rows),np.array(cols))),shape=A_nn.shape,dtype=np.float64)
    clustering = SpectralClustering(n_clusters=n_clusters,affinity="precomputed_nearest_neighbors").fit(A_sparse)
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
def reduce_problem(prob: WTAProblem, n_clusters: int, rng=np.random):
    (n,m) = prob.p.shape
    weapon_labels = weapon_clustering(prob,n_clusters,rng=rng)
    # weapon_labels = foo_clustering(prob,n_clusters,rng=rng)
    target_labels = target_clustering(prob,weapon_labels)
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(n_clusters)]
    return problems, coalitions, missions

# a sanity check for comparing the above to random clusters
def random_reduction(prob: WTAProblem, n_clusters: int, rng=np.random):
    (n,m) = prob.p.shape
    weapon_labels = rng.choice(n_clusters, size=n)
    target_labels = rng.choice(n_clusters, size=m)
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(n_clusters)]
    return problems, coalitions, missions
