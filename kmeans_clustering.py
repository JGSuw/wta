from tkinter.font import names
import numpy as np
from simple_wta import WTAProblem, random_wta_factory
from sklearn.cluster import KMeans 
import heapq
from math import floor

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

def distance_metric(prob: WTAProblem, i: int, j: int):
    y = prob.p[i,:]-prob.p[j,:]
    return np.sum(prob.v*y**2)

def weapon_clustering(prob: WTAProblem, n_clusters: int, rng=np.random):
    N = range(np.shape(prob.p)[0])
    clustering = KMeans(n_clusters = n_clusters, random_state=rng).fit(np.sqrt(prob.v)*prob.p)
    # clustering = KMeans(n_clusters = n_clusters, random_state=rng).fit(prob.p)
    return clustering.labels_

# a simple greedy heuristic for target clustering
def target_clustering(prob: WTAProblem, weapon_labels: np.array):
    (n,m) = prob.p.shape
    A = prob.p*prob.v
    heap = []
    I = []
    J = list(range(m))
    m = np.zeros(m)
    while J:
        if len(I) == 0:
            I = list(range(n))
            heap = [(-A[i,j],i,j) for i in I for j in J]
            heapq.heapify(heap)
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
    target_labels = target_clustering(prob,weapon_labels)
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    
    # find dead coalitions to dissolve
    dead_coalitions = []
    i = 0
    for i in range(len(coalitions)):
        if len(missions[i]) == 0:
            dead_coalitions.append(coalitions[i])

    # build training set for k-nn classificaiton
    training_set = list(range(n))
    for c in dead_coalitions:
        for w in c:
            training_set.remove(w)

    # apply k-nn classification to weapons in dead coalitions
    for c in dead_coalitions:
        for w in c:
            d = np.sum(prob.v*(prob.p[w,:]-prob.p[training_set,:])**2,axis=1)
            sorted_order = np.argsort(d)
            votes = weapon_labels[sorted_order[:5]]
            count = [np.sum(votes == v) for v in votes]
            new_c = votes[np.argmax(count)]
            coalitions[new_c] = np.append(coalitions[new_c],w)

    # remove dead coalitions and missions
    i = 0
    while i < len(coalitions):
        if len(missions[i]) == 0:
            coalitions.pop(i)
            missions.pop(i)
        else:
            i+=1

    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(len(coalitions))]
    return problems, coalitions, missions

def random_reduction(prob: WTAProblem, n_clusters: int, rng=np.random):
    (n,m) = prob.p.shape
    div = floor(n/n_clusters)
    weapon_labels = np.array([])
    for i in range(div):
        weapon_labels = np.append(weapon_labels,rng.choice(n_clusters,size=n_clusters,replace=False))
    weapon_labels = np.append(weapon_labels,rng.choice(n_clusters,size=n%n_clusters,replace=False))
    div = floor(m/n_clusters)
    target_labels = np.array([])
    for i in range(div):
        target_labels = np.append(target_labels,rng.choice(n_clusters,size=n_clusters,replace=False))
    target_labels = np.append(target_labels,rng.choice(n_clusters,size=m%n_clusters,replace=False))
    coalitions = [np.where(weapon_labels==i)[0] for i in range(n_clusters)]
    missions = [np.where(target_labels==i)[0] for i in range(n_clusters)]
    problems = [WTAProblem(prob.v[missions[i]],(prob.p[coalitions[i],:])[:,missions[i]]) for i in range(n_clusters)]
    return problems, coalitions, missions