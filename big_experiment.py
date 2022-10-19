import numpy as np
import simple_wta
import ahuja
import spectral_clustering
import kmeans_clustering
import regret_matching
from time import perf_counter
from matplotlib import pyplot as plt
from multiprocessing import Pool
import pandas as pd


def experiment(seed):
    rng = np.random.RandomState(seed=seed)
    # n_c = np.array([10,11])
    # n_c = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    n_c = np.array([5,10,15,20])
    n_w = np.array([n**2 for n in n_c])
    n_t = np.array([int(3*n/4) for n in n_w])

    c_greedy = []
    c_ahuja = []
    t_ahuja = []
    c_kmeans = []
    c_random = []
    t0_kmeans = []
    t1_kmeans = []
    t_random = []
    for i in range(len(n_c)):
        prob = simple_wta.random_wta_factory(n_w[i],n_t[i],rng=rng)
        x_greedy = simple_wta.greedy(prob)
        c_greedy.append(prob.objective(x_greedy))
        t0 = perf_counter()
        x_ahuja = ahuja.optimize(prob,maxiters=1000,ftol_abs=1e-4)
        t_ahuja.append(perf_counter()-t0)
        c_ahuja.append(prob.objective(x_ahuja))
        t0 = perf_counter()
        kmeans_cluster_data = kmeans_clustering.reduce_problem(prob,n_c[i],rng=rng)
        t0_kmeans.append(perf_counter()-t0)
        t1 = perf_counter()
        x_kmeans = [ahuja.optimize(p,maxiters=1000,ftol_abs=1e-4) for p in kmeans_cluster_data[0]]
        t1_kmeans.append(perf_counter()-t1)
        probs = kmeans_cluster_data[0]
        c_kmeans.append(np.sum([probs[i].objective(x_kmeans[i]) for i in range(len(kmeans_cluster_data[0]))]))
        t0 = perf_counter()
        random_cluster_data = kmeans_clustering.random_reduction(prob,n_c[i],rng=rng)
        x_random= [ahuja.optimize(p,maxiters=1000,ftol_abs=1e-4) for p in random_cluster_data[0]]
        probs = random_cluster_data[0]
        c_random.append(np.sum([probs[i].objective(x_random[i]) for i in range(len(random_cluster_data[0]))]))
        t_random.append(perf_counter()-t0)

    c_greedy = np.array(c_greedy)
    c_ahuja = np.array(c_ahuja)
    c_kmeans = np.array(c_kmeans)
    t0_kmeans = np.array(t0_kmeans)
    t1_kmeans = np.array(t1_kmeans)
    data = np.vstack((n_c,n_w,n_t,c_greedy,c_ahuja,c_kmeans,c_random,t_ahuja,t0_kmeans,t1_kmeans,t_random))
    columns = ["nc","nw","nt","cg","ca","ck","cr","ta","tk0","tk1","tr"]
    df = pd.DataFrame(data=data.T,columns=columns)
    df.to_csv("data/data%d.csv"%(seed),encoding="utf-8",index=False)
    
if __name__ == "__main__":
    t0 = perf_counter()
    N = 32
    with Pool(8) as p:
        p.map(experiment,list(range(N)))
    # experiment(1)
    print(perf_counter()-t0)
 