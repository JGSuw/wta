from simple_wta import WTAProblem, wlu, greedy
import numpy as np



def iterate(prob: WTAProblem, P: np.matrix, x: np.array, tau: float, rng=np.random):

    (n,m) = prob.p.shape
    i = rng.choice(n)
    idx = [j for j in range(n) if j!=i]
    # compute utility
    U = np.array([wlu(prob,i,a,x[idx]) for a in range(m)])/tau
    exp = np.exp(U)
    # update probability vector 
    P[i,:] = exp/np.sum(exp)
    # sample action
    x[i] = rng.choice(m,p=P[i,:])

    return x,P