from simple_wta import WTAProblem, wlu, greedy
import numpy as np
# would like to implement basic regret matching algorithm in the Arslan paper
# this will give me a point of comparison to greedy optimistic and convex relaxation approaches

# computes average regret for agent i given lumped average regret rk computed over k iterations
# and action profile xmi
def average_regret(prob: WTAProblem, 
                    i: int,             # agent to compute regret for
                    k: int,             # iterate for regret weighting 
                    rk: np.array,          # average regret vector (last iterate)    
                    xk: np.ndarray):    # action of all players at iterate k
    n,m = np.shape(prob.p)
    r = np.zeros(m)
    idx = [j for j in range(n) if j!=i]
    for j in range(m):
        xi = j
        r[j] = (k-1)/k * rk[j] + 1/k * (wlu(prob,i,xi,xk[idx])-wlu(prob,i,xk[i],xk[idx]))
    return r

def action_update(r: np.ndarray, rng=np.random):
    z = np.array([max(0,y) for y in r])
    sum_z = np.sum(z)
    if sum_z > 0:
        p = z/sum_z
        return rng.choice(len(r),p=p),p
    else:
        return rng.choice(len(r)),np.ones(len(r))/len(r)

def greedy_regret(prob: WTAProblem):
    x = greedy(prob)
    A = np.zeros(prob.p.shape)
    for i in range(len(x)):
        A[i,x[i]] = 1
    return A*prob.v

def iterate(prob: WTAProblem, R: np.matrix, P: np.matrix, x: np.array, k: int, rng=np.random):
    (n,m) = prob.p.shape
    for i in range(n):
        R[i,:] = average_regret(prob,i,k,R[i,:],x)
    for i in range(n):
        x[i],P[i,:] = action_update(R[i,:], rng=rng)
    return x,P,R

"""
Simulates the dynamics of (synchronous) distributed regret matching.
"""
def learning_dynamics(prob: WTAProblem, N: int, rng=np.random):
    (n,m) = np.shape(prob.p)
    # initialize distributions to be uniform
    P = np.ones((n,m))/m
    # initialize target assignments randomly
    x = rng.choice(m,size=(n,))
    # initialize regret vectors to be zero
    R = greedy_regret(prob)
    for k in range(1,N):
        for i in range(n):
            # calculate average regret of player i
            R[i,:] = average_regret(prob,i,k,R[i,:],x)
        for i in range(n):
            x[i],P[i,:] = action_update(R[i,:], rng=rng)
    return x,P,R
