import numpy as np

"""
A basic WTA problem. Depending on the action space, can represent different
versions of the problem. No constraints are added in this case.
"""
class WTAProblem:
    def __init__(self,
                 v: np.ndarray, # target values
                 p: np.matrix,  # probablility of kill matrix
    ):
        assert(np.shape(v)[0] == np.shape(p)[1])
        self.v = v
        self.p = p
    
    # WTA objective function for pure strategies, where x is an array of integers.
    def objective(self, x: np.ndarray):
        q = np.ones(np.shape(self.p)[1])
        for i in range(len(x)):
            q[x[i]] = q[x[i]]*(1-self.p[i,x[i]])
        return np.sum(self.v*q)

    # WTA objective function for mixed strategies, where x is a matrix whose rows
    # are probability density functions corresponding the the weapon's mixed strategies.
    def mixed_objective(self, x: np.matrix):
        q = np.prod((1-(self.p * x)),0)
        return np.sum(self.v*q)

    def random_sample_objective(self, x: np.matrix):
        (n,m) = np.shape(self.p)
        profile = np.array([np.random.choice(m,p=x[i,:]) for i in range(n)])
        return self.objective(profile)

# wonderful life utility for the WTA problem 
def wlu(prob: WTAProblem, i: int, xi: int, xmi: np.array):
    n,m = np.shape(prob.p)
    q = np.ones(m)
    for j in range(n-1):
        k = xmi[j]
        q[k] = q[k]*(1-prob.p[j,k])
    u1 = -np.sum(prob.v*q)
    q[xi] = q[xi]*(1-prob.p[i,xi])
    u2 = -np.sum(prob.v*q)
    return u2 - u1

# wonderful life utility, but with mixed strategies
def mixed_wlu(prob: WTAProblem, i: int, xi: np.ndarray, xmi: np.matrix):
    n,m = np.shape(prob.p)
    idx = [j for j in range(n) if j != i]
    q = np.prod((1-prob.p[idx,:]*xmi),0)
    u1 = -np.sum(prob.v*q)
    u2 = -np.sum(prob.v*(q*(1-prob.p[i,:]*xi)))
    return u2 - u1


def random_wta_factory(n,m):
    v = np.array(range(2,9),dtype=np.float64)
    pk = np.zeros(9)
    pk[1:] = np.linspace(start=.2,stop=.9,num=8)
    pdf = .85/8*np.ones(9)
    pdf[0] = .15
    return WTAProblem(np.random.choice(v,size=(m,)),np.random.choice(pk,size=(n,m),p=pdf))


# the convex relaxation of the WTA objective used in Kat Hendrick's paper
def convex_objective(prob: WTAProblem, x: np.matrix):
    q = np.prod((1-prob.p)**x,0)
    return np.sum(prob.v*q)


# combinatorial lower-bounding objective from Ahuja 2007
def comb_lower_bound_obj(problem: WTAProblem, x: np.matrix):
    n,m = np.shape(problem.p)
    # find pmax for each target
    pmax = np.array([np.max(problem.p[:,j]) for j in range(m)])
    # construct new WTA problem from pmax
    p = np.vstack([pmax[j]*np.ones(n) for j in range(m)]).T
    new_prob = WTAProblem(problem.v,p)
    return new_prob.objective(x)

# implementation of a maximum marginal return algorithm
# to solve the combinatorial lower-bounding objective from Ahuja 2007
# solves for pure strategies
def clb_mmr_alg(problem: WTAProblem):
    n,m = np.shape(problem.p)
    pmax = np.array([np.max(problem.p[:,j]) for j in range(m)])
    import heapq
    nodes = [(-problem.v[j],j) for j in range(m)]
    heapq.heapify(nodes)
    x = np.zeros(n,dtype=int)
    for i in range(n):
        node = heapq.heappop(nodes)
        j = node[1]
        value = node[0]*(1-pmax[j])
        x[i] = j
        heapq.heappush(nodes,(value,j))
    return x

    