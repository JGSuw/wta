import numpy as np
import heapq

"""
A basic WTA problem. Depending on the action space, can represent different
versions of the problem. No constraints are added in this case.
"""
class WTAProblem:
    def __init__(self,
                 v: np.ndarray, # target values
                 p: np.matrix,  # probablility of kill matrix
                 q_init = None 
    ):
        assert(np.shape(v)[0] == np.shape(p)[1])
        self.v = v
        if q_init == None:
            self.q_init = np.ones(v.shape)
        else:
            self.q_init = q_init
        self.p = p
    
    # WTA objective function for pure strategies, where x is an array of integers.
    def objective(self, x: np.ndarray):
        q = self.q_init.copy()
        for i in range(len(x)):
            q[x[i]] = q[x[i]]*(1-self.p[i,x[i]])
        return np.sum(self.v*q)

    def max_objective(self, x: np.array):
        q = self.q_init.copy()
        for i in range(len(x)):
            q[x[i]] = q[x[i]]*(1-self.p[i,x[i]])
        return np.sum(self.v*(1-q))

    # WTA objective function for mixed strategies, where x is a matrix whose rows
    # are probability density functions corresponding the the weapon's mixed strategies.
    def mixed_objective(self, x: np.matrix):
        q = self.q_init*np.prod((1-(self.p * x)),0)
        return np.sum(self.v*q)

    def random_sample_objective(self, x: np.matrix):
        (n,m) = np.shape(self.p)
        profile = np.array([np.random.choice(m,p=x[i,:]) for i in range(n)])
        return self.objective(profile)

    def copy(self):
        return WTAProblem(self.v.copy(),self.p.copy())

# wonderful life utility for the WTA problem 
def wlu(prob: WTAProblem, i: int, xi: int, xmi: np.array):
    n,m = np.shape(prob.p)
    q = prob.q_init.copy()
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


# def random_wta_factory(n,m,rng=np.random):
#     v = np.array(range(1,9),dtype=np.float64)
#     pk = np.zeros(9)
#     pk[1:] = np.linspace(start=.2,stop=.9,num=8)
#     pdf = .9/8*np.ones(9)
#     pdf[0] = .1
#     return WTAProblem(rng.choice(v,size=(m,)),rng.choice(pk,size=(n,m),p=pdf))

def random_wta_factory(n,m,rng=np.random):
    return WTAProblem(rng.uniform(low=1., high=25.0,size=(m,)),rng.uniform(low=0.1,high=0.9,size=(n,m)))


# the convex relaxation of the WTA objective used in Kat Hendrick's paper
def convex_objective(prob: WTAProblem, x: np.matrix):
    q = prob.q_init.copy()
    q = q*np.prod((1-prob.p)**x,0)
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

# def partial_assignment_utility(problem: WTAProblem, x: np.array):
#     n_w,n_t = problem.p.shape
#     u = np.zeros((n_w,n_t))
#     for i in range(n_w):
#         for j in range(n_t):
#             q = np.prod([1-problem.p[k,j] for k in range(n_w) if (k!=i and x[k]==j)])
#             u[i,j] = problem.v[j]*q*problem.p[i,j]
#     return u

def old_greedy(prob: WTAProblem):
    n_w, n_t = prob.p.shape
    x = np.full(n_w,-1,dtype=int)
    u = prob.p*prob.v*prob.q_init
    print(u.shape)
    for _ in range(n_w):
        i,j = np.unravel_index(np.argmax(u),u.shape)
        if x[i] != -1:
            print("wtf")
            print(u[i,:])
        x[i] = j
        for l in range(n_w):
            q = prob.q_init[j]*np.prod([1-prob.p[l,j] for k in range(n_w) if (k!=l and  x[k]==j)])
            u[l,j] = prob.v[j]*q*prob.p[l,j]
        for j in range(n_t):
            u[i,j] = -1
    return x

def greedy(prob: WTAProblem):
    (n,m) = prob.p.shape
    q = prob.q_init.copy()
    x = np.full(n,-1,dtype=int)
    W = list(range(n))
    h = [[-prob.v[j]*q[j]*prob.p[i,j],i,j] for i in W for j in range(m)]
    heapq.heapify(h)
    while len(W) > 0:
        node = heapq.heappop(h)
        while node[1] not in W:
            node = heapq.heappop(h)
        i = node[1]
        j = node[2]
        W.remove(i)
        x[i] = j
        q[j] = q[j]*(1-prob.p[i,j])
        # update heap entries
        for node in h:
            if node[2]==j:
                node[0] = -prob.v[j]*q[j]*prob.p[node[1],j]
        heapq.heapify(h)
    return x
