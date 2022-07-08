import cvxpy as cvx
import numpy as np
from simple_wta import WTAProblem, convex_objective
from scipy.optimize import minimize, Bounds

def scipy_optimize(prob: WTAProblem):
    (n,m) = np.shape(prob.p)
    bounds = Bounds(np.zeros(n*m),np.ones(n*m))
    constraints = {
        "type": "ineq",
        "fun": lambda x: -np.sum(x.reshape((n,m)),1) + 1,
    }

    f = lambda x: convex_objective(prob,x.reshape((n,m)))
    x0 = 1/m*np.ones(n*m)
    return minimize(f,x0,method="trust-constr",bounds=bounds,constraints=constraints,options={"maxiter": 10})
