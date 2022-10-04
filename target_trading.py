import numpy as np
from simple_wta import WTAProblem, wlu
import ahuja
import heapq

# def greedy_refinement(prob,assignment):
#     pass

def make_offer(prob: WTAProblem, assignment: list, pk: list, v):
    u0 = prob.max_objective(assignment)
    offers = []
    for t in range(prob.p.shape[1]):
        old_p = prob.p[:,t]
        old_v = prob.v[t]
        prob.p[:,t] = pk
        prob.v[t] = v
        new_assignment = ahuja.optimize_from_initial(prob, assignment)
        u = prob.max_objective(new_assignment)
        if u-u0 > 0.:
            offers.append((t,u-u0))
        prob.p[:,t] = old_p
        prob.v[t] = old_v
    return offers

def evaluate_offer(prob: WTAProblem, assignment, offer, target, pk: np.matrix, v: np.array):
    u0 = prob.max_objective(assignment)
    switch_utility = []
    for i in range(len(offer)):
        old_p = prob.p[:,target]
        old_v = prob.v[target]
        prob.p[:,target] = pk[:,i]
        prob.v[target] = v[i]
        new_assignment = ahuja.optimize_from_initial(prob, assignment)
        u = prob.max_objective(new_assignment)
        switch_utility.append(offer[i][1]+u-u0)
        prob.p[:,target] = old_p
        prob.v[target] = old_v
    if np.any(np.array(switch_utility) > 0.):
        return offer[np.argmax(switch_utility)][0]
    else:
        return None
