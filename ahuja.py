#from curses import meta
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from os.path import exists
from simple_wta import WTAProblem, clb_mmr_alg

def generate_random_assignment(p_survive, V):
    targets, weapons = p_survive.shape
    share = np.rint(weapons*V / np.sum(V)).astype(int)
    while (np.sum(share) > weapons):
        share[np.argmax(share)] -= 1

    target_sets = []
    indices = np.arange(weapons, dtype=int)
    
    for j in range(targets):
        weapons = np.random.choice(indices, share[j], False)
        indices = [i for i in indices if i not in weapons]
        target_sets.append(weapons)
    while (len(indices) > 0):
        r = np.random.randint(targets)
        target_sets[r] = np.append(target_sets[r], indices[0])
        indices = np.delete(indices, 0)
    return target_sets

def evaluate_solution(p_survive, V, assignment):
    assert (len(assignment) == V.shape[0]), f"There are {len(assignment)} sets, expected {V.shape[0]}"

    EV_per_target = np.copy(V)
    for j in range(len(assignment)):
        log = np.log(p_survive[j, assignment[j]])
        EV_per_target[j] = np.exp(np.sum(log))*V[j,0]
    return EV_per_target, np.sum(EV_per_target)

def refine_solution(p_survive, V, assignment, EV_per_target):
    assignment_changed = False
    EV = np.sum(EV_per_target)

    for j in range(len(assignment)):
        relevant_probs = p_survive[j, assignment[j]]
        assignment[j] = assignment[j][np.argsort(relevant_probs)]
        min_diff = 0.
        new_target = j
        for jj in range(len(assignment)):
            if len(assignment[j]) > 0:
                diff = EV_per_target[j]*(1./p_survive[j, assignment[j][-1]]-1) + EV_per_target[jj]*(p_survive[jj, assignment[j][-1]]- 1)
                if (diff < min_diff):
                    min_diff = diff
                    new_target = jj

        if new_target != j:
            assignment_changed = True
            assignment[new_target] = np.append(assignment[new_target], assignment[j][-1])
            assignment[j] = np.delete(assignment[j], -1)
            EV_per_target, EV = evaluate_solution(p_survive, V, assignment)

    #EV_per_target, EV = evaluate_solution(p_survive, V, assignment) # Get better results with it here, but less proper
    return assignment, assignment_changed, EV_per_target, EV

def apply_cycle(assignment, cycle):
    # Note that this should only be called for a cycle, not a path
    original_assignment = np.copy(assignment)
    for r, l in zip (cycle, cycle[1:]):
        t_r = -1
        t_l = -1
        for j in range(len(original_assignment)):
            if r in original_assignment[j]:
                t_r = j
            if l in original_assignment[j]:
                t_l = j
        if (t_l < 0):
            print("There are %d weapons" % count_weapons(assignment))
        assert t_r >= 0, "Target not found for r = %d" % r
        assert t_l >= 0, "Target not found for l = %d" % l
        #print(" PRE: w(%d): %d \t w(%d): %d"% (t_r, len(assignment[t_r]), t_l, len(assignment[t_l])))
        assignment[t_l] = np.append(assignment[t_l], r)
        assignment[t_r] = assignment[t_r][assignment[t_r] != r]
        #print("POST: w(%d): %d \t w(%d): %d"% (t_r, len(assignment[t_r]), t_l, len(assignment[t_l])))
    return assignment

def make_graph(p_survive, assignment, ev_per_target):
    DG = nx.DiGraph()
    for j in range(len(assignment)):
        for jj in range(len(assignment)):
            if (j != jj):
                for r in assignment[j]:
                    for l in assignment[jj]:
                        c_r_l = float(ev_per_target[jj]*((p_survive[jj, r]/p_survive[jj, l]) - 1.))
                        #print("%d targeting %d replaces %d targeting %d, c =  %f"% (r, j, l, jj, c_r_l))
                        DG.add_weighted_edges_from([ (r, l, c_r_l) ])
    return DG

def count_weapons(assignment):
    w = 0
    for j in range(len(assignment)):
        w += len(assignment[j])
    return w

def optimize(prob: WTAProblem, maxiters=100, ftol_abs = 1e-4, verbose=False):
    p_survive = (1 - prob.p).T
    v = prob.v.reshape((len(prob.v),1))
    weap_assignment = clb_mmr_alg(prob)

    assignment = [np.where(weap_assignment == i)[0] for i in range(len(v))]

    ev_per_target, ev = evaluate_solution(p_survive, v, assignment)
    last_ev = ev
    if verbose:
        print("Initial: %.4f" % ev)

    #for target_set in assignment:
    #    print(target_set)
    meta_assignment_changed = True
    iters = 1
    while (meta_assignment_changed) and iters < maxiters:
        meta_assignment_changed=False
        # Refine assignment matrix
        assignment_changed = True
        i=0
        while (assignment_changed) and iters < maxiters:
            iters += 1
            assignment, assignment_changed, ev_per_target, ev = refine_solution(p_survive, v, assignment, ev_per_target)
            i +=1
            if verbose:
                print("Updated: %.4f" % ev)
            if abs(ev-last_ev) < ftol_abs:
                break
            last_ev = ev
        
        #for target_set in assignment:
        #    print(target_set)

        if i > 1:
            meta_assignment_changed = True
        DG = make_graph(p_survive, assignment, ev_per_target)

        #print(nx.find_negative_cycle(DG, 0))
        assignment_changed = True
        while (assignment_changed) and iters < maxiters:
            iters += 1
            try:
                cycle = nx.find_negative_cycle(DG, 0)
                #print(cycle)
                assignment_changed = True
                assignment = apply_cycle(assignment, cycle)
                ev_per_target, ev = evaluate_solution(p_survive, v, assignment)
                if abs(ev-last_ev) < ftol_abs:
                    break
                last_ev = ev
                DG = make_graph(p_survive, assignment, ev_per_target)
                if verbose:
                    print(" Cycled: %.4f" % ev)
                meta_assignment_changed = True
            except nx.NetworkXError:
                if verbose:
                    print("No cycle found")
                assignment_changed = False
            except nx.NetworkXNoCycle:
                if verbose:
                    print("No cycle found")
                assignment_changed = False

    weapon_assignment = np.zeros(prob.p.shape[0],dtype=int)
    for j in range(len(prob.v)):
        for w in assignment[j]:
            weapon_assignment[w] = j
    
    return weapon_assignment