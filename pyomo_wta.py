from pyomo.environ import *
import numpy as np
import pickle
from os.path import exists
import os

n_weapons = 100
n_targets = 40

#np_pk = np.random.random((n_weapons, n_targets))*0.7+0.2
#np_v = np.random.random((n_targets))*8. + 2.0

PICKLE_FILE = './pickled_solution.pickle'
if exists(PICKLE_FILE):
  pfile = open(PICKLE_FILE, 'rb')
  p_survive = pickle.load(pfile)
  np_pk = 1-p_survive
  np_pk = np_pk.transpose()
  np_v = pickle.load(pfile)
  np_v = np.squeeze(np_v)
  assignment = pickle.load(pfile)
  print("Loading previous scenario and solution")
  a, b = p_survive.shape
  #achieved_comp_local = np.ones(num_targets)
  #for i in range(len(assignment)):
  #  for j in assignment[i]:
  #    print("%d, %d"%(j, i))
  #    achieved_comp_local[i] *= p_survive[i, j]
  #print("Optimal from Ahuja: %f" % np.dot(np.reshape(achieved_comp_local,-1), np.reshape(V, -1)))

else:
  print("Generating random scenario")
  # Generate a Pk matrix
  np_pk = np.random.uniform(0.2, 0.9, (num_targets, num_weapons))

  # Generate a value vector
  np_v = np.random.uniform(2, 10, (num_targets ))

print(np_pk.shape)
model = ConcreteModel()

## Define sets ##
model.i = RangeSet(n_weapons)  # Set of weapons
model.j = RangeSet(n_targets)   # Set of targets


def pk_fun(model, i, j):
    return np_pk[i-1, j-1]
model.pk = Param(model.i, model.j, initialize=pk_fun, mutable=False)
def v_fun(model, j):
    return np_v[j-1]
model.v = Param(model.j, initialize=v_fun, mutable=False)
## Define variables ##
#  Variables
#       x(i,j)  shipment quantities in cases
#       z       total transportation costs in thousands of dollars ;
#  Positive Variable x ;
model.x = Var(model.i, model.j, within=Binary, doc='Weapon effort')

## Define contrains ##
# supply(i)   observe supply limit at plant i
# supply(i) .. sum (j, x(i,j)) =l= a(i)
def supply_rule(model, i):
  return sum(model.x[i,j] for j in model.j) == 1

model.supply = Constraint(model.i, rule=supply_rule, doc='Can only engage one target')

def objective_rule(model):
    obj = 0.0  
    for kj in model.j:
      comp = 1.0
      for ki in model.i:
        comp *= 1.0 - model.pk[ki,kj]*model.x[ki,kj]
      obj += model.v[kj]*comp
    return obj
model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

## Display of the output ##
def pyomo_postprocess(options=None, instance=None, results=None):
  model.x.display()

if __name__ == '__main__':
    # This emulates what the pyomo command-line tools does
    from pyomo.opt import SolverFactory
    import pyomo.environ
    #opt = SolverFactory("cplex_direct")
    opt = SolverFactory("ipopt")
    results = opt.solve(model)#, mip_solver='cplex_persistent', nlp_solver='cplex_direct')#, strategy='LBB', mip_solver='cplex_direct')
    #sends results to stdout
    results.write()
    print("\nDisplaying Solution\n" + '-'*60)
    pyomo_postprocess(None, model, results)
    #print(objective_rule(model))
    print(value(model.OBJ()))
