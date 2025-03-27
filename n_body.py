import numpy as np
from systems import NBodyProblem
from ivp_solvers import RKF, RK4, RK2, ExplicitEuler, ImplicitEuler


n=5
init_state=np.array((5.,0.,0.,1.,    0., 6., 0., -1.5,    -1., -2., 0., 0.,   30., 1., -10., 0.,     -5., 0., 0., -1.))
masses=[100, 6, 5, 0.1, 30]
system=NBodyProblem(init_state, n, masses, grav_const=1)
solver=RKF(tol=1e-9, dt_min=1e-12, dt_max=0.01)
solver.solve_until(system, 20)
system.animate()
