import numpy as np
import pygame
from systems import NBodyProblem
from numeric_iterators import RKF


n=5
init_state=np.array((5.,0.,0.,1.,    0., 6., 0., -1.5,    -1., -2., 0., 0.,   30., 1., -10., 0.,     -5., 0., 0., -1.))
masses=[100, 6, 5, 0.1, 30]
system=NBodyProblem(init_state, n, masses, grav_const=1)
rkf_int=RKF(system, tol=1e-9, dt_min=1e-12, dt_max=0.01)
rkf_int.iterate_until(20)
system.animate()
