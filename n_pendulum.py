import numpy as np
from systems import NPendulum
from numeric_iterators import RKF


n=3
init_state=np.array((3.14, np.pi, 0, 0, 1, 0))
ms=(1.6,1, 1.2)
ls=(1,.9, .6)
system=NPendulum(init_state, n, ms, ls, 9.81)
rkf_int=RKF(system, tol=1e-6)
rkf_int.iterate_until(15)
system.animate()
