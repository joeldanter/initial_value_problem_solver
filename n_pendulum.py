import numpy as np
from systems import NPendulum
from numeric_iterators import RKF


n=3
init_state=np.array((3.14, np.pi, 0, 0, 1, 0))
ms=(1.6,1, 1.2)
ls=(1,.9, .6)
system=NPendulum(init_state, n, ms, ls, 9.81)
iterator=RKF(tol=1e-6)
iterator.iterate_until(system, 5)
system.animate()
