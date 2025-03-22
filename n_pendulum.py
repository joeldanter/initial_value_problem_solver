import numpy as np
from systems import NPendulum
from numeric_iterators import RK4


n=2
init_state=np.array((1, 4, 0, -3))
ms=(1,1)
ls=(1,1)
system=NPendulum(init_state, n, ms, ls, 9.81)
rkf_int=RK4(system, dt=0.01)
rkf_int.iterate_until(4)
system.animate()
