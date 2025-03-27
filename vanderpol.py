import numpy as np
from systems import VanDerPol
from ivp_solvers import RKF, RK4, RK2, ExplicitEuler, ImplicitEuler
import matplotlib.pyplot as plt


init_state=np.array((2, 0))
solvers=[ExplicitEuler(dt=0.01),
             ImplicitEuler(dt=0.01),
             RK4(dt=0.01)]
solver_names=['Explicit euler dt=0.01',
              'Implicit euler dt=0.01',
              'Runge-Kutta 4 dt=0.01']
colors='rgbmcy'

fig, ax = plt.subplots()
for i in range(len(solvers)):
    system=VanDerPol(init_state, mu=4)
    solvers[i].solve_until(system, 60)
    x,y=[],[]

    for state in system.states:
        x.append(state[1][0])
        y.append(state[1][1])

    ax.plot(x,y,color=colors[i], label=solver_names[i])

ax.set_xlabel('x')
ax.set_ylabel('v')
ax.legend()
plt.show()
