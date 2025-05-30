import numpy as np
from systems import Lorenz
from ivp_solvers import RKF, RK4, RK2, ExplicitEuler, ImplicitEuler
import matplotlib
import matplotlib.pyplot as plt


init_state=np.array((1, 1, 2))
solvers=[ExplicitEuler(dt=0.01),
             ImplicitEuler(dt=0.01),
             RK4(dt=0.01)]
solver_names=[f'Explicit euler dt={solvers[0].dt}',
              f'Implicit euler dt={solvers[1].dt}',
              f'Runge-Kutta 4 dt={solvers[2].dt}']
colors='rgbmcy'
font = {'family' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
for i in range(len(solvers)):
    system=Lorenz(init_state)
    solvers[i].solve_until(system, 5)
    x,y,z=[],[],[]

    for state in system.states:
        x.append(state[1][0])
        y.append(state[1][1])
        z.append(state[1][2])

    ax.plot(x,y,z,color=colors[i],label=solver_names[i])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
