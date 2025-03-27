import numpy as np
from systems import HarmonicOscillator
from ivp_solvers import RKF, RK4, RK2, ExplicitEuler, ImplicitEuler
import matplotlib
import matplotlib.pyplot as plt


init_state=np.array((0, 4))
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

fig, (ax1, ax2) = plt.subplots(2)
for i in range(len(solvers)):
    system=HarmonicOscillator(init_state, k=2)
    solvers[i].solve_until(system, 15)
    t,x,e=[],[],[]

    for state in system.states:
        t.append(state[0])
        x.append(state[1][0])
        e.append(system.energy(state[1]))
    
    ax1.plot(t,x,color=colors[i], label=solver_names[i])
    ax2.plot(t,e,color=colors[i], label=solver_names[i])

ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend()

ax2.set_xlabel('t')
ax2.set_ylabel('Energia')
ax2.legend()
plt.show()
