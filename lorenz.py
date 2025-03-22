import numpy as np
from systems import Lorenz
from numeric_iterators import ExplicitEuler, ImplicitEuler, RK4
import matplotlib.pyplot as plt


init_state=np.array((1, 1, 2))
integrators=[ExplicitEuler(dt=0.01),
             ImplicitEuler(dt=0.01),
             RK4(dt=0.01)]
colors='rgbmcy'

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
for i in range(len(integrators)):
    system=Lorenz(init_state)
    integrators[i].iterate_until(system, 5)
    x,y,z=[],[],[]

    for state in system.states:
        x.append(state[1][0])
        y.append(state[1][1])
        z.append(state[1][2])

    ax.plot(x,y,z,color=colors[i])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
