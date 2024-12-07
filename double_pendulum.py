import numpy as np
import matplotlib.pyplot as plt
from systems import DoublePendulum, NPendulum
from numeric_iterators import RK2, RK4, RKF


# Double Pendulum test
# nothing crazy, no visualisation
init_state=np.array((0, 1.5, 0, 0))
integrators=[RKF(NPendulum(init_state, 2, (1, 1), (1, 1), 9.81), tol=1e-6, dt_min=1e-8, dt_max=0.01),
             RKF(DoublePendulum(init_state, 1, 1, 1, 1, 9.81), tol=1e-6, dt_min=1e-8, dt_max=0.01)]

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

for i in range(len(integrators)):
    integrators[i].iterate_until(5)
    x,y,z=[],[],[]
    system=integrators[i].system
    for state in system.states:        
        angle1=state[1][0]
        angle2=state[1][1]
        ang_vel1=state[1][2]
        ang_vel2=state[1][3]

        try:
            x1=np.sin(angle1)*system.l1
            y1=-np.cos(angle1)*system.l1
            pos1=np.array((x1,y1))
            x2=np.sin(angle2)*system.l2
            y2=-np.cos(angle2)*system.l2
            pos2=np.array((x2,y2))+pos1
        except:
            x1=np.sin(angle1)*system.ls[0]
            y1=-np.cos(angle1)*system.ls[0]
            pos1=np.array((x1,y1))
            x2=np.sin(angle2)*system.ls[1]
            y2=-np.cos(angle2)*system.ls[1]
            pos2=np.array((x2,y2))+pos1

        x.append(pos2[0])
        y.append(pos2[1])
        z.append(state[0]+i*0.1)

    ax.plot(x,y,z,color='rgbm'[i])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
