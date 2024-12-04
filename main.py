import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from system import HarmonicOscillator, NBodyProblem, Lorenz, DoublePendulum, NPendulum
from numeric_intergator import ExplicitEuler, ImplicitEuler, RK2, RK4, RKF



# n-body problem
init_state=np.array((5.,0.,0.,1.,    0., 6., 0., -1.5,    -1., -2., 0., 0.,   30., 1., -10., 0.,     -5., 0., 0., -1.))
masses=[100, 6, 5, 0.1, 30]
colors=['red', 'green', 'blue', 'magenta', 'cyan']
n=5
rkf_int=RKF(NBodyProblem(init_state, n, masses, grav_const=1), tol=1e-6, dt_min=1e-12, dt_max=0.01)
rkf_int.iterate_until(20)

pygame.init()
states=rkf_int.system.states
pygame.display.set_caption('Pygame window')
screen_size=np.array((1920, 1080))
screen=pygame.display.set_mode(screen_size)
starttime=pygame.time.get_ticks()

lasti=0
for i in range(1, len(states)):
    if states[i][0]-states[lasti][0]>=0.001:
        for body in range(rkf_int.system.n):
            pos1=np.array((states[lasti][1][body*4], states[lasti][1][body*4+1]))
            pos2=np.array((states[i][1][body*4], states[i][1][body*4+1]))
            pos1*=50
            pos2*=50
            pos1[1]*=-1
            pos2[1]*=-1
            pos1+=np.array(screen_size/2)
            pos2+=np.array(screen_size/2)
            
            color=colors[body]
            pygame.draw.line(screen, color, pos1, pos2, 2)

        pygame.display.update()

        real_t=pygame.time.get_ticks()-starttime
        sim_t=states[i][0]*1000    
        if (real_t<sim_t):
            pygame.time.wait(int(sim_t-real_t))
            
        lasti=i
pygame.quit()

'''
# Double Pendulum
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
'''
'''
# Lorenz
init_state=np.array((1, 1, 2))
integrators=[ExplicitEuler(Lorenz(init_state), dt=0.001),
             ImplicitEuler(Lorenz(init_state), dt=0.001),
             RK4(Lorenz(init_state), dt=0.001)]
colors='rgbmcy'

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
for i in range(len(integrators)):
    integrators[i].iterate_until(20)
    x,y,z=[],[],[]

    for state in integrators[i].system.states:
        x.append(state[1][0])
        y.append(state[1][1])
        z.append(state[1][2])

    ax.plot(x,y,z,color=colors[i])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
'''
# problemak
# energia
# adatstrukturak
