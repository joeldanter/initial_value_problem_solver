import numpy as np
import pygame
from system import NBodyProblem
from numeric_integrator import RKF


init_state=np.array((5.,0.,0.,1.,    0., 6., 0., -1.5,    -1., -2., 0., 0.,   30., 1., -10., 0.,     -5., 0., 0., -1.))
masses=[100, 6, 5, 0.1, 30]
colors=['red', 'green', 'blue', 'magenta', 'cyan']
n=5
rkf_int=RKF(NBodyProblem(init_state, n, masses, grav_const=1), tol=1e-6, dt_min=1e-12, dt_max=0.01)
rkf_int.iterate_until(20)

pygame.init()
states=rkf_int.system.states
pygame.display.set_caption('n-body problem')
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run=False

        real_t=pygame.time.get_ticks()-starttime
        sim_t=states[i][0]*1000    
        if (real_t<sim_t):
            pygame.time.wait(int(sim_t-real_t))
            
        lasti=i
pygame.quit()
