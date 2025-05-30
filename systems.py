
import numpy as np
from scipy.integrate import dblquad
from abc import ABC, abstractmethod
import scipy.optimize
import pygame


# TODO: terrible data structures
class System(ABC):
    def __init__(self, init_state):
        self.states=[(0, init_state)]
        self.t=0

    @abstractmethod
    def diff_eq(self, state):
        pass

    def last_state(self):
        return self.states[-1][1]

    def append_new_state(self, new_state, dt):
        self.t+=dt
        self.states.append((self.t, new_state))

class HarmonicOscillator(System):
    def __init__(self, init_state, k=1):
        super().__init__(init_state)
        self.k=k
    
    def diff_eq(self, state):
        #x,v
        x=state[0]
        v=state[1]
        a=-self.k*x
        return np.array((v, a))
    
    def energy(self, state):
        x,v=state[0],state[1]
        potential_E=self.k * x**2 / 2
        kinetic_E=v**2 / 2
        return potential_E+kinetic_E

class VanDerPol(System):
    def __init__(self, init_state, mu=0.5):
        super().__init__(init_state)
        self.mu=mu

    def diff_eq(self, state):
        #x,v
        x=state[0]
        v=state[1]
        a= self.mu *(1 - x**2)* v - x
        return np.array((v,a))

class NBodyProblem(System):
    def __init__(self, init_state, n, masses, grav_const=1):
        super().__init__(init_state)
        self.n=n
        self.grav_const=grav_const
        self.massess=masses

    '''
    def diff_eq(self, state):
        #(x11, x12, v11, v12), (x21, x22, v21, v22), (x31, x32, v31, v32)...
        xs=[]
        vs=[]
        for i in state:
            xs.append(np.array((state[0], state[1])))
            vs.append(np.array((state[2], state[3])))

        accels=[]
        for i in range(self.n):
            accels.append(np.zeros(2))
        for i in range(0, self.n-1):
            for j in range(i+1, self.n):
                d=xs[j]-xs[i]
                s=self.grav_const*d/(np.linalg.norm(d)**3)
                accels[i]+=s*self.massess[j]
                accels[j]-=s*self.massess[i]

        res=[]
        for i in range(self.n):
            diffs=np.zeros(0)
            diffs[0]=vs[i][0]
            diffs[0]=vs[i][1]
            diffs[1]=accels[i][0]
            diffs[2]=accels[i][1]
            res.append(diffs)
        return res
    '''

    def diff_eq(self, state):
        #x11, x12, v11, v12, x21, x22, v21, v22, x31, x32, v31, v32...
        xs=[]
        vs=[]
        for i in range(self.n):
            xs.append(np.array((state[i*4], state[i*4+1])))
            vs.append(np.array((state[i*4+2], state[i*4+3])))

        accels=[]
        for i in range(self.n):
            accels.append(np.zeros(2))
        for i in range(0, self.n-1):
            for j in range(i+1, self.n):
                d=xs[j]-xs[i]
                s=self.grav_const*d/(np.linalg.norm(d)**3)
                accels[i]+=s*self.massess[j]
                accels[j]-=s*self.massess[i]

        res=np.zeros(self.n*4)
        for i in range(self.n):
            res[i*4]=vs[i][0]
            res[i*4+1]=vs[i][1]
            res[i*4+2]=accels[i][0]
            res[i*4+3]=accels[i][1]        
        return res
    
    def energy(self, state):
        kinetic_E=0
        for i in range(self.n):
            v_x=state[i*4+2]
            v_y=state[i*4+3]
            v_sqr=v_x**2 + v_y**2
            kinetic_E += self.massess[i] * v_sqr / 2 # 1/2*m*v^2
        
        potential_E=0
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                x1=state[i*4]
                y1=state[i*4+1]
                x2=state[j*4]
                y2=state[j*4+1]
                r=np.sqrt((x1-x2)**2+(y1-y2)**2)
                potential_E += - self.grav_const * self.massess[i]*self.massess[j] / r
        return potential_E+kinetic_E

    def animate(self):
        pygame.init()
        pygame.display.set_caption('n-body problem')
        screen_size=np.array((1920, 1080))
        screen=pygame.display.set_mode(screen_size)
        trace_surface = pygame.surface.Surface(screen_size)
        bodies_surface = pygame.surface.Surface(screen_size).convert_alpha()
        render_font=pygame.font.SysFont('monospace', 40)
        starttime=pygame.time.get_ticks()
        
        states=self.states
        colors=['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
        lasti=0
        for state_i in range(1, len(states)):
            time=states[state_i][0]
            prev_time=states[lasti][0]
            if time-prev_time>=0.01: # make sure we can keep up
                state=states[state_i][1]
                prev_state=states[lasti][1]

                # traces
                bodies_surface.fill((0,0,0,0))
                for body in range(self.n):
                    pos1=np.array((prev_state[body*4], prev_state[body*4+1]))
                    pos2=np.array((state[body*4], state[body*4+1]))
                    pos1*=50
                    pos2*=50
                    pos1[1]*=-1
                    pos2[1]*=-1
                    pos1+=np.array(screen_size/2)
                    pos2+=np.array(screen_size/2)
                    
                    color=colors[body%len(colors)]
                    pygame.draw.line(trace_surface, color, pos1, pos2, 3)
                    pygame.draw.circle(bodies_surface, color, pos2, 10)
                screen.blit(trace_surface, (0,0))
                screen.blit(bodies_surface, (0,0))

                # texts
                texts=[f't={time:0.3f}',
                f'dt={time-states[state_i-1][0]:0.6f}',
                f'E={self.energy(state):0.8f}']

                for text_i in range(len(texts)):
                    label=render_font.render(texts[text_i], True, 'white')
                    screen.blit(label, (10, 10+text_i*40))

                # exit
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # time
                real_t=pygame.time.get_ticks()-starttime
                sim_t=time*1000
                if (real_t<sim_t):
                    pygame.time.wait(int(sim_t-real_t))

                lasti=state_i
        pygame.quit()

class Lorenz(System):
    def __init__(self, init_state, sigma=10., rho=28., beta=8/3):
        super().__init__(init_state)
        self.sigma=sigma
        self.rho=rho
        self.beta=beta

    def diff_eq(self, state):
        #x,y,z
        x=state[0]
        y=state[1]
        z=state[2]
        return np.array((self.sigma*(y-x), x*(self.rho-z)-y,x*y-self.beta*z))

class DoublePendulum(System):
    def __init__(self, init_state, m1, m2, l1, l2, g):
        super().__init__(init_state)
        self.m1=m1
        self.m2=m2
        self.l1=l1
        self.l2=l2
        self.g=g
    
    def diff_eq(self, state):
        # ang1, ang2, ang_vel1, ang_vel2
        angle1=state[0]
        angle2=state[1]
        ang_vel1=state[2]
        ang_vel2=state[3]
        
        a1=(self.m1+self.m2)*self.l1**2
        b1=self.m2*self.l1*self.l2*np.cos(angle1-angle2)
        c1=-(self.m1+self.m2)*self.g*self.l1*np.sin(angle1)-self.m2*self.l1*self.l2*ang_vel2**2*np.sin(angle1-angle2)

        a2=self.m2*self.l1*self.l2*np.cos(angle1-angle2)
        b2=self.m2*self.l2**2
        c2=self.m2*self.l1*self.l2*ang_vel1**2*np.sin(angle1-angle2)-self.m2*self.g*self.l2*np.sin(angle2)
    
        angular_accel1=(b2*c1-b1*c2)/(a1*b2-a2*b1)
        angular_accel2=(a2*c1-a1*c2)/(a2*b1-a1*b2)

        return np.array((ang_vel1, ang_vel2, angular_accel1, angular_accel2))

class NPendulum(System):
    def __init__(self, init_state, n, ms, ls, g):
        super().__init__(init_state)
        self.n=n
        self.ms=ms
        self.ls=ls
        self.g=g
        self.prev_accels=np.zeros(n)
    
    def diff_eq(self, state):
        # ang1, ang2, ..., angn, ang_vel1, ang_vel2, ..., ang_veln
        f=lambda accels: [self.eq_part(state, accels, i) for i in range(self.n)]
        solved_accels=scipy.optimize.newton_krylov(f, self.prev_accels)
        result=np.zeros(2*self.n)
        for i in range(self.n):
            result[i]=state[self.n+i]
            result[self.n+i]=solved_accels[i]
        return result
    
    def eq_part(self, state, accels, j):
        angs=state[:self.n]
        vels=state[self.n:]
        sum1=0
        for k in range(self.n):
            sum1+=self.g*self.ls[j]*np.sin(angs[j])*self.ms[k]*self.sigma(j, k)
            sum1+=self.ms[k]*self.ls[j]**2*accels[j]*self.sigma(j, k)
            sum2=0
            for q in range(k, self.n):
                sum2+=self.ms[q]*self.sigma(j, q)
            sum1+=sum2*self.ls[j]*self.ls[k]*np.sin(angs[j]-angs[k])*vels[j]*vels[k]
            sum1+=sum2*self.ls[j]*self.ls[k]*(np.sin(angs[k]-angs[j])*(vels[j]-vels[k])*vels[k] + self.phi(j,k)*np.cos(angs[j]-angs[k])*accels[k])
        return sum1

    def sigma(self, j, k):
        return j<=k
    
    def phi(self, j, k):
        return j!=k
    
    def energy(self, state):
        T=0
        V=0
        for i in range(self.n):
            vx=0
            vy=0
            h=0
            for j in range(i+1):
                theta=state[j]
                omega=state[self.n+j]
                vx += np.cos(theta)*omega*self.ls[j]
                vy += np.sin(theta)*omega*self.ls[j]

                h-=self.ls[j]*np.cos(theta)
            v_sqr= vx**2 + vy**2
            T+= 1/2 * self.ms[i] * v_sqr

            V+=self.ms[i]*self.g*h
        
        return T+V

    def animate(self):
        pygame.init()
        pygame.display.set_caption('n-pendulum')
        screen_size=np.array((1920, 1080))
        screen=pygame.display.set_mode(screen_size)
        trace_surface = pygame.surface.Surface(screen_size)
        pendulum_surface = pygame.surface.Surface(screen_size).convert_alpha()
        render_font=pygame.font.SysFont('monospace', 40)
        starttime=pygame.time.get_ticks()
        
        states=self.states
        lasti=0
        for state_i in range(1, len(states)):
            time=states[state_i][0]
            prev_time=states[lasti][0]
            if time-prev_time>=0.01: # make sure we can keep up
                state=states[state_i][1]
                prev_state=states[lasti][1]
                
                # pendulums and traces
                pendulum_surface.fill((0,0,0,0))
                prev_pos=screen_size/2
                for point_i in range(self.n):
                    theta=state[point_i]
                    displacement=150*self.ls[point_i]*np.array((np.sin(theta), np.cos(theta)))
                    pos=prev_pos+displacement
                    pygame.draw.circle(pendulum_surface, 'white', pos, 10)
                    pygame.draw.line(pendulum_surface, 'white', prev_pos, pos, 3)
                    prev_pos=pos

                #pygame.draw.line(trace_surface, 'red', prev_state, pos, 3)
                screen.blit(trace_surface, (0,0))
                screen.blit(pendulum_surface, (0,0))

                # texts
                texts=[f't={time:0.3f}',
                f'dt={time-states[state_i-1][0]:0.6f}',
                f'E={self.energy(state):0.8f}']

                for text_i in range(len(texts)):
                    label=render_font.render(texts[text_i], True, 'white')
                    screen.blit(label, (10, 10+text_i*40))

                # exit
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # time
                real_t=pygame.time.get_ticks()-starttime
                sim_t=time*1000
                if (real_t<sim_t):
                    pygame.time.wait(int(sim_t-real_t))

                lasti=state_i
        pygame.quit()

class TorusGravity(System):
    # TODO no clue if this works, havent tested at all
    # TODO ROTATION HASNT BEEN ADDED
    def __init__(self, init_state, grav_const, outer_r, inner_r, torus_m, point_m):
        super().__init__(init_state)
        self.grav_const=grav_const
        self.torus_volume=2 * np.pi**2 * inner_r**2 * outer_r
        self.outer_r=outer_r
        self.inner_r=inner_r
        self.torus_m=torus_m
        self.point_m=point_m

    def diff_eq(self, state):
        x=state[0]
        y=state[1]
        z=state[2]
        vx=state[3]
        vy=state[4]
        vz=state[5]

        multiplier=self.grav_const*(self.torus_m+self.point_m)/self.torus_volume
        ax = multiplier*dblquad(lambda l,phi:self.torus_x_quad(x,y,z,phi,l,-np.sqrt(self.inner_r**2-(l-self.outer_r)**2),np.sqrt(self.inner_r**2-(l-self.outer_r)**2)),
                                0, 2*np.pi,
                                lambda phi: self.outer_r-self.inner_r, lambda phi: self.outer_r+self.inner_r)[0]
        ay = multiplier*dblquad(lambda l,phi:self.torus_y_quad(x,y,z,phi,l,-np.sqrt(self.inner_r**2-(l-self.outer_r)**2),np.sqrt(self.inner_r**2-(l-self.outer_r)**2)),
                                0, 2*np.pi,
                                lambda phi: self.outer_r-self.inner_r, lambda phi: self.outer_r+self.inner_r)[0]
        az = multiplier*dblquad(lambda l,phi:self.torus_z_quad(x,y,z,phi,l,-np.sqrt(self.inner_r**2-(l-self.outer_r)**2),np.sqrt(self.inner_r**2-(l-self.outer_r)**2)),
                                0, 2*np.pi,
                                lambda phi: self.outer_r-self.inner_r, lambda phi: self.outer_r+self.inner_r)[0]
        
        return np.array((vx, vy, vz, ax, ay, az))
    
    def torus_x_quad(self, x,y,z,phi,l,h1,h2):
        v=(x-l*np.cos(phi))**2+(z-l*np.sin(phi))**2
        a=(h1-y)/(v*np.sqrt((h1-y)**2+v))
        b=(h2-y)/(v*np.sqrt((h2-y)**2+v))
        int=b-a
        return int*l*(l*np.cos(phi)-x)
    
    def torus_y_quad(self,x,y,z,phi,l,h1,h2):
        a= -l / np.sqrt((l*np.cos(phi)-x)**2+(h1-y)**2+(l*np.sin(phi)-z)**2)
        b= -l / np.sqrt((l*np.cos(phi)-x)**2+(h2-y)**2+(l*np.sin(phi)-z)**2)
        return b-a

    def torus_z_quad(self, x,y,z,phi,l,h1,h2):
        v=(x-l*np.cos(phi))**2+(z-l*np.sin(phi))**2
        a=(h1-y)/(v*np.sqrt((h1-y)**2+v))
        b=(h2-y)/(v*np.sqrt((h2-y)**2+v))
        int=b-a
        return int*l*(l*np.sin(phi)-z)
    