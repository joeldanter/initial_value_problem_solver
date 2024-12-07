
import numpy as np
from scipy.integrate import dblquad
from abc import ABC, abstractmethod
import scipy.optimize


# TODO: terrible data structures
# TODO: add energy calculations
class System(ABC):
    def __init__(self, init_state):
        self.states=[[0, init_state]]
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
    # TODO: not sure if this is right
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
    