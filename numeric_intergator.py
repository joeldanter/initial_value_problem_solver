import numpy as np
import scipy
from abc import ABC, abstractmethod


class NumericIntegrator(ABC):
    def __init__(self, system):
        self.system=system
    
    @abstractmethod
    def iterate_until(self, t_end):
        pass

class ExplicitEuler(NumericIntegrator):
    def __init__(self, system, dt=0.01):
        super().__init__(system)
        self.dt=dt
    
    def step(self):
        state=self.system.last_state()
        new_state = state+self.system.diff_eq(state)*self.dt
        self.system.append_new_state(new_state, self.dt)

    def iterate_until(self, t_end):
        while self.system.t<=t_end:
            self.step()

class ImplicitEuler(NumericIntegrator):
    def __init__(self, system, dt=0.01):
        super().__init__(system)
        self.dt=dt
    
    def step(self):
        state=self.system.last_state()
        f = lambda next_state: next_state - self.dt*self.system.diff_eq(next_state) - state
        new_state = scipy.optimize.newton_krylov(f, state)
        self.system.append_new_state(new_state, self.dt)

    def iterate_until(self, t_end):
        while self.system.t<=t_end:
            self.step()
        
class RK2(NumericIntegrator):
    def __init__(self, system, dt=0.01):
        super().__init__(system)
        self.dt=dt
    
    def step(self):
        state=self.system.last_state()
        k1=self.dt*self.system.diff_eq(state)
        k2=self.dt*self.system.diff_eq(state+k1/2)
        new_state = state+k2
        self.system.append_new_state(new_state, self.dt)

    def iterate_until(self, t_end):
        while self.system.t<=t_end:
            self.step()

class RK4(NumericIntegrator):
    def __init__(self, system, dt=0.01):
        super().__init__(system)
        self.dt=dt
    
    def step(self):
        state=self.system.last_state()
        k1=self.dt*self.system.diff_eq(state)
        k2=self.dt*self.system.diff_eq(state+k1/2)
        k3=self.dt*self.system.diff_eq(state+k2/2)
        k4=self.dt*self.system.diff_eq(state+k3)
        new_state = state+(k1+2*k2+2*k3+k4)/6
        self.system.append_new_state(new_state, self.dt)

    def iterate_until(self, t_end):
        while self.system.t<=t_end:
            self.step()

class RKF(NumericIntegrator):
    def __init__(self, system, tol=1e-6, dt_min=1e-8, dt_max=0.01):
        super().__init__(system)
        self.tol=tol
        self.dt_min=dt_min
        self.dt_max=dt_max

    def iterate_until(self, t_end):
        dt=self.dt_max
        while self.system.t<t_end:
            if t_end<self.system.t+dt:
                dt=t_end-self.system.t
            elif dt<self.dt_min:
                raise Exception('Minimum dt exceeded')
            
            state=self.system.last_state()
            k1=dt*self.system.diff_eq(state)
            k2=dt*self.system.diff_eq(state  +k1/4)
            k3=dt*self.system.diff_eq(state  +3*k1/32        +9*k2/32)
            k4=dt*self.system.diff_eq(state  +1932*k1/2197   -7200*k2/2197   +7296*k3/2197)
            k5=dt*self.system.diff_eq(state  +439*k1/216     -8*k2           +3680*k3/513    -845*k4/4104)
            k6=dt*self.system.diff_eq(state  -8*k1/27        +2*k2           -3544*k3/2565   +1859*k4/4104   -11*k5/40)
            R=np.linalg.norm(k1/360-128*k3/4275-2197*k4/75240+k5/50+2*k6/55)/dt # TODO linalg nemtom jo e
            if R<=self.tol:
                new_state=state+25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
                self.system.append_new_state(new_state, dt)
            
            if R==0:
                delta=4
            else:
                delta=0.84*np.power(self.tol/R, 0.25)
            if delta <= 0.1:
                dt*=0.1
            elif 4<=delta:
                dt*=4
            else:
                dt*=delta
            if self.dt_max<dt:
                dt=self.dt_max
