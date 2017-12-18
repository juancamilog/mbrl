import gym
import numpy as np

from scipy.integrate import ode

class ODEEnv(gym.Env):
    def __init__(self, name='ODEEnv', integrator='dopri5',
                 atol=1e-12, rtol=1e-12,
                 *args, **kwargs):
        super(ODEEnv, self).__init__(name=name, *args, **kwargs)
        integrator = kwargs.get('integrator', 'dopri5')
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)

        # initialize ode solver
        self.solver = ode(self.dynamics).set_integrator(integrator,
                                                        atol=atol,
                                                        rtol=rtol)

    def set_state(self, state):
        if self.state is None or\
           np.linalg.norm(np.array(state)-np.array(self.state)) > 1e-12:
            # float64 required for the ode integrator
            self.state = np.array(state, dtype=np.float64).flatten()
        # set solver internal state
        self.solver = self.solver.set_initial_value(self.state)
        # get time from solver
        self.t = self.solver.t

    def _step(self, action):
        self.apply_control(action)
        dt = self.dt
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t + dt)
        self.state = np.array(self.solver.y)
        self.t = self.solver.t
        cost = None
        if self.loss_func is not None:
            cost = self.loss_func(self.state)
        return self.state, cost, False, {}

    def dynamics(self, *args, **kwargs):
        msg = "You need to implement self.dynamics in the ODEEnv subclass."
        raise NotImplementedError(msg)
