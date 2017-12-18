# pylint: disable=C0103
'''
Contains the Cartpole envionment, along with default parameters and
a rendering class
'''
import numpy as np

from gym import spaces
from matplotlib import pyplot as plt

from ode import ODEEnv
from mpl_draw import MPLRenderer


class Cartpole(ODEEnv):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, pole_length=0.5, pole_mass=0.5,
                 cart_mass=0.5, friction=0.1, gravity=9.82,
                 state0_dist=None,
                 loss_func=None,
                 name='Cartpole',
                 *args, **kwargs):
        super(Cartpole, self).__init__(name=name, *args, **kwargs)
        # cartpole system parameters
        self.l = pole_length
        self.m = pole_mass
        self.M = cart_mass
        self.b = friction
        self.g = gravity

        # initial state
        if state0_dist is None:
            self.state0_dist = utils.distributions.Gaussian(
                [0, 0, 0, 0], (0.1**2)*np.eye(4))
        else:
            self.state0_dist = state0_dist

        # reward/loss function
        if loss_func is None:
            self.loss_func = cost.build_loss_func(cartpole_loss, False,
                                                  'cartpole_loss')
        else:
            self.loss_func = cost.build_loss_func(loss_func, False,
                                                  'cartpole_loss')

        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None

        # 4 state dims (x ,x_dot, theta_dot, theta)
        o_lims = np.array([np.finfo(np.float).max for i in range(4)])
        self.observation_space = spaces.Box(-o_lims, o_lims)
        # 1 action dim (x_force)
        a_lims = np.array([np.finfo(np.float).max for i in range(1)])
        self.action_space = spaces.Box(-a_lims, a_lims)

    def dynamics(self, t, z):
        l, m, M, b, g = self.l, self.m, self.M, self.b, self.g
        f = self.u if self.u is not None else np.array([0])

        sz, cz = np.sin(z[3]), np.cos(z[3])
        cz2 = cz*cz
        a0 = m*l*z[2]*z[2]*sz
        a1 = g*sz
        a2 = f[0] - b*z[1]
        a3 = 4*(M+m) - 3*m*cz2

        dz = np.zeros((4, 1))
        dz[0] = z[1]                                      # x
        dz[1] = (2*a0 + 3*m*a1*cz + 4*a2)/a3              # dx/dt
        dz[2] = -3*(a0*cz + 2*((M+m)*a1 + a2*cz))/(l*a3)  # dtheta/dt
        dz[3] = z[2]                                      # theta

        return dz

    def _reset(self):
        state0 = self.state0_dist.sample()
        self.set_state(state0)
        return self.state

    def _render(self, mode='human', close=False):
        if self.renderer is None:
            self.renderer = CartpoleDraw(self)
            self.renderer.init_ui()
        self.renderer.update(*self.get_state(noisy=False))

    def _close(self):
        if self.renderer is not None:
            self.renderer.close()


class CartpoleDraw(MPLRenderer):
    def __init__(self, cartpole_plant, refresh_period=(1.0/240),
                 name='CartpoleDraw'):
        super(CartpoleDraw, self).__init__(cartpole_plant,
                                           refresh_period, name)
        l = self.plant.l
        m = self.plant.m
        M = self.plant.M

        self.mass_r = 0.05*np.sqrt(m)  # distance to corner of bounding box
        self.cart_h = 0.5*np.sqrt(M)

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        cart_xy = (self.center_x-0.5*self.cart_h,
                   self.center_y-0.125*self.cart_h)
        self.cart_rect = plt.Rectangle(cart_xy, self.cart_h,
                                       0.25*self.cart_h, facecolor='black')
        self.pole_line = plt.Line2D((self.center_x, 0), (self.center_y, l),
                                    lw=2, c='r')
        self.mass_circle = plt.Circle((0, l), self.mass_r, fc='y')

    def init_artists(self):
        self.ax.add_patch(self.cart_rect)
        self.ax.add_patch(self.mass_circle)
        self.ax.add_line(self.pole_line)

    def _update(self, state, t, *args, **kwargs):
        l = self.plant.l

        cart_x = self.center_x + state[0]
        cart_y = self.center_y
        if self.plant.angle_dims:
            mass_x = l*state[3] + cart_x
            mass_y = -l*state[4] + cart_y
        else:
            mass_x = l*np.sin(state[3]) + cart_x
            mass_y = -l*np.cos(state[3]) + cart_y

        self.cart_rect.set_xy((cart_x-0.5*self.cart_h,
                               cart_y-0.125*self.cart_h))
        self.pole_line.set_xdata(np.array([cart_x, mass_x]))
        self.pole_line.set_ydata(np.array([cart_y, mass_y]))
        self.mass_circle.center = (mass_x, mass_y)

        return (self.cart_rect, self.pole_line, self.mass_circle)
