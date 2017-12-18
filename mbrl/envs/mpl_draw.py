import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.colors import cnames
from multiprocessing import Process, Pipe, Event
from time import time, sleep
from threading import Thread

class MPLRenderer(object):
    def __init__(self, mpl, refresh_period=(1.0/240),
                 name='MPLRenderer', *args, **kwargs):
        super(MPLRenderer, self).__init__()
        self.name = name
        self.env = env
        self.drawing_thread = None
        self.polling_thread = None

        self.dt = refresh_period
        self.exec_time = time()
        self.scale = 150  # pixels per meter

        self.center_x = 0
        self.center_y = 0
        self.running = Event()

        self.polling_pipe, self.drawing_pipe = Pipe()

    def init_ui(self):
        self.fig = plt.figure(self.name)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        self.ax = plt.gca()
        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)
        self.init_artists()
        plt.ion()
        plt.show()

    def drawing_loop(self, drawing_pipe):
        # start the matplotlib plotting
        self.init_ui()

        while self.running.is_set():
            exec_time = time()
            # get any data from the polling loop
            updts = None
            while drawing_pipe.poll():
                data_from_env = drawing_pipe.recv()
                if data_from_env is None:
                    self.running.clear()
                    break

                # get the visuzlization updates from the latest state
                state, t = data_from_env
                updts = self.update(state, t)
                self.update_canvas(updts)

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.close()

    def close(self):
        # close the matplotlib windows, clean up
        plt.ioff()
        plt.close(self.fig)

    def update(self, *args, **kwargs):
        updts = self._update(*args, **kwargs)
        self.update_canvas(updts)

    def _update(self, *args, **kwargs):
        msg = "You need to implement the self._update() method in your\
 MPLRenderer class."
        raise NotImplementedError(msg)

    def init_artists(self, *args, **kwargs):
        msg = "You need to implement the self.init_artists() method in your\
 MPLRenderer class."
        raise NotImplementedError(msg)

    def update_canvas(self, updts):
        if updts is not None:
            # update the drawing from the env state
            self.fig.canvas.restore_region(self.bg)
            for artist in updts:
                self.ax.draw_artist(artist)
            self.fig.canvas.blit(self.ax.bbox)
            # sleep to guarantee the desired frame rate
            exec_time = time() - self.exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.exec_time = time()

    def polling_loop(self, polling_pipe):
        current_t = -1
        while self.running.is_set():
            exec_time = time()
            state, t = self.env.get_state()
            if t != current_t:
                polling_pipe.send((state, t))

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time, 0))

    def start(self):
        print_with_stamp('Starting drawing loop', self.name)
        self.drawing_thread = Process(target=self.drawing_loop,
                                      args=(self.drawing_pipe, ))
        self.drawing_thread.daemon = True
        self.polling_thread = Thread(target=self.polling_loop,
                                     args=(self.polling_pipe, ))
        self.polling_thread.daemon = True
        # self.drawing_thread = Process(target=self.run)
        self.running.set()
        self.polling_thread.start()
        self.drawing_thread.start()

    def stop(self):
        self.running.clear()

        if self.drawing_thread is not None and self.drawing_thread.is_alive():
            # wait until thread stops
            self.drawing_thread.join(10)

        if self.polling_thread is not None and self.polling_thread.is_alive():
            # wait until thread stops
            self.polling_thread.join(10)

        print_with_stamp('Stopped drawing loop', self.name)


# an example that plots lines
class LivePlot(MPLRenderer):
    def __init__(self, env, refresh_period=1.0,
                 name='Serial Data', H=5.0, angi=[]):
        super(LivePlot, self).__init__(env, refresh_period, name)
        self.H = H
        self.angi = angi
        # get first measurement
        state, t = env.get_state()
        self.data = np.array([state])
        self.t_labels = np.array([t])

        # keep track of latest time stamp and state
        self.current_t = t
        self.previous_update_time = time()
        self.update_period = refresh_period

    def init_artists(self):
        self.lines = [plt.Line2D(self.t_labels, self.data[:, i],
                                 c=next(color_generator)[0])
                      for i in range(self.data.shape[1])]
        self.ax.set_aspect('auto', 'datalim')
        for line in self.lines:
            self.ax.add_line(line)
        self.previous_update_time = time()

    def update(self, state, t):
        if t != self.current_t:
            if len(self.data) <= 1:
                self.data = np.array([state]*2)
                self.t_labels = np.array([t]*2)

            if len(self.angi) > 0:
                state[self.angi] = (state[self.angi]+np.pi) % (2*np.pi) - np.pi

            self.current_t = t
            # only keep enough data points to fill the window to avoid using
            # up too much memory
            curr_time = time()
            self.update_period = 0.95*self.update_period + \
                0.05*(curr_time - self.previous_update_time)
            self.previous_update_time = curr_time
            history_size = int(1.5*self.H/self.update_period)
            self.data = np.vstack((self.data, state))[-history_size:, :]
            self.t_labels = np.append(self.t_labels, t)[-history_size:]

            # update the lines
            for i in range(len(self.lines)):
                self.lines[i].set_data(self.t_labels, self.data[:, i])

            # update the plot limits
            plt.xlim([self.t_labels.min(), self.t_labels.max()])
            plt.xlim([t-self.H, t])
            mm = self.data.mean()
            ll = 1.05*np.abs(self.data[:, :]).max()
            plt.ylim([mm-ll, mm+ll])
            self.ax.autoscale_view(tight=True, scalex=True, scaley=True)

        return self.lines
