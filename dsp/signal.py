#%%
from scipy import signal
import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt

def linspace(start, stop, dt):
    return np.linspace(start, stop, int(round((stop - start)/dt)), endpoint=False)

class Signal:

    version = '0.1'
    
    def __init__(self, x_val, y_val, domain='time', title='Signal'):
        self.x_val = x_val
        self.y_val = y_val
        self.domain = domain
        self.dt = np.mean(np.diff(x_val))
        self.fs = 1/self.dt
        self.title = title

    def fft(self):
        return Signal(npfft.fft(self.y_val), npfft.fftfreq(self.y_val.size, self.dt), domain='frequency')

    def conv(self, sig):
        filtered = signal.convolve(self.y_val, sig.y_val, mode='full')
        x_min = min(self.x_val) + min(sig.x_val)
        x_max = max(self.x_val) + max(sig.x_val)
        common_x = np.linspace(x_min, x_max, self.x_val.size + sig.x_val.size -1)
        return Signal(common_x, filtered)

    def pad(self, common_x, fill = None):
        padding = common_x.size -self.x_val.size
        front_pad = int(round(abs((common_x[0] - self.x_val[0])) / np.mean(np.diff(self.x_val))))
        back_pad = padding - front_pad
        #print(common_x[0], common_x[-1], self.x_val[0], self.x_val[-1])
        #print(padding, front_pad, back_pad)
        return np.pad(self.y_val, (front_pad, back_pad), 'constant', constant_values=(fill, fill)) 


    @classmethod
    def from_func(cls, func, start, stop, dt, **kwargs):
        x = linspace(start, stop, dt)
        y = func(x)
        return Signal(x, y)

    @staticmethod
    def common_time(signals):
        t_min = signals[0].x_val[0]
        t_max = signals[0].x_val[-1]
        for signal in signals:
            if signal.domain == 'time':
                t_min = signal.x_val[0] if t_min > signal.x_val[0] else t_min
                t_max = signal.x_val[-1] if t_max < signal.x_val[-1] else t_max
        return np.linspace(t_min, t_max, round((t_max - t_min)/np.mean(np.diff(signals[0].x_val))) + 1)
        #return linspace(t_min, t_max, signals[0].dt)
    
    @staticmethod
    def add(signals):
        common_x = Signal.common_time(signals)
        sig_sum = np.zeros(common_x.size)
        for signal in signals:
            sig_sum += signal.pad(common_x, 0)
        return Signal(common_x, sig_sum)

    @staticmethod
    def plot(signals, columns=1, *args, **kwargs):
        common_x = Signal.common_time(signals)
        n = len(signals)
        fig, subs = plt.subplots(int(n/columns), columns, sharex=True)

        for signal, subplot in zip(signals, subs):
            y = signal.pad(common_x)
            subplot.set_title(signal.title)
            subplot.plot(common_x, y)
            subplot.grid(True)
    
        fig.tight_layout()
        fig.show()

# Test for custom stuff
f = lambda t: 5* np.sin(2*np.pi*t)
t0 = linspace(0, 2*np.pi, 0.1)
t1 = linspace(-2*np.pi, 0, 0.1)
y0 = np.sin(t0)
y1 = np.cos(t1)
foo = Signal(t0, y0 * 5, title='Sinus')
bar = Signal(t1, y1, title='Cosinus')
foobar = Signal.from_func(f, -3*np.pi, 3*np.pi, 0.1)
t2 = linspace(-3*np.pi, 3*np.pi, 0.1)
test = Signal(t2, f(t2))
Signal.plot((foo, test, Signal.add((foo, test))))
#Signal.plot(test.fft())
