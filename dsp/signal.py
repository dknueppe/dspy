#%%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def linspace(start, stop, dt):
    return np.linspace(start, stop, int(round((stop - start)/dt)), endpoint=True)

class Signal:

    version = '0.1'
    
    def __init__(self, x_val, y_val, domain='time', title='Signal'):
        self.x_val = x_val
        self.y_val = y_val
        self.domain = domain
        self.dt = x_val[1] - x_val[0]
        self.fs = 1/self.dt
        self.title = title

    def fft(self):
        x = np.fft.fftshift(np.fft.fftfreq(self.x_val.size, self.dt))
        y = np.fft.fftshift(np.fft.fft(self.y_val))
        return Signal(x, y, domain='frequency', title='Spectrum')

    def ifft(self):
        x = np.fft.fftshift()

    def conv(self, sig):
        filtered = signal.convolve(self.y_val, sig.y_val, mode='full')
        x_min = min(self.x_val) + min(sig.x_val)
        x_max = max(self.x_val) + max(sig.x_val)
        common_x = np.linspace(x_min, x_max, self.x_val.size + sig.x_val.size -1)
        return Signal(common_x, filtered)

    def pad(self, common_x, fill = None):
        padding = common_x.size -self.x_val.size
        if self.domain == 'time':
            front_pad = int(round(abs((common_x[0] - self.x_val[0])) / np.mean(np.diff(self.x_val))))
        else :
            front_pad = padding // 2
        back_pad = padding - front_pad
        return np.pad(self.y_val, (front_pad, back_pad), 'constant', constant_values=(fill, fill))

    @classmethod
    def from_func(cls, func, start, stop, dt, **kwargs):
        x = linspace(start, stop, dt)
        y = func(x)
        return Signal(x, y)

    @staticmethod
    def common_time(signals):
        if isinstance(signals, Signal):
            return signals.x_val
        x_min = signals[0].x_val[0]
        x_max = signals[0].x_val[-1]
        for signal in signals:
            if signal.domain == 'time':
                x_min = signal.x_val[0] if x_min > signal.x_val[0] else x_min
                x_max = signal.x_val[-1] if x_max < signal.x_val[-1] else x_max
        return np.linspace(x_min, x_max, round((x_max - x_min)/np.mean(np.diff(signals[0].x_val))) + 1)
        #return linspace(x_min, x_max, signals[0].dt)
    
    @staticmethod
    def add(signals):
        common_x = Signal.common_time(signals)
        sig_sum = np.zeros(common_x.size)
        for signal in signals:
            sig_sum += signal.pad(common_x, 0)
        return Signal(common_x, sig_sum)

    @staticmethod
    def plot(signals, columns=1, *args, **kwargs):
        if isinstance(signals, Signal):
            fig = plt.figure(figsize=(10,90/21))
            if signals.domain == 'frequency':
                plt.plot(signals.x_val, abs(signals.y_val)/signals.y_val.size)
            else:
                plt.plot(signals.x_val, signals.y_val)
            plt.grid(True)
            plt.title(signals.title)

        else:
            common_x = Signal.common_time(signals)
            n = len(signals)
            fig, subs = plt.subplots(int(n/columns), columns, figsize=(10,90/21*n/columns), sharex='col')

            for signal, subplot in zip(signals, subs):
                y = signal.pad(common_x)
                subplot.set_title(signal.title)
                if signal.domain == 'frequency':
                    subplot.plot(signal.x_val, abs(y)/y.size)
                else:
                    subplot.plot(common_x, y)
                subplot.grid(True)
    
        fig.tight_layout()
        fig.show()

# Test for custom stuff
f = lambda t: 5* np.sin(2*np.pi*t)
t0 = linspace(-2*np.pi, 2*np.pi, 0.1)
t1 = linspace(-2*np.pi, 0, 0.1)
y0 = np.sin(t0)
y1 = np.cos(t1)
foo = Signal(t0, y0 * 5, title='Sinus')
bar = Signal(t1, y1, title='Cosinus')
foobar = Signal.from_func(f, -3*np.pi, 3*np.pi, 0.1)
print(foobar.dt)
t2 = linspace(-3*np.pi, 3*np.pi, 0.1)
test = Signal(t2, f(t2))
selection = (foo, bar, foobar, test, Signal.add((foo, test)))
Signal.plot(selection)
Signal.plot(test)
Signal.plot(test.fft())
Signal.plot((test, test.fft()),columns=2)

#DTMF
upper_freq = (1029, 1336, 1477)
lower_freq = (697, 770, 852, 941)
keys = {1=(0, 0)}


#%%
import sounddevice as sd
import numpy as np

fs = 44100
duration = 3
tone = 20
t = np.linspace(0, tone*duration, fs*duration)
output = 2 * np.sin(2*np.pi*tone*t)
sd.play(output, fs)