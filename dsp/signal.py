#%%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def linspace(start, stop, dt):
    return np.linspace(start, stop, int(round((stop - start)/dt)) + 1, endpoint=True)

def ifftfreq(freq_min, freq_max, time_frame, n=50):
    print("ifftfreq param: ", freq_min, freq_max, time_frame, n)
    return linspace(time_frame[0], time_frame[1], 1 / ((freq_max - freq_min) / n))

class Signal:
    """Represents a signal in either time or frequency domain
    """
    version = '0.1'
    
    def __init__(self, x_val, y_val, domain='time', title='Signal', time_frame=None):
        if x_val.size != y_val.size:
            raise ValueError("mismatching length of amplitude and time/frequency x = %d, y = %d", x_val.size, y_val.size)
        self.x_val = x_val
        self.y_val = y_val
        self.domain = domain
        self.dt = np.mean(np.diff(x_val))
        self.fs = 1/self.dt
        self.title = title
        self.time_frame = (x_val[0], x_val[-1]) if domain == 'time' else time_frame

    def __add__(self, signal):
        common_x = Signal.common_time(self, signal)
        sig_sum = np.zeros(common_x.size)
        sig_sum += self.padded(common_x, 0)
        sig_sum += signal.padded(common_x, 0)
        return Signal(common_x, sig_sum)

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    def fft(self):
        x = np.fft.fftfreq(self.x_val.size, self.dt)
        y = np.fft.fft(self.y_val)
        return Signal(x, y, domain='frequency', title='Spectrum', time_frame=self.time_frame)

    def ifft(self):
        x = self.x_val
        y = np.fft.ifft(self.y_val)
        return Signal(x, y)

    def conv(self, sig):
        convolved = signal.convolve(self.y_val, sig.y_val, mode='full')
        x_min, x_max = self.time_frame
        common_x = np.linspace(x_min, x_max, self.x_val.size + sig.x_val.size -1)
        return Signal(common_x, convolved)

    def padded(self, common_x, fill = None):
        padding = common_x.size - self.x_val.size
        if self.domain == 'time':
            front_pad = int(abs(common_x[0] - self.x_val[0]) // self.dt)
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
    def common_time(*args):
        total_min = min(args[0].x_val)
        total_max = max(args[-1].x_val)
        for signal in args:
            if signal.domain != 'time':
                raise ValueError('no common time for frequency domain signals')
            x_min = min(signal.x_val)
            x_max = max(signal.x_val)
            total_min = x_min if x_min < total_min else total_min
            total_max = x_max if x_max > total_max else total_max
        common_x = linspace(total_min, total_max, args[0].dt)
        return common_x

    @staticmethod
    def plot(*args, **kwargs):
        col= 1
        n = len(args)
        tmp = kwargs.get('columns')
        columns = tmp if tmp is not None else col

        if n == 1:
            fig = plt.figure(figsize=(10,90/21))
            if args[0].domain == 'frequency':
                plt.plot(args[0].x_val, abs(np.fft.fftshift(args[0].y_val))/args[0].y_val.size)
            else:
                plt.plot(args[0].x_val, args[0].y_val)
            plt.grid(True)
            plt.title(args[0].title)

        else:
            common_x = Signal.common_time(*args)
            fig, subs = plt.subplots(int(n/columns), columns, figsize=(10,90/21*n/columns), sharex='col')
            print(subs)

            for signal, subplot in zip(args, subs):
                y = signal.padded(common_x)
                subplot.set_title(signal.title)
                if signal.domain == 'frequency':
                    subplot.plot(np.fft.fftshift(signal.x_val), abs(np.fft.fftshift(y))/y.size)
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
t2 = t0 + 3
y2 = np.zeros(t2.size)
y2[y2.size // 2 :] = 1
foo = Signal(t0, y0 * 5, title='Sinus')
bar = Signal(t2, y2 * 5, title='Heaviside')
foobar = Signal.from_func(f, -3*np.pi, 3*np.pi, 0.1)
Signal.plot(foo)
Signal.plot(foo, bar, foobar, foobar + foo + bar)
#print(foo)
#print(foobar.dt)
t2 = linspace(-3*np.pi, 3*np.pi, 0.1)
test = Signal(t2, f(t2))
#selection = (foo, foo.fft)
Signal.plot(foobar.fft(), columns=2)
#Signal.plot(test)
#Signal.plot(test.fft())
#Signal.plot((test, test.fft()),columns=2)
test_time = linspace(0, 3, 0.0003)
test_freq = np.fft.fftfreq(test_time.size, 1/np.mean(np.diff(test_time)))
print("dt from test_freq = ", np.mean(np.diff(np.fft.fftshift(test_freq))))
test_time_from_freq = ifftfreq(min(test_freq), max(test_freq), (0,3), test_freq.size)
print("test_time_from_freq = ", test_time_from_freq)

#DTMF
upper_freq = (1029, 1336, 1477)
lower_freq = (697, 770, 852, 941)
#keys = {1=(0, 0)}


#%%
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Tone generation
fs = 44100
duration = 3
tone = 1000
t = np.linspace(0, tone*duration, fs*duration)
output = 20 * np.sin(2*np.pi*tone*t)
#sd.play(output, fs)

# playing .wav
FSample, samples = wav.read('/home/d/Downloads/audio_2.wav')
sd.play(samples, FSample)
