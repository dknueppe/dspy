#%%
import collections
import math
import matplotlib.pyplot    as     plt
import numpy                as     np
import sounddevice          as     sd
from functools              import singledispatch
from scipy                  import signal
from IPython.display        import Audio

def linspace(start, stop, fs):
    return np.linspace(start, stop, int(round((stop - start) * fs)) + 1, endpoint=True)

TimeFrame = collections.namedtuple('TimeFrame', 'start stop')

class DomainError(Exception):
    pass

class Signal:
    """Represents a signal in either time or frequency domain
    """
    version = '0.2'

    def __init__(self, x_val, y_val, domain='time', title='Signal', time_frame=None):
        if x_val.size != y_val.size:
            raise ValueError("mismatching length of x = {}, y = {}" \
                .format(x_val.size, y_val.size))
        if domain == 'time':
            dt = np.mean(np.diff(x_val))
            self.fs = int(round(1/dt))
        elif domain == 'frequency':
            x_min = min(x_val)
            self.fs = int(round(1./(1. / (-2. * x_min))))
        else:
            raise ValueError("Signal must be either 'frequency' or 'time' domain")
        self.__x_val = x_val
        self.__y_val = y_val
        self.domain = domain
        self.title = title
        self.time_frame = TimeFrame(start=x_val[0], stop=x_val[-1]) \
            if domain == 'time' else time_frame

    @classmethod
    def from_func(cls, func, start, stop, fs, **kwargs):
        x = linspace(start, stop, fs)
        y = func(x)
        return Signal(x, y, time_frame=(start, stop), **kwargs)
    
    @classmethod
    def from_wav(cls, wav, **kwargs):
        title = kwargs.get('title', 'Signal')
        FSample, samples = wav
        t = np.linspace(0, samples.size / FSample, samples.size)
        return Signal(t, samples, title=title)

    def __add__(self, signal):
        common_x, timeframe = Signal.common_time(self, signal)
        sig_sum = np.zeros(common_x.size)
        sig_sum += self.padded(common_x, 0)
        sig_sum += signal.padded(common_x, 0)
        return Signal(common_x, sig_sum, time_frame=timeframe)
    
    def __sub__(self, signal):
        common_x, timeframe = Signal.common_time(self, signal)
        sig_sum = np.zeros(common_x.size)
        sig_sum -= self.padded(common_x, 0)
        sig_sum -= signal.padded(common_x, 0)
        return Signal(common_x, sig_sum, time_frame=timeframe)

    def __mult__(self, signal):
        common_x, timeframe = Signal.common_time(self, signal)
        sig_prod = self.padded(common_x, 0) * signal.padded(common_x, 0)
        return Signal(common_x, sig_prod, time_frame=timeframe)

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    @property
    def dt(self):
        return 1 / self.fs

    @property
    def size(self):
        return self.__x_val.size

    @property
    def fft(self):
        x = np.fft.fftfreq(self.__x_val.size, self.dt)
        y = np.fft.fft(self.__y_val)
        return Signal(x, y, domain='frequency', title='Spectrum', time_frame=self.time_frame)

    @property
    def ifft(self):
        start = 0.
        stop = start + (self.size -1) / self.fs
        x = linspace(start, stop, self.fs)
        y = np.fft.ifft(self.__y_val)
        return Signal(x, y, time_frame=self.time_frame)

    @property
    def amp(self):
        return self.__y_val

    @property
    def time_scale(self):
        if self.domain == 'frequency':
            raise DomainError('wrong domain you fool!')
        return self.__y_val

    @property
    def freq_scale(self):
        if self.domain == 'time':
            raise DomainError('wrong domain you fool!')
        return self.__y_val

    def conv(self, sig):
        convolved = signal.convolve(self.__y_val, sig.__y_val, mode='full')
        x_min, x_max = self.time_frame
        common_x = np.linspace(x_min, x_max, self.__x_val.size + sig.__x_val.size -1)
        return Signal(common_x, convolved)

    def padded(self, common_x, fill=None):
        padding = common_x.size - self.__x_val.size
        if padding == 0:
            return self.__y_val
        if self.domain == 'time':
            front_pad = int(abs(common_x[0] - self.__x_val[0]) // self.dt)
            back_pad = padding - front_pad
            return np.pad(self.__y_val, (front_pad, back_pad), 'constant', constant_values=(fill, fill))
        else :
            front_pad = padding // 2
            back_pad = padding - front_pad
            return np.pad(np.fft.fftshift(self.__y_val), (front_pad, back_pad), 'constant', constant_values=(fill, fill))

    def play_sound(self):
        print(self.domain, self.fs)
        if self.domain == 'frequency' or self.fs >= 50000:
            raise ValueError('cannot play sound from this array (either it is freqdomain or sampled to quick')
        sd.play(self.__y_val, self.fs)
        #Audio(data=self.__y_val, rate=self.fs, autoplay=True)

    @staticmethod
    def common_time(*args):
        total_min = args[0].time_frame.start
        total_max = args[-1].time_frame.stop
        for signal in args:
            if signal.domain == 'time':
               # raise DomainError('no common time for frequency domain signals')
                x_min = signal.time_frame.start
                x_max = signal.time_frame.stop
                total_min = x_min if x_min < total_min else total_min
                total_max = x_max if x_max > total_max else total_max
        common_x = linspace(total_min, total_max, args[0].fs)
        time_frame = TimeFrame(start=total_min, stop=total_max)
        return (common_x, time_frame)

    @staticmethod
    def plot(*args, **kwargs):
        if len(args) <= 1:
            signal = args[0]
            if signal.domain == 'time':
                x = signal.__x_val
                y = signal.__y_val
            elif signal.domain == 'frequency':
                x = signal.__x_val
                y = np.abs(signal.__y_val / signal.size)
            fig = plt.figure()
            plt.title(signal.title)
            plt.plot(x, y)
            plt.grid(True)
            plt.show()
        else:
            cols = kwargs.get('columns', 1)
            sharex = kwargs.get('sharex', False)
            common_x, timeframe = Signal.common_time(*args)
            rows =  math.ceil(len(args) / cols)
            fig, subs = plt.subplots(rows, cols, figsize=(10, 90 / 21 * rows), sharex=sharex)
            for signal, subplot in zip(args, subs.flat):
                if signal.domain == 'time':
                    if sharex == False:
                        x = signal._Signal__x_val
                        y = signal._Signal__y_val
                    elif sharex == True:
                        x = common_x
                        y = signal.padded(common_x)
                    else:
                        return NotImplemented
                elif signal.domain == 'frequency':
                    if sharex == False:
                        x = signal._Signal__x_val
                        y = np.abs(np.fft.fftshift(signal._Signal__y_val / signal.size)) 
                    elif sharex == True:
                        x = np.fft.fftshift(np.fft.fftfreq(common_x.size, signal.dt)) 
                        y = np.abs(signal.padded(common_x) / signal.size)
                else:
                    print('How did we get here?')
                    return NotImplemented
                subplot.plot(x, y)
                subplot.grid(True)
                subplot.set_title(signal.title)
            fig.tight_layout()
            fig.show()


#@singledispatch
#def _plot(*args, **kwargs):
#    if kwargs.pop('verbose', False):
#        print(type(args))
#        print(args)
#
#@_plot.register(Signal)
#def _(args, **kwargs):
#    plt.title(args.title)
#    plt.plot(args._Signal__x_val, args._Signal__y_val)
#    plt.grid(True)
#    plt.show()


#%%
# Test for custom stuff
#f = lambda t: 5* np.sin(2*np.pi*t)
#t0 = linspace(-2*np.pi, 2*np.pi, 0.1)
#t1 = linspace(-2*np.pi, 0, 0.1)
#y0 = np.sin(t0)
#y1 = np.cos(t1)
#t2 = t0 + 3
#y2 = np.zeros(t2.size)
#y2[y2.size // 2 :] = 1
#foo = Signal(t0, y0 * 5, title='Sinus')
#bar = Signal(t2, y2 * 5, title='Heaviside')
#foobar = Signal.from_func(f, -3*np.pi, 3*np.pi, 0.1)
#Signal.plot(foo)
#Signal.plot(foo, bar, foobar, foobar + foo + bar)
#
#def cos_add():
#    t = linspace(-0.1, 0.1, 0.0002)
#    f = lambda k : (1 / (k + 1)) * np.cos(2 * np.pi * k * 250 * t)
#    y = np.zeros(t.size)
#    for n in range(6):
#        y += f(n)
#    return Signal(t, y, time_frame=(-0.1, 0.1))
#fourier = cos_add()
#Signal.plot(fourier)
#fft = fourier.fft
#Signal.plot(fft)

#print(foo)
#print(foobar.dt)
#t2 = linspace(-3*np.pi, 3*np.pi, 0.1)
#test = Signal(t2, f(t2))
#selection = (foo, foo.fft)
#Signal.plot(foobar.fft(), columns=2)
#Signal.plot(test)
#Signal.plot(test.fft())
#Signal.plot((test, test.fft()),columns=2)
#test_time = linspace(0, 3, 0.0003)
#test_freq = np.fft.fftfreq(test_time.size, 1/np.mean(np.diff(test_time)))
#print("dt from test_freq = ", np.mean(np.diff(np.fft.fftshift(test_freq))))
#test_time_from_freq = ifftfreq(min(test_freq), max(test_freq), (0,3), test_freq.size)
#print("test_time_from_freq = ", test_time_from_freq)

#%%
import scipy.io.wavfile as wav

audio = wav.read('/home/daniel/Downloads/audio_1.wav')
audio1 = Signal.from_wav(audio, title='War of the Worlds (interence)')
audio = wav.read('/home/daniel/Downloads/audio_2.wav')
audio2 = Signal.from_wav(audio)
#plt.plot(np.linspace(0,samples.size/FSample, samples.size), samples)
#Signal.plot(audio1, audio2, audio1.fft, audio2.fft)
Signal.plot(audio1, audio1.fft.ifft, columns=1)
Signal.plot(audio2.fft)
#Signal.plot(audio1)
#Signal.plot(audio2)
#Signal.plot(audio1.fft, audio2.fft, columns=1)
audio1.play_sound()

#%%
#DTMF
upper_freq = (1029, 1336, 1477)
lower_freq = (697, 770, 852, 941)
