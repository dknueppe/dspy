#%%
import collections
import math
import matplotlib.pyplot    as     plt
import numpy                as     np
import sounddevice          as     sd
from functools              import singledispatch
from scipy                  import signal
import scipy.io.wavfile     as     wav
from IPython.display        import Audio

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def linspace(start, stop, fs):
    """Creates a numpy ndarray containing values from start to stop (both inclusive) where
       the difference array[n + 1] - array[n] is equal to the reciprocal of fs
    """
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
    def from_wav(cls, audio, **kwargs):
        title = kwargs.get('title', 'Signal')
        FSample, samples = wav.read(audio)
        t = np.linspace(0, samples.size / FSample, samples.size)
        return Signal(t, samples, title=title)

    def __add__(self, signal):
        if isinstance(signal, Signal):
            common_x, timeframe = Signal.common_time(self, signal)
            sig_sum = np.zeros(common_x.size)
            sig_sum += self.padded(common_x, 0)
            sig_sum += signal.padded(common_x, 0)
            return Signal(common_x, sig_sum,title=self.title + ' + ' + signal.title,\
                time_frame=timeframe)
        else:
            return Signal(self.__x_val, self.__y_val * signal, title=self.title, domain=self.domain)

    def __iadd__(self, signal):
        self.__y_val += signal
        return self
    
    def __sub__(self, signal):
        if isinstance(signal, Signal):
            common_x, timeframe = Signal.common_time(self, signal)
            sig_sum = np.zeros(common_x.size)
            sig_sum -= self.padded(common_x, 0)
            sig_sum -= signal.padded(common_x, 0)
            return Signal(common_x, sig_sum,title=self.title + ' - ' + signal.title,\
                time_frame=timeframe)
        else:
            return Signal(self.__x_val, self.__y_val * signal, title=self.title, domain=self.domain)

    def __isub__(self, signal):
        self.__y_val -= signal
        return self

    def __mul__(self, signal):
        if isinstance(signal, Signal):
            common_x, timeframe = Signal.common_time(self, signal)
            sig_prod = self.padded(common_x, 0) * signal.padded(common_x, 0)
            return Signal(common_x, sig_prod,title=self.title + ' * ' + signal.title,\
                time_frame=timeframe)
        else:
            return Signal(self.__x_val, self.__y_val * signal, title=self.title, domain=self.domain)

    def __imul__(self, signal):
        self.__y_val *= signal
        return self

    def __truediv__(self, signal):
        if isinstance(signal, Signal):
            common_x, timeframe = Signal.common_time(self, signal)
            sig_prod = self.padded(common_x, 0) / signal.padded(common_x, 0)
            return Signal(common_x, sig_prod,title=self.title + ' / ' + signal.title,\
                time_frame=timeframe)
        else:
            return Signal(self.__x_val, self.__y_val / signal, title=self.title, domain=self.domain)

    def __itruediv__(self, signal):
        self.__y_val /= signal
        return self

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
    def max(self):
        return max(self.__x_val)

    @property
    def min(self):
        return min(self.__x_val)

    @property
    def mean(self):
        return np.mean(self.__x_val)
    
    @property
    def fft(self):
        x = np.fft.fftfreq(self.__x_val.size, self.dt)
        y = np.fft.fft(self.__y_val)
        return Signal(x, y, domain='frequency', title='Spectrum of ' + self.title, time_frame=self.time_frame)

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

    def heaviside(self, start = 0, x2 = 1):
        return np.heaviside(self.__x_val - start, x2)

    def rect(self, start, stop):
        return self.heaviside(start) - self.heaviside(stop)

    def dirac(self, t = 0):
        d = np.zeros(self.size)
        i = int(-(self.__x_val[0] - t) * self.fs)
        if i < 0:
            return d
        else:
            d[i] = 1
            return d

    def cancel_freq(self, *args):
        if self.domain == 'time':
            raise DomainError('wrong domain you fool!')
        for freq in args:
            index = ((freq * self.size) // self.fs)
            tmp = self.size // 3000
            amp = (self.__y_val[index - tmp -1] + self.__y_val[index + 1 + tmp]) // 2
            self.__y_val[index - tmp : index + tmp] = amp
            self.__y_val[-index - tmp : -index + tmp] = amp
        return self

    def periodogram(self, block_size=None):
        x, y = signal.periodogram(self.__y_val, fs=self.fs, nfft=block_size)
        return Signal(x, y, domain='frequency', title='Periodogram of ' + self.title)

    def fft_lim(self, lim):
        x = np.fft.fftfreq(self.size, self.dt)
        y = np.fft.fft(self.__y_val)
        index = int(self.size // (self.fs / lim))
        fig = plt.figure(figsize=(12, 90 / 16), dpi=120)
        plt.title('Spectrum of {}, from 0 to {} HZ'.format(self.title, lim))
        plt.stem(x[:index], np.abs(y[:index]) / self.size)
        plt.grid(True)
        fig.show()
        return Signal(x[:index], y[:index], title=self.title, domain='frequency')

    def sample_down(self, r):
        y = self.__y_val[::r]
        x = self.__x_val[::r]
        return Signal(x, y, title=self.title)

    def sample_up(self, r=2):
        if r != 2:
            raise NotImplementedError('sorry!')
        x_tmp = np.zeros(self.size)
        x_tmp[:-1:1] = self.__x_val[:-1:1] + np.diff(self.__x_val) / 2
        y_tmp = np.zeros(self.size)
        y_tmp[:-1:1] = self.__y_val[:-1:1] + np.diff(self.__y_val) / 2
        x_tmp[-1] = self.__x_val[-1]
        y_tmp[-1] = self.__y_val[-1]
        x = (np.stack((self.__x_val, x_tmp))).flatten(order='F')
        y = (np.stack((self.__y_val, y_tmp))).flatten(order='F')
        return Signal(x, y, title=self.title)

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
            return np.pad(self.__y_val, (front_pad, back_pad), \
                'constant', constant_values=(fill, fill))
        else :
            front_pad = padding // 2
            back_pad = padding - front_pad
            return np.pad(np.fft.fftshift(self.__y_val), \
                (front_pad, back_pad), 'constant', \
                constant_values=(fill, fill))

    def play_sound(self):
        if  self.fs >= 50000:
            raise ValueError('frequency to high')
        if self.domain == 'frequency':
            raise DomainError('cannot play frequency domain signals')
        #sd.play(self.__y_val, self.fs)
        Audio(data=self.__y_val, rate=self.fs, autoplay=True)

    def save_as_wav(self, name='Signals.wav'):
        wav.write(name, self.fs, self.__y_val.real)

    def plot_properties(self):
        fig, subs= plt.subplots(3, 2, figsize=(12, 90 / 16), dpi=120)

        titles =  (r'\textbf{Signal}',
                   r'\textbf{Spektrum}',
                   r'\textbf{Realteil}',
                   r'\textbf{Betrag}',
                   r'\textbf{Imaginärteil}',
                   r'\textbf{Phase}')

        xlabels = (r'\textit{t}$[s]$',
                   r'\textit{f}$[kHz]$',
                   r'\textit{t}$[s]$',
                   r'\textit{t}$[s]$',
                   r'\textit{t}$[s]$',
                   r'\textit{t}$[s]$')

        ylabels = ('Amplitude',
                   r'$|H(e^{j{\omega}T}|$',
                   r'$\Re\{f(t)\}$',
                   r'${|f(t)|}$',
                   r'$\Im\{f(t)\}$',
                   r'$\angle\{f(t)\}$')

        values = ((self.__x_val, self.__y_val),
                  (np.fft.fftshift(self.fft.__x_val / 1000), \
                      np.abs(np.fft.fftshift(self.fft.__y_val)) / self.size),
                  (self.__x_val, self.__y_val.real),
                  (self.__x_val, np.abs(self.__y_val)),
                  (self.__x_val, self.__y_val.imag),
                  (self.__x_val, np.angle(self.__y_val.real)))
                  
        for subplot, title, xlabel, ylabel, value in \
            zip(subs.flat, titles, xlabels, ylabels, values):
            subplot.plot(value[0], value[1])
            subplot.set_title(title)
            subplot.set(xlabel=xlabel,ylabel=ylabel)
            subplot.grid(True)
        
        fig.tight_layout()
        fig.show()

    @staticmethod
    def common_time(*args):
        total_min = args[0].time_frame.start
        total_max = args[-1].time_frame.stop
        for signal in args:
            if signal.domain == 'time':
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
            fig = plt.figure(figsize=(12, 90 / 16), dpi=120)
            plt.title(signal.title)
            plt.plot(x, y)
            plt.grid(True)
            plt.show()
        else:
            cols = kwargs.get('columns', 1)
            sharex = kwargs.get('sharex', False)
            common_x, timeframe = Signal.common_time(*args)
            rows =  math.ceil(len(args) / cols)
            fig, subs = plt.subplots(rows, cols, figsize=(10, 90 / 21 * rows), dpi=120, sharex=sharex)
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
                        y = np.abs(signal._Signal__y_val / signal.size)
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

##%%
## Test for custom stuff
#f = lambda t: 5* np.sin(2*np.pi*t)
#t0 = linspace(-2*np.pi, 2*np.pi, 10)
#t1 = linspace(-2*np.pi, 0, 10)
#y0 = np.sin(t0)
#y1 = np.cos(t1)
#t2 = t0 + 3
#y2 = np.zeros(t2.size)
#y2[y2.size // 2 :] = 1
#foo = Signal(t0, y0 * 5, title='Sinus')
#bar = Signal(t2, y2 * 5, title='Heaviside')
#foobar = Signal.from_func(f, -3*np.pi, 3*np.pi, 10)
#Signal.plot(foo)
#Signal.plot(foo, bar, foobar, foobar + foo + bar, sharex=True)
#
#def cos_add():
#    t = linspace(-0.1, 0.1, 5000)
#    f = lambda k : (1 / (k + 1)) * np.cos(2 * np.pi * k * 250 * t)
#    y = np.zeros(t.size)
#    for n in range(6):
#        y += f(n)
#    return Signal(t, y, time_frame=(-0.1, 0.1))
#fourier = cos_add()
#Signal.plot(fourier)
#fft = fourier.fft
#Signal.plot(fft)
#
##%%
#import scipy.io.wavfile as wav
#
#audio = wav.read('/home/daniel/Downloads/audio_1.wav')
#audio1 = Signal.from_wav(audio, title='War of the Worlds (interence)')
#audio = wav.read('/home/daniel/Downloads/audio_2.wav')
#audio2 = Signal.from_wav(audio)
##plt.plot(np.linspace(0,samples.size/FSample, samples.size), samples)
#Signal.plot(audio1, audio2, audio1.fft, audio2.fft)
#Signal.plot(audio1, audio1.fft.ifft, columns=1)
#Signal.plot(audio1.fft)
#audio1.plot_properties()
#Signal.plot(audio1)
#Signal.plot(audio2)
#Signal.plot(audio1.fft, audio2.fft, columns=1)
##audio1.play_sound()
#
##%%
##DTMF
#upper_freq = (1029, 1336, 1477)
#lower_freq = (697, 770, 852, 941)
#


#%%
#audio = Signal.from_wav('/home/daniel/Downloads/audio_1.wav')
#audio_fixed = (audio.fft.cancel_freq(1000, 3000)).ifft
#audio_fixed.plot_properties()
#audio_fixed.save_as_wav("filtered.wav")
#audio_fixed *= 1 / max(abs(audio_fixed.amp))
#Signal.plot(audio, audio.fft, audio_fixed, audio_fixed.fft, columns=2)
##%%
#t = np.linspace(0, 0.016, 2048, endpoint=False)
#y = signal.sawtooth(t * 2 * np.pi * 1000, width=0)
#sig = Signal(t, y, title='Sawtooth')
#sig -= np.mean(sig.amp)
#sig.plot_properties()
#print(np.mean(sig.amp), sig.fs)
#foo = sig.fft_lim(10000)
#
#%%
audio = Signal.from_wav('/home/daniel/Downloads/audio_2.wav')
print(audio.fs)
audio = audio.sample_down(r=2)
audio.plot_properties()
print(audio.fs)
audio = audio.sample_up(r=2)
audio.plot_properties()
print(audio.fs)

#%%
