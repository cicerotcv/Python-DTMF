# coding utf8
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
from suaBibSignal import *
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf
import peakutils
from math import sqrt

frequencies = {
    "0": [941, 1336],
    "1": [697, 1209],
    "2": [697, 1336],
    "3": [697, 1477],
    "4": [770, 1209],
    "5": [770, 1336],
    "6": [770, 1477],
    "7": [852, 1209],
    "8": [852, 1336],
    "9": [852, 1477],
    "#": [941, 1477],
    "X": [941, 1209],
    "A": [697, 1633],
    "B": [770, 1633],
    "C": [852, 1633],
    "D": [941, 1633],
}
fs = 44100  # pontos por segundo (frequência de amostragem)
A = 1.5   # Amplitude
T = 3  # Tempo em que o seno será gerado
t = np.linspace(-T/2, T/2, int(T*fs))


class Wave():
    def __init__(self, char: str):
        self.f1 = frequencies[str(char)][0]
        self.f2 = frequencies[str(char)][1]
        self.char = str(char)

        self.x, self.y1 = generateSin(self.f1, T, fs)
        self.x, self.y2 = generateSin(self.f2, T, fs)
        self.y = self.y1 + self.y2
        self.yFiltrado = LPF(self.y, 995, fs)

    def plot(self):

        fig, ax = plt.subplots(ncols=2, figsize=(9, 3))

        # Frequências isoladas
        ax[0].set_xlim([0, T/200])
        ax[0].plot(t, self.y1, 'r--', label=f"F1: {self.f1} Hz")
        ax[0].plot(t, self.y2, 'b', label=f"F2: {self.f2} Hz")
        ax[0].grid()
        ax[0].set_title(f"Sinal {self.char}")
        ax[0].set_xlabel("t (s)")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend(loc="upper left")

        ax[1].grid()
        ax[1].plot(t, self.y, 'b', label="Soma")
        ax[1].legend()
        ax[1].set_title(f"Sinal {self.char}")
        ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("Amplitude")
        ax[1].set_xlim(0, T/200)

#         plt.figure()
#         plt.plot(t, self.yFiltrado, 'b')
#         plt.xlim(0, T/200)
#         plt.grid()
#         plt.title('Filtrado no tempo')

    def play(self, times):
        print(f"'{self.char}': {self.f1} Hz - {self.f2} Hz")
        for i in range(times):
            sd.play(self.y)
            sd.wait()

    def match(self, array):
        f1, f2 = array
        return sqrt((f1 - self.f1)**2 + (f2 - self.f2)**2)


def shift(sound):
    lst = list(sound[:, 0])
    for pos, y in enumerate(lst):
        if y <= -1e-4 or y >= 1e-4:
            return sound[pos:]
    return sound


def calc_peaks(sound, min_dist=50,):
    yAudio = sound[:, 1]
    X, Y = calcFFT(yAudio, fs)
    for i in range(1, 10):
        index = peakutils.indexes(
            np.abs(Y), thres=0.1*(10-i), min_dist=min_dist)
        # print(X[index])
        if len(index) >= 4:
            plt.figure("Fourier Audio", figsize=(9, 3))
            plt.title(f"Melhor ajuste de threshold: {0.1*(10-i)}")
            plt.plot(X, np.abs(Y))
            plt.grid()
            results = X[index]
            return results[(len(index) % 2) + len(results)//2:].T


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def record_audio(duration):  # duration in seconds
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    print(f"recording shape: {myrecording.shape}")
    return shift(myrecording)


def match_char(peaks):
    x0, y0 = peaks
    min = 1000
    match = ''
    for char, (x1, y1) in frequencies.items():
        curr_dist = dist(x1, y1, x0, y0)
        if curr_dist < min:
            min = curr_dist
            match = char
    return match.upper()


def generateSin(freq, time, fs):
    n = time*fs  # numero de pontos
    x = np.linspace(0, time, int(n))  # eixo do tempo
    s = np.sin(freq*x*2*np.pi)
    return (x, s)


def LPF(signal, cutoff_hz, fs):
    from scipy import signal as sg
    #####################
    # Filtro
    #####################
    # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
    nyq_rate = fs/2
    width = 5.0/nyq_rate
    ripple_db = 120.0  # dB
    N, beta = sg.kaiserord(ripple_db, width)
    taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    return(sg.lfilter(taps, 1.0, signal))


def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N = len(signal)
    T = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, fftshift(yf))
