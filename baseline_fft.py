import numpy as np

# Compute FFT without knowing the sampling rate
def baseline_fft(signal):
    # Compute the FFT
    fft_result = np.fft.fft(signal)
    # Compute the normalized frequencies corresponding to the FFT result
    return fft_result