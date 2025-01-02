import cmath
import math
import numpy as np

def fft_v1(x):
    """
    Compute the discrete Fourier transform of the 1D list/array x
    using the Cooley-Tukey FFT algorithm (recursive version).
    
    Parameters
    ----------
    x : list or np.ndarray of complex or float
        The input signal (time-domain samples).
        
    Returns
    -------
    X : np.ndarray of complex
        The DFT of x, same length as x.
        
    Note
    ----
    - This implementation assumes the length of x is a power of 2.
    - If x is real-valued, convert each sample to a complex number 
      before calling fft_cooley_tukey.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Base case
    if N <= 1:
        return x
    
    # Divide: separate even and odd indices
    X_even = fft_v1(x[0::2])   # FFT of even-indexed terms
    X_odd  = fft_v1(x[1::2])   # FFT of odd-indexed terms
    
    # Prepare output array
    X = np.zeros(N, dtype=complex)
    
    # Combine step
    # Twiddle factors: e^{-2 pi j k / N}
    for k in range(N // 2):
        # The "twiddle" factor:
        t = cmath.exp(-2j * math.pi * k / N) * X_odd[k]
        X[k]         = X_even[k] + t      # "Top" half of the output
        X[k + N//2]  = X_even[k] - t      # "Bottom" half of the output
        
    return X

def fft_v2(signal):
    """
    Compute the FFT of a list of complex samples 'signal' using the
    Cooley-Tukey algorithm (recursive version).
    
    :param signal: List or np.ndarray of complex numbers, typically of length N=2^m.
    :return: np.ndarray of complex numbers representing the FFT of the input.
    """
    signal = np.asarray(signal, dtype=complex)
    N = len(signal)
    
    # Base case
    if N <= 1:
        return signal
    
    # 1) Split into even and odd indexed parts
    even_part = fft_v2(signal[0::2])  # FFT of even indices
    odd_part  = fft_v2(signal[1::2])  # FFT of odd indices
    
    # 2) Combine the two halves
    combined = np.zeros(N, dtype=complex)
    for k in range(N//2):
        # The "twiddle factor" e^(-2πik/N)
        # cmath.exp(...) returns complex exponential
        twiddle = cmath.exp(-2j * cmath.pi * k / N) * odd_part[k]
        
        # Combine results
        combined[k]           = even_part[k] + twiddle
        combined[k + N//2]    = even_part[k] - twiddle
    
    return combined

def fft_v3(signal):
    """
    Compute the FFT of a list of complex samples 'signal' using the
    Cooley-Tukey algorithm (recursive version).
    
    :param signal: List or np.ndarray of complex numbers, typically of length N=2^m.
    :return: np.ndarray of complex numbers representing the FFT of the input.
    """
    signal = np.asarray(signal, dtype=complex)
    n = len(signal)
    
    # Base case: if the signal length is 1, return it
    if n == 1:
        return signal
    
    # Divide: Separate the even and odd indices
    even = fft_v3(signal[0::2])
    odd = fft_v3(signal[1::2])
    
    # Combine: Apply the FFT formula
    combined = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        combined[k] = even[k] + t
        combined[k + n // 2] = even[k] - t
    return combined
