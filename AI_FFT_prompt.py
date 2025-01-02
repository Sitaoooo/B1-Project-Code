import math
import cmath
import numpy as np

def fft_prompt_v1(signal):
    """
    Compute the Fast Fourier Transform (FFT) of a list of complex numbers using
    the Cooley-Tukey algorithm.

    Parameters:
    signal (list of complex or np.ndarray): The input signal in the time domain. The length of the signal must be a power of 2.

    Returns:
    np.ndarray: The FFT of the input signal, representing the frequency domain.
    """
    signal = np.asarray(signal, dtype=complex)
    n = signal.shape[0]

    # Base case: if the input length is 1, return the signal itself
    if n == 1:
        return signal

    # Check if the length of the signal is a power of 2
    if n % 2 != 0:
        raise ValueError("The length of the input signal must be a power of 2.")

    # Divide step: separate the signal into even and odd indexed elements
    even = fft_prompt_v1(signal[0::2])
    odd = fft_prompt_v1(signal[1::2])

    # Combine step: calculate the FFT based on the formula
    result = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        twiddle_factor = cmath.exp(-2j * math.pi * k / n)  # W_N^k = e^(-2*pi*i*k/N)
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + n // 2] = even[k] - twiddle_factor * odd[k]

    return result



def fft_prompt_v2(signal):
    """
    Computes the Fast Fourier Transform (FFT) of a given signal using the Cooley-Tukey algorithm.

    Parameters:
        signal (list of complex): A list of complex numbers representing the input signal in the time domain.

    Returns:
        list of complex: A list of complex numbers representing the FFT of the input signal in the frequency domain.
    """
    n = len(signal)

    # Base case: if the input contains only one element, return it
    if n == 1:
        return signal

    # Ensure the input length is a power of 2
    if n % 2 != 0:
        raise ValueError("Length of input signal must be a power of 2.")

    # Divide step: split the signal into even and odd parts
    even = fft_prompt_v2(signal[0::2])
    odd = fft_prompt_v2(signal[1::2])

    # Combine step
    result = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        # Calculate the twiddle factor
        twiddle = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + twiddle
        result[k + n // 2] = even[k] - twiddle

    return result

def fft_prompt_v3(signal):
    """
    Compute the Fast Fourier Transform (FFT) of a signal using the Cooley-Tukey algorithm.

    Parameters:
        signal (list or np.ndarray): A list of complex numbers representing the time-domain signal.

    Returns:
        np.ndarray: A list of complex numbers representing the frequency-domain signal.
    """
    signal = np.asarray(signal, dtype=complex)
    n = signal.shape[0]

    # Base case: if the input contains only one element, return it
    if n == 1:
        return signal

    # Ensure the input length is a power of 2
    if n % 2 != 0:
        raise ValueError("The length of the input signal must be a power of 2.")

    # Divide the signal into even and odd indexed elements
    even = fft_prompt_v3(signal[0::2])
    odd = fft_prompt_v3(signal[1::2])

    # Precompute the twiddle factors
    twiddle_factors = [cmath.exp(-2j * math.pi * k / n) for k in range(n // 2)]

    # Combine the results
    combined = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        t = twiddle_factors[k] * odd[k]
        combined[k] = even[k] + t
        combined[k + n // 2] = even[k] - t

    return combined