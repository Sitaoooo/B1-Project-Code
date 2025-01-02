import numpy as np
from memory_profiler import profile

@profile
def manual_fft(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = manual_fft(x[::2])
        X_odd = manual_fft(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X
    

if __name__ == "__main__":
    # Test the manual_fft function
    x = np.random.rand(8)
    result = manual_fft(x)
    print("x: ", x)
    print("result: ", result)