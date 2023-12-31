import numpy as np

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        # [1]*5 => [1,1,1,1,1]
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            # assign 0 when start=None.
            start = start or 0
        else:
            # np.random.randint(low, high=None, size=None, ...) => Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high). If high is None (the default), then results are from [0, low).
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y