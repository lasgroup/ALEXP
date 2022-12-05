import numpy as np

def _handle_input_dimensionality(x, y=None):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)

    assert x.ndim == 2

    if y is not None:
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 2

        return x, y
    else:
        return x