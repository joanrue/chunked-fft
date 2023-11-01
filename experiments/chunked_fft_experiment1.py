import time

import numpy as np
import dask.array as da

import pyxu.util as pxu
from pyxu.operator import FFT

real = True
scales = np.arange(14, 18)
ndims = [2,]
nrealisations = 5

times = np.zeros((len(scales), len(ndims), nrealisations, 2))
for s_id, scale in enumerate(scales):
    dim = 2 ** scale

    for n_id, ndim in enumerate(ndims):
        shape = (dim, ) * ndim

        print(f"\n {shape=}")
        chunk_size = ("auto", ) * len(shape)

        for r in range(nrealisations):
            print(f"Realization {r}")
            rng = np.random.default_rng(r)

            if real:
                x = rng.random(size=shape)
            else:
                x = rng.random(size=shape) + 0j
                x = pxu.view_as_real(x)
                chunk_size = chunk_size + (2,)


            x_da = da.asarray(x).rechunk(chunk_size)
            if r == 0:
                print(f"Chunksize = {x_da.chunksize}")


            try:
                fftc = FFT(
                    arg_shape=shape,
                    real=real,
                    chunked=True,
                )
                start = time.perf_counter()
                y_da = fftc.apply(x_da).compute()
                end = time.perf_counter()
                times[s_id, n_id, r,0] = end - start
            except Exception:
                times[s_id, n_id, r, 0] = np.nan

            try:
                fftc = FFT(
                    arg_shape=shape,
                    real=real,
                    chunked=False,
                )
                start = time.perf_counter()
                y = fftc.apply(x)
                end = time.perf_counter()
                times[s_id, n_id, r, 1] = end - start
            except Exception:
                times[s_id, n_id, r, 1] = np.nan
np.save("times", times)