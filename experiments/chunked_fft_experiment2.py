import time

import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da

import pyxu.util as pxu
from pyxu.operator import FFT



# Creating a LocalCluster with dashboard enabled
cluster = LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=":8787")
client = Client(cluster)

real = True
nrealisations = 5

times = np.zeros((nrealisations, 2))
shape = (2304, 2304, 400)
dtype = "float32"
chunk_size = (576, 576, 50)

for r in range(nrealisations):
    print(f"Realization {r}")
    rng = np.random.default_rng(r)

    if real:
        x = rng.random(size=shape).astype(dtype)
    else:
        x = rng.random(size=shape).astype(dtype) + 0j
        x = pxu.view_as_real(x)
        chunk_size = chunk_size + (2,)


    x_da = da.asarray(x).rechunk(chunk_size)
    if r == 0:
        print(f"Chunksize = {x_da.chunksize}")


    fftc = FFT(
        arg_shape=shape,
        real=real,
        chunked=True,
    )
    start = time.perf_counter()
    y_da = fftc.apply(x_da).compute()
    end = time.perf_counter()
    times[r, 0] = end - start

np.save("times", times)