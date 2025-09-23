import dpnp as np
from time import perf_counter

# Set up a aquare array of size N
N = 8192
x = np.random.randn(N,N)

# warmup
for i in range(10):
    y = np.matmul(x,x)

# launch and time the kernel in the default async mode
times_async = []
for i in range(10):
    tic = perf_counter()
    y = np.matmul(x,x)
    toc = perf_counter()
    times_async.append(toc-tic)
print(f'Async execution time: {sum(times_async)/len(times_async):>.4e} sec')
print('WARNING: kernel launch time only!\n')

# launch and time the kernel forcing CPU-GPU sync
times_sync = []
for i in range(10):
    tic = perf_counter()
    y = np.matmul(x,x)
    y.sycl_queue.wait() # sync the CPU and GPU by waiting on the queue to return
    toc = perf_counter()
    times_sync.append(toc-tic)
print(f'Sync execution time: {sum(times_sync)/len(times_sync):>.4e} sec')
print('This is the real runtime of the kernel!')

# Expected output:
#	Async execution time: 3.8199e-04 sec
#	WARNING: kernel launch time only!
#
#	Sync execution time: 3.5182e-01 sec
#	This is the real runtime of the kernel!

