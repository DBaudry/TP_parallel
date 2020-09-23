import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool


def MC_pi(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform(size=2*N).reshape((N, 2))
    count = (np.sqrt((u[:, 0]-0.5)**2+(u[:, 1]-0.5)**2) < 0.5).sum()
    return 4 * count / N, count


def agg(results, N):
    n_cpu = len(results)
    count2 = 0
    for x in results:
        count2 += x[1]
    return 4*count2/n_cpu/N


def reseed(myseed):
    np.random.seed(myseed)


def main(n_cpu, N, seeds=None):
    # with multiprocessing
    if seeds is None:
        seeds = np.arange(n_cpu+1)
    pool = Pool(processes=n_cpu)  # multiprocessing.cpu_count() for all cpu
    n_per_unit = [N for i in range(n_cpu)]
    print(n_per_unit)
    pool.map(reseed, seeds)
    res2 = pool.map(MC_pi, n_per_unit)
    pool.close()
    pool.join()
    print(res2)
    return res2

if __name__ == "__main__":
    N = 1000000
    n_cpu = 5

    # with Multiprocessing
    print("results with multiprocessing")
    res2 = main(n_cpu, N, seeds=np.random.randint(100000, size=n_cpu))
    print("Aggregated estimate: %f" % (agg(res2, N)))
    print('___________')

    # with joblib
    print("results with joblib")
    res = Parallel(n_jobs=n_cpu)(delayed(MC_pi)(N) for _ in range(n_cpu))
    print(res)
    print("Aggregated estimate: %f" % (agg(res, N)))