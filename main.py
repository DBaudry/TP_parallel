import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from func import main, MC_pi, agg

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