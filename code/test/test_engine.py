import sys
sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import numpy as np
import LSMC_engine


def test_engine():
    # Market Parameters
    s0 = np.array([40])
    T = 1

    # Simulation parameters
    M = 2 ** 14
    dt = 1 / 252.
    d = 1
    d1 = 1
    d2 = 1

    LSMC_engine_european = LSMC_engine.LSMC_engine(d1=d1, d2=d2, d=d, M=M, dt=dt, T=T, x0=s0, task='pricing', r=[0.1, 0.2], sig=[0.3, 0.5], K=[40, 50])
    LSMC_engine_european.run()


def main():
    test_engine()


if __name__ == '__main__':
    main()

