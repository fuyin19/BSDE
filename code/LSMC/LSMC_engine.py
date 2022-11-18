import numpy as np
from code.LSMC import LSMC

class LSMC_engine(object):
    def __init__(self, FBSDEs, configs_sim, **kwargs):
        self.FBSDEs = FBSDEs
        self.configs_sim = configs_sim
        self.kwargs = kwargs

        output_shape = (self.FBSDEs[0].d2, len(self.FBSDEs), len(self.configs_sim))  # d2 x n_cf_FBSDE x n_cf_sim

        self.res = np.zeros(shape=output_shape)

    def run(self):
        for (i, FBSDE) in enumerate(self.FBSDEs):
            for (j, cf_sim) in enumerate(self.configs_sim):
                solver = LSMC.LSMC_linear(FBSDE, cf_sim, **self.kwargs)
                solver.solve()
                self.res[:, i, j] = solver.y0
