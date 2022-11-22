import numpy as np


class Engine(object):
    def __init__(self, FBSDEs, configs_solver, solver, **kwargs):
        # Dynamic and Config
        self.FBSDEs = FBSDEs
        self.configs_solver = configs_solver

        # M
        self.solver = solver
        self.kwargs = kwargs

        output_shape = (self.FBSDEs[0].d2, len(self.FBSDEs), len(self.configs_solver))  # d2 x n_cf_FBSDE x n_cf_sim
        self.res = np.zeros(shape=output_shape)

    def run(self):
        for (i, FBSDE) in enumerate(self.FBSDEs):
            for (j, cf_solver) in enumerate(self.configs_solver):
                solver = self.solver(FBSDE, cf_solver, **self.kwargs)
                solver.solve()
                self.res[:, i, j] = solver.y0
