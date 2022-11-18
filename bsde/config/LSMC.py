class config_LSMC(object):
    def __init__(self, model_params, basis_funcs=None):
        self.model_params = model_params
        self.basis_funcs = basis_funcs


class config_linear(config_LSMC):
    def __init__(self, model_params, reg_method=None, basis_funcs=None):
        super(config_linear, self).__init__(model_params, basis_funcs)
        self.reg_method = reg_method


class config_svm(config_LSMC):
    def __init__(self, model_params, basis_funcs=None):
        super(config_svm, self).__init__(model_params, basis_funcs)


class config_NN(config_LSMC):
    def __init__(self, model_params, basis_funcs=None):
        super(config_NN, self).__init__(model_params, basis_funcs)
