import numpy as np
from algorithms.regression_oracle import RegressionOracle
from algorithms.acquisition import UCB

class MultWeights():
    def __init__(self, num_features, model_inds, domain, feature_map, T, likelihood_std = 0.01):
        self.algos = []
        self.models = []
        self.model_inds = model_inds
        for model_ind in model_inds:
            eta_model = np.zeros(num_features)
            eta_model[model_ind] = 1
            model = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=likelihood_std, eta=eta_model)
            self.models.append(model)
        self.algos = [UCB(model, domain, beta=2.0) for model in self.models]
        self.weights = np.ones(len(self.models))
        self.etaMW = np.sqrt(np.log(len(self.models)) / T)

    def update(self, evaluation):
        losses = []
        for i, algo in enumerate(self.algos):
            # update algo with new rewards
            algo.add_data(evaluation['x'], evaluation['y'])
            # calculate the loss for new algo
            losses.append(self.loss(algo.acquisition(evaluation['x']),evaluation['y']))
        # update weights
        losses = np.array(losses).squeeze()
        self.weight_update(losses)

    def select(self):
        probs = self.weights/np.sum(self.weights)
        i = np.random.choice(range(len(self.models)), 1,p=probs)
        algo = self.algos[i[0]]
        print('Choosed feaures:',self.model_inds[i[0]])
        x = algo.next()
        x_bp = algo.best_predicted()
        return x, x_bp

    def loss(self, x1, x2):
        return (x1- x2)**2


    def weight_update(self, losses):
        self.weights = np.multiply(self.weights, (1 - self.etaMW * losses / np.max(losses)))