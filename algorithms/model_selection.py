import time

import numpy as np
import matplotlib.pyplot as plt
import itertools
from algorithms.regression_oracle import RegressionOracle, LassoOracle
from algorithms.acquisition import UCB, Greedy, AcquisitionAlgorithm
from environment.feature_map import FeatureMap
from environment.domain import ContinuousDomain, DiscreteDomain
from typing import List, Optional
import ray
from math import log, exp

class MultWeights():
    def __init__(self, num_models, model_inds, domain, feature_map, T: float, likelihood_std: float =  0.01, random_state = None):
        self._rds = np.random if random_state is None else random_state
        self.algos = []
        self.models = []
        self.model_inds = model_inds
        assert feature_map.size == num_models
        for model_ind in model_inds:
            eta_model = np.zeros(num_models)
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
        return x, x_bp, i

    def loss(self, x1, x2):
        return (x1- x2)**2


    def weight_update(self, losses):
        self.weights = np.multiply(self.weights, (1 - self.etaMW * losses / np.max(losses)))


class LEXP:
    def __init__(self, domain, feature_map: FeatureMap, T: float, likelihood_std: float =  0.01,
                 etaEXP: Optional[float]=None, gammaEXP: Optional[float] = None,
                 banditalg = 'UCB',
                 anytime: bool = False,
                 lambda_coef: Optional[float] = None, betaUCB: Optional[float] = 2,
                 theta_oracle: Optional[np.ndarray] = None,
                 random_state= None):

        self._rds = np.random if random_state is None else random_state
        self.algos = []
        self.models = []
        self.num_models = feature_map.num_groups
        self.feature_map = feature_map
        self.theta_oracle = theta_oracle # for toy experiments where we run this algorithm with oracle knowledge of theta
        self.t = 1
        self.anytime = anytime

        # constants
        if self.anytime:
            if gammaEXP is not None:
                self.gammaEXP = gammaEXP * np.sqrt(np.maximum(np.log(len(self.models)), 1))
            else:
                self.gammaEXP = np.sqrt(np.maximum(np.log(len(self.models)), 1))  #this is just a placeholder ratio for now...
            if etaEXP is not None:
                self.etaEXP = etaEXP* np.sqrt(np.maximum(np.log(len(self.models)), 1))
            else:
                self.etaEXP = np.sqrt(np.maximum(np.log(len(self.models)), 1)) #this is just a placeholder ratio for now...
        else:
            if gammaEXP is not None:
                self.gammaEXP = gammaEXP * np.sqrt(np.maximum(np.log(len(self.models)), 1) / T)
            else:
                self.gammaEXP = np.sqrt(np.maximum(np.log(len(self.models)), 1) / T)  #this is just a placeholder ratio for now...
            if etaEXP is not None:
                self.etaEXP = etaEXP * np.sqrt(np.maximum(np.log(len(self.models)), 1) / T)
            else:
                self.etaEXP = np.sqrt(
                    np.maximum(np.log(len(self.models)), 1) / T)  # this is just a placeholder ratio for now...


        # model_t shows which model is chosen at time t
        self.weights = np.ones(self.num_models)
        self.probs = self.weights/np.sum(self.weights)


        # initialize models
        for j in range(self.num_models):
            eta_model = np.array([int(i == j) for i in range(self.num_models)])
            model = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=likelihood_std, eta=eta_model)
            self.models.append(model)
        if banditalg == 'UCB':
            self.algos = [UCB(model, domain, beta=betaUCB) for model in self.models]
        elif banditalg =='Greedy':
            self.algos = [Greedy(model, domain) for model in self.models]
        else:
            raise NotImplementedError

        # initialize lasso solver
        self.regression_oracle = LassoOracle(feature_map=self.feature_map, lambda_coef=lambda_coef,
                                likelihood_std=likelihood_std, domain=domain)

    def select(self):
        if np.all(self.weights==1): #if there's no history, just randomly pick an action
            x = x_bp = self.algos[0].explore()
            i = 0
        else:
            # print('nonunif weights')
            self.probs = self.weights/np.sum(self.weights)
            explore = self._rds.choice([0, 1], p=[1-self.gammaEXP, self.gammaEXP])
            i = self._rds.choice(range(self.num_models),p=self.probs)
            algo = self.algos[i]
            if explore == 1:
                x = algo.explore()
            else:
                # print('Choosed feaures:',self.feature_map.groups[i])
                x = algo.next()
            x_bp = algo.best_predicted()
        return x, x_bp,i

    def select_virtual(self,true_model):
        algo = self.algos[true_model]
        x = algo.next()
        x_bp = algo.best_predicted()
        return x, x_bp
    # @ray.remote
    # def update_ucb(self, algo, evaluation):
    #     algo.add_data(evaluation['x'], evaluation['y'])
    #     test_x = algo.next()
    #     phi_x = self.feature_map.get_feature(test_x)
    #     if self.theta_oracle is None:
    #         # returns.append(np.dot(phi_x, self.regression_oracle.theta_hat.T))
    #         pred_mean, _ = self.regression_oracle.predict(test_x)
    #         return pred_mean
    #     else:
    #         return np.dot(phi_x, self.theta_oracle.T)

    def update(self,evaluation):
        self.t += 1
        if self.anytime:
            self.gammaEXP = self.gammaEXP*np.sqrt((self.t-1)/self.t)
            self.etaEXP = self.etaEXP*np.sqrt((self.t-1)/self.t)
        # first update your lasso solver/reward estimator
        self.regression_oracle.add_data(evaluation['x'], evaluation['y'])
        if self.regression_oracle._num_train_points > 1 :
            self.regression_oracle.fit_lasso()

        # then go over all the algos. add the new datapoint to them, and get their suggested action.
        returns = []
        times = []
        for algo in self.algos:
            # update algo with new rewards

            algo.add_data(evaluation['x'], evaluation['y'])

            # get algorithms suggestion and pass is through the feature map
            t = time.time()
            test_x = algo.next()
            times.append(time.time() - t)
            phi_x = self.feature_map.get_feature(test_x)
            if self.theta_oracle is None:
                # returns.append(np.dot(phi_x, self.regression_oracle.theta_hat.T))
                pred_mean, _ = self.regression_oracle.predict(test_x)
                returns.append(pred_mean)
            else:
                returns.append(np.dot(phi_x, self.theta_oracle.T))
        print('Total model updating time:{:.1f}'.format(np.sum(times)/60))
        # now update the weights accordingly:
        # print('updating weights')
        self.weight_update(np.array(returns).squeeze())
    #
    # def update_parallel(self, evaluation):
    #     # first update your lasso solver/reward estimator
    #     self.regression_oracle.add_data(evaluation['x'], evaluation['y'])
    #     if self.regression_oracle._num_train_points > 1 :
    #         self.regression_oracle.fit_lasso()
    #
    #     # then go over all the algos. add the new datapoint to them, and get their suggested action.
    #     inputs = [(algo, evaluation) for algo in self.algos]
    #     futures = [self.update_ucb.remote(input[0], input[1]) for input in inputs]
    #     returns = ray.get(futures)
    #     print(returns)
    #     self.weight_update(np.array(returns).squeeze())

    def weight_update(self, returns):
        self.weights = np.multiply(self.weights, np.exp(-1*self.etaEXP * returns)) #since we're minimizing f, we're looking for the model with min return
        # print('updating weights')

    def visualize_probs(self, true_model):
        xlab = np.arange(1, len(self.probs)+1)
        plt.bar(xlab, self.probs, color='#4f7992')
        plt.annotate('*', xy=(xlab[true_model], self.probs[true_model]), xytext=(xlab[true_model], self.probs[true_model]), fontsize=10)
        plt.xticks(xlab)


class Corral:

    def __init__(self, domain, feature_map: FeatureMap, T: float, likelihood_std: float =  0.01,
                 etaCorral=0.1, betaUCB: Optional[float] = 2, gammaCorral = 1, random_state = None):

        self._rds = np.random if random_state is None else random_state
        self.algos = []
        self.models = []
        self.feature_map = feature_map
        # self.hyperparam_list = hyperparam_list
        self.m = feature_map.num_groups  # len(self.hyperparam_list)
        self.probs = np.ones(self.m) / self.m
        self.gamma = gammaCorral / T
        self.beta = exp(1 / log(T))
        self.rho = np.asarray([2 * self.m] * self.m)
        self.etas = np.ones(self.m) * etaCorral
        self.T = T
        self.counter = 0

        # model_t shows which model is chosen at time t
        # self.weights = np.ones(self.m)
        # self.probs = self.weights/np.sum(self.weights)


        # initialize models
        for j in range(self.m):
            eta_model = np.array([int(i == j) for i in range(self.m)])
            model = RegressionOracle(domain=domain, feature_map=feature_map, likelihood_std=likelihood_std, eta=eta_model)
            self.models.append(model)
            self.algos = [UCB(model, domain, beta=betaUCB) for model in self.models]

    def select(self):
        if np.array_equal(self.probs, np.ones(self.m) / self.m): #if there's no history, just randomly pick an action
            x = x_bp = self.algos[0].explore()
            self.chosen_model = 0
        else:
            self.chosen_model = self._rds.choice(range(self.m), p=self.probs)
            algo = self.algos[self.chosen_model]
            x = algo.next()
            x_bp = algo.best_predicted()
        return x, x_bp, self.chosen_model

    def update(self, evaluation):
        loss = evaluation['y'] / self.probs[self.chosen_model]
        if self.counter == 0:
            for algo in self.algos:
                # update algo with new rewards
                algo.add_data(evaluation['x'], evaluation['y']/self.m)
        else:
            self.algos[self.chosen_model].add_data(evaluation['x'], loss)
        self.update_distribution(self.chosen_model, loss)


    def update_distribution(self, arm_idx, loss):
        l = np.zeros(self.m)
        p = self.probs[arm_idx]
        assert (p > 1e-8)
        l[arm_idx] = loss / p  # importance weighted loss vector
        probas_new = self.log_barrier_OMD(self.probs, l, self.etas)
        self.probs = (1 - self.gamma) * probas_new + self.gamma * 1.0 / self.m
        assert (min(self.probs) > 1e-8)

        self.update_etas()
        self.counter += 1


    def update_etas(self):
        '''Updates the eta vector'''
        for i in range(self.m):
            if 1.0 / self.probs[i] > self.rho[i]:
                self.rho[i] = 2.0 / self.probs[i]
                self.etas[i] = self.beta * self.etas[i]

    def log_barrier_OMD(self, p, loss, etas, tol=1e-5):
        '''Implements Algorithm 2 in the paper
        Updates the probabilities using log barrier function'''
        assert (len(p) == len(loss) and len(loss) == len(etas))
        assert (abs(np.sum(p) - 1) < 1e-3)

        xmin = min(loss)
        xmax = max(loss)
        pinv = np.divide(1.0, p)
        thresh = min(np.divide(pinv, etas) + loss)  # the max value such that all denominators are positive
        xmax = min(xmax, thresh)

        def log_barrier(x):
            assert isinstance(x, float)
            inv_val_vec = (np.divide(1.0, p) + etas * (loss - x))
            if (np.min(np.abs(inv_val_vec)) < 1e-5):
                print(thresh, xmin, x, loss)
            assert (np.min(np.abs(inv_val_vec)) > 1e-5)
            val = np.sum(np.divide(1.0, inv_val_vec))
            return val

        x = self.binary_search(log_barrier, xmin, xmax, tol)

        assert (abs(log_barrier(x) - 1) < 1e-2)

        inv_probas_new = np.divide(1.0, self.probs) + etas * (loss - x)
        assert (np.min(inv_probas_new) > 1e-6)
        probas_new = np.divide(1.0, inv_probas_new)
        assert (abs(sum(probas_new) - 1) < 1e-1)
        probas_new = probas_new / np.sum(probas_new)

        return probas_new

    def binary_search(self, func, xmin, xmax, tol=1e-5):
        ''' func: function
        [xmin,xmax] is the interval where func is increasing
        returns x in [xmin, xmax] such that func(x) =~ 1 and xmin otherwise'''

        assert isinstance(xmin, float)
        assert isinstance(func(0.5 * (xmax + xmin)), float)

        l = xmin
        r = xmax
        while abs(r - l) > tol:
            x = 0.5 * (r + l)
            if func(x) > 1.0:
                r = x
            else:
                l = x

        x = 0.5 * (r + l)
        return x