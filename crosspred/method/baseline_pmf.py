import numpy as np
from utils.metric import rmse


class Baseline:
    def __init__(self, train_data):
        self.predict(train_data.copy())

    def predict(self, train_data):
        raise NotImplementedError

    def rmse(self, test_data):
        return rmse(test_data, self.predicted)


class UniformRandomBaseline(Baseline):
    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        masked_train = np.ma.masked_array(data, nan_mask)
        pmin, pmax = masked_train.min(), masked_train.max()
        data[nan_mask] = np.random.uniform(pmin, pmax, nan_mask.sum())
        self.predicted = data


class GlobalMeanBaseline(Baseline):
    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        data[nan_mask] = data[~nan_mask].mean()
        self.predicted = data


class MeanOfMeansBaseline(Baseline):
    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        masked_train = np.ma.masked_array(data, nan_mask)
        global_mean = masked_train.mean()
        user_means = masked_train.mean(axis=1)
        item_means = masked_train.mean(axis=0)
        self.predicted = data.copy()
        n, m = data.shape
        for i in range(n):
            for j in range(m):
                if np.ma.isMA(item_means[j]):
                    self.predicted[i, j] = np.mean((global_mean, user_means[i]))
                else:
                    self.predicted[i, j] = np.mean((global_mean, user_means[i], item_means[j]))
