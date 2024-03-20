from abc import ABC, abstractmethod

import optuna
import xgboost as xgb
from lightgbm import LGBMRegressor as lgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract base class for all models
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        trains the model on the given data

        Args:
             x_train: training data
             y_train: training label / target data
        """
        pass

    @abstractmethod
    def optimize(self, trail, x_train, y_train, x_test, y_test):
        """
        :param trail: Optuma trail object
        :param x_train: training data
        :param y_train: training target data
        :param x_test: testing data
        :param y_test: testing target data
        """
        pass

class RandomForestModel(Model):
    """
    RF model
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trail, x_train, y_train, x_test, y_test):
        n_estimators = trail.suggest_int("n_estimators",1,200)
        max_depth = trail.suggest_int("max_depth",1,20)
        min_samples_split = trail.suggest_int("min_samples_split",2,20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = lgbm(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    # for linear regression , there are no hyperparamters to tune
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)

class HyperparameterTuner:
    """
    class to perform hyperparameter tuning
    it uses model strategy to perform tuning
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=5):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trail: self.model.optimize(
            trail,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test
        ), n_trials=n_trials)
        return study.best_trail.params