from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

class mltools:
    def __init__(self, conf):
        self.normalize = conf['normalize']
        self.n_jobs = conf['n_jobs']
        self.fit_intercept = conf['fit_intercept']
        self.max_depth = conf['max_depth']
        self.n_estimators = conf['n_estimators']
        self.min_impurity_decrease = conf['min_impurity_decrease']

        self.train_X = conf['X_train']
        self.train_y = conf['y_train']
        self.test_X = conf['X_test']
        self.test_y = conf['y_test']


    def linreg(self):
        """
        Linear regression according to conf-file. fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize, n_jobs=self.n_jobs)

        return clf


    def lasso(self):
        """
        Linear Model trained with L1 prior as regularizer according to conf-file.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = Lasso(fit_intercept=self.fit_intercept, normalize=self.normalize)

        return clf


    def lassocv(self):
        """
        Linear Model trained with L1 prior as regularizer according to conf-file. Retrain model along regularization
        path.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = LassoCV(fit_intercept=self.fit_intercept, normalize=self.normalize, n_jobs=self.n_jobs)

        return clf


    def elasticnet(self):
        """
        Linear Model trained with L2 prior as regularizer according to conf-file.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = ElasticNet(fit_intercept=self.fit_intercept, normalize=self.normalize)

        return clf


    def elasticnetcv(self):
        """
        Linear Model trained with L2 prior as regularizer according to conf-file. Retrain model along regularization
        path.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = ElasticNetCV(fit_intercept=self.fit_intercept, normalize=self.normalize, n_jobs=self.n_jobs)

        return clf


    def ridge(self):
        """
        Linear Model trained with L1 + L2 prior as regularizer according to conf-file.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = Ridge(fit_intercept=self.fit_intercept, normalize=self.normalize)

        return clf


    def ridgecv(self):
        """
        Linear Model trained with L1 + L2 prior as regularizer according to conf-file. Retrain model along regularization
        path.
        fit_intercept centers data, normalize normalize data
        based on mean and dividing by the l2-term. n_jobs = # of CPU's to use
        :return: Model object
        """
        clf = RidgeCV(fit_intercept=self.fit_intercept, normalize=self.normalize)

        return clf


    def rfr(self):
        """
        Random forest regressor based on conf-file
        :return: Model object
        """
        clf = RandomForestRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators,
                                    min_impurity_decrease=self.min_impurity_decrease)

        return clf


    def etr(self):
        """
        Extra boosted trees regressor based on conf-file
        :return: Model object
        """
        clf = ExtraTreesRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators,
                                  min_impurity_decrease=self.min_impurity_decrease)

        return clf


    def trainmodel(self, clf):
        """
        Train model clf with X and y data.
        :param clf: SkLearn ML object
        :return: Trained model clf_fit
        """
        clf_fit = clf.fit(self.train_X, self.train_y)

        return clf_fit


    def predict(self, clf_fit):
        """
        Make predictions based on trained model clf_fit
        :param clf_fit: Pre-trained model
        :return predictions
        """
        predictions = clf_fit.predict(self.test_X)

        return predictions


    def getR2score(self, clf_fit):
        """
        Get the scoring (residual sum of squares) for the SkLearn ML model
        :return: Returns the coefficient of determination R^2 of the prediction.
        """
        score = clf_fit.score(self.test_X, self.test_y)
        print('R2-score: {}'.format(round(score, 2)))

        return score