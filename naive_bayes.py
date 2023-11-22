import numpy as np
from sklearn.naive_bayes import _BaseNB
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import _check_partial_fit_first_call, check_classification_targets
from sklearn.utils.validation import column_or_1d, check_array


class GaussianNB(_BaseNB):
    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y)
        y = column_or_1d(y, warn=True)
        return self._partial_fit(X, y, np.unique(y), _refit=True, sample_weight=sample_weight)

    def _check_X(self, X):
        return check_array(X)

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        X, y = check_X_y(X, y)
        _, n_features = X.shape
        classes = check_classification_targets(y)
        unique_y = np.unique(y)

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.theta_ = np.zeros((len(classes), n_features))
            self.sigma_ = np.zeros((len(classes), n_features))
            self.class_count_ = np.zeros(len(classes), dtype=np.float64)

        if self.priors is None:
            self.class_prior_ = np.zeros(len(classes), dtype=np.float64)

        for i, y_i in enumerate(classes):
            X_i = X[y == y_i]
            N_i = X_i.shape[0]
            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :], X_i, sample_weight)
            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.var_smoothing

        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        total_ssd = (n_past * var + n_new * new_var +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var