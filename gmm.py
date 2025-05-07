import torch
from torch.distributions import MultivariateNormal

class GaussianMixture:
    def __init__(self, n_components, n_features, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.weights = torch.ones(n_components) / n_components
        self.means = torch.randn(n_components, n_features)
        self.covariances = torch.stack([torch.eye(n_features) for _ in range(n_components)])

    def _e_step(self, X):
        N = X.shape[0]
        responsibilities = torch.zeros(N, self.n_components)
        for k in range(self.n_components):
            mvn = MultivariateNormal(self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * mvn.log_prob(X).exp()
        responsibilities /= responsibilities.sum(dim=1, keepdim=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        N_k = responsibilities.sum(dim=0)
        self.weights = N_k / X.shape[0]
        self.means = (responsibilities.T @ X) / N_k.unsqueeze(1)
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k].unsqueeze(1) * diff).T @ diff / N_k[k]
            self.covariances[k] += torch.eye(self.n_features) * 1e-6  # Regularization

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        prev_log_likelihood = None
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood = torch.sum(torch.log(responsibilities.sum(dim=1)))
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        responsibilities = self._e_step(X)
        return responsibilities

    def predict(self, X):
        responsibilities = self.predict_proba(X)
        return responsibilities.argmax(dim=1)

# Example usage:
# gmm = GaussianMixture(n_components=3, n_features=2)
# gmm.fit(data)
# labels = gmm.predict(data)