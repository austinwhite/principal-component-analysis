import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        covarience = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(covarience)

        eigenvectors = eigenvectors.T
        i_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[i_sorted]
        eigenvectors = eigenvectors[i_sorted]

        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
