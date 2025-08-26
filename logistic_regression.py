import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

class LogisticRegression():
    def __init__(self, n_iters, lr, fit_intercept=True, standardize=True, tol=None,
                 threshold=0.5, random_state=None,):
        self.n_iters=n_iters
        self.lr=lr
        self.fit_intercept=fit_intercept
        self.standardize=standardize
        self.tol=tol
        self.threshold=threshold
        self.random_state=random_state

        self.w = None           # (d,)
        self.b = 0.0            # scalar
        self.mu = None
        self.sigma = None       # (d,)
        self.history = []       # list over J per iter

    def _ensure2D(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        return X

    def _prepare_data(self, X, y=None, training=False):
        X = self._ensure2D(X)
        if self.standardize:
            if training:
                self.mu = X.mean(axis=0)
                self.sigma = X.std(axis=0, ddof=0)
                self.sigma[self.sigma == 0.0] = 1.0
            if not self.mu or not self.sigma:
                raise ValueError("mu or sigma not initialized")
                
            X = (X - self.mu) / self.sigma

        if y is None:
            return X
            
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have different number of samples")
        
        unique = np.unique(y)
        if not np.all(np.isin(unique, (0.0,1.0))):
            raise ValueError("y must be binary (0 or 1)")

        return X, y
    
    def fit(self):

        pass

    def predict(self):

        pass

    def _binary_cross_entropy(self):

        pass    