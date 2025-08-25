import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

class LinearRegression():
    
    def __init__(self, n_iters=5000, fit_intercept=True, standardize=True,
                 tol=None, random_state=None):
        """
        n_iters: max number of GD-iterations
        fit_intercept: learning a separate b (intercept)
        standardize: standarizes X (0-mean, 1-std) in training
        tol: early stopping if |ΔJ| < tol
        """
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.tol = tol
        self.random_state = random_state
        # set in fit()
        self.w = None           # (d,)
        self.b = 0.0            # scalar
        self.mu_ = None         # (d,) feature-mean (only if standardize)
        self.sigma_ = None      # (d,) feature-std  (only if standardize)
        self.history_ = []      # liste over J per iter
        
    def _ensure_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _prepare_X(self, X, training=False):
        X = self._ensure_2d(X)
        if self.standardize:
            if training:
                self.mu_ = X.mean(axis=0)
                self.sigma_ = X.std(axis=0, ddof=0)
                self.sigma_[self.sigma_ == 0.0] = 1.0  # avoid division by zero
            X = (X - self.mu_) / self.sigma_
        return X    

    def fit(self, X, y, lr=0.05):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
            lr (float): learning rate
        """
        X = self._prepare_X(X, training=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape

        # init
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)              # zero is safe when standardizing
        self.b = 0.0 if self.fit_intercept else 0.0

        prev_J = np.inf
        for t in range(self.n_iters):
            # forward
            y_pred = X @ self.w + (self.b if self.fit_intercept else 0.0)
            err = y_pred - y

            # loss (med 1/(2n) makes gradients simpler)
            J = (err @ err) / (2.0 * n)
            self.history_.append(J)

            # early stopping
            if self.tol is not None and abs(prev_J - J) < self.tol:
                logger.info(f"tidlig stopp ved iter={t}, J={J:.6f}")
                break
            prev_J = J

            # gradients
            grad_w = (X.T @ err) / n
            if self.fit_intercept:
                grad_b = err.mean()

            # update
            self.w -= lr * grad_w
            if self.fit_intercept:
                self.b -= lr * grad_b

        return self
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = self._prepare_X(X, training=False)
        y_pred = X @ self.w + (self.b if self.fit_intercept else 0.0)

        return y_pred

    def _MSE(self, y_true, y_pred):
        """
        Computes Mean Squared Error (MSE)
        
        Args:
            y_true (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats
            
        Returns:
            A float
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        return (np.square(y_true - y_pred)).mean()



