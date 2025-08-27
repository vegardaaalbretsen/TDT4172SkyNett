import numpy as np
import logging

logger = logging.getLogger(__name__)

class LogisticRegression:
    def __init__(self, n_iters, lr, fit_intercept=True, standardize=True, tol=None,
                 threshold=0.5, random_state=None, lambda_l2=0.0, grad_tol=None):
        self.n_iters = n_iters
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.tol = tol                  # tidlig stopp på Δloss
        self.grad_tol = grad_tol        # tidlig stopp på ||grad||
        self.threshold = threshold
        self.random_state = random_state
        self.lambda_l2 = float(lambda_l2)

        self.w = None           # (d,)
        self.b = 0.0            # scalar
        self.mu = None
        self.sigma = None       # (d,)
        self.history = []       # list of loss per iter (inkl. L2)

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
            if self.mu is None or self.sigma is None:
                raise ValueError("mu or sigma not initialized")
            X = (X - self.mu) / self.sigma

        if y is None:
            return X
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have different number of samples")
        unique = np.unique(y)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError("y must be binary (0 or 1)")
        return X, y

    def fit(self, X, y):
        X, y = self._prepare_data(X, y, training=True)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for it in range(self.n_iters):
            z = X @ self.w + (self.b if self.fit_intercept else 0.0)
            p = self._sigmoid(z)

            J = self._binary_cross_entropy(z, y)
            J_reg = J + 0.5 * self.lambda_l2 * np.dot(self.w, self.w)
            self.history.append(J_reg)

            if self.tol is not None and len(self.history) > 1:
                if abs(self.history[-1] - self.history[-2]) < self.tol:
                    logger.info(f"Early stopping (ΔJ<{self.tol}) at iter {it}")
                    break

            dz = (p - y)                          # dL/dz
            grad_w = (X.T @ dz) / n + self.lambda_l2 * self.w
            if self.fit_intercept:
                grad_b = dz.mean()
                self.b -= self.lr * grad_b
            else:
                grad_b = 0.0

            self.w -= self.lr * grad_w

            gnorm = np.linalg.norm(np.r_[grad_w, grad_b])
            if self.grad_tol is not None and gnorm < self.grad_tol:
                logger.info(f"Early stopping (||g||<{self.grad_tol}) at iter {it}")
                break

            if it % 1000 == 0:
                logger.debug(f"iter={it} J={J_reg:.6f} ||g||={gnorm:.4e}")

        return self

    def predict(self, X, return_proba=False, threshold=None):
        X = self._prepare_data(X, training=False)
        z = X @ self.w + (self.b if self.fit_intercept else 0.0)
        p = self._sigmoid(z)
        if return_proba:
            return p
        tau = self.threshold if threshold is None else threshold
        return (p >= tau).astype(int)

    def score(self, X, y, threshold=None):
        p = self.predict(X, return_proba=True)
        tau = self.threshold if threshold is None else threshold
        yhat = (p >= tau).astype(int)
        y = np.asarray(y, dtype=int).reshape(-1)
        return (yhat == y).mean()

    def log_loss(self, X, y):
        X, y = self._prepare_data(X, y, training=False)
        z = X @ self.w + (self.b if self.fit_intercept else 0.0)
        return self._binary_cross_entropy(z, y)   # uten L2 for rapportering

    def metrics_at(self, X, y, threshold=None):
        p = self.predict(X, return_proba=True)
        tau = self.threshold if threshold is None else threshold
        yhat = (p >= tau).astype(int)
        y = np.asarray(y, dtype=int).reshape(-1)

        tp = int(((yhat == 1) & (y == 1)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        acc  = (tp + tn) / max(len(y), 1)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def roc_auc(self, X, y):
        p = self.predict(X, return_proba=True)
        y = np.asarray(y, dtype=int).reshape(-1)
        n = len(y)
        if n == 0:
            return 0.0
        P = int(y.sum()); N = n - P
        if P == 0 or N == 0:
            return 0.5

        ranks = self._rankdata_avg(p)  # 1..n
        sum_pos = ranks[y == 1].sum()
        auc = (sum_pos - P*(P+1)/2) / (P*N)
        return float(auc)

    @staticmethod
    def _rankdata_avg(x):
        x = np.asarray(x, float)
        order = np.argsort(x)
        ranks = np.zeros_like(x, dtype=float)
        i = 0; n = len(x)
        while i < n:
            j = i
            while j+1 < n and x[order[j+1]] == x[order[i]]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j+1]] = avg_rank
            i = j + 1
        return ranks

    def _binary_cross_entropy(self, z, y):
        z = np.asarray(z, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        return np.mean(np.logaddexp(0.0, z) - y * z)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def poly2(X):
        """Utvid [x0,x1] -> [x0,x1,x0*x1,x0^2,x1^2]."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 2)
        x0 = X[:, 0]; x1 = X[:, 1]
        return np.c_[x0, x1, x0*x1, x0**2, x1**2]

    def find_best_threshold(self, X_val, y_val, grid=None, metric="accuracy"):
        if grid is None:
            grid = np.linspace(0.05, 0.95, 181)
        p = self.predict(X_val, return_proba=True)
        y = np.asarray(y_val, dtype=int).reshape(-1)
        if metric != "accuracy":
            raise ValueError("Only 'accuracy' supported.")
        scores = [((p >= t).astype(int) == y).mean() for t in grid]
        i = int(np.argmax(scores))
        return float(grid[i]), float(scores[i])