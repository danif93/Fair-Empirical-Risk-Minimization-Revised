# ----- Third Party Imports
from sklearn.metrics.pairwise import (linear_kernel, rbf_kernel,
                                      polynomial_kernel, sigmoid_kernel)


# ---------------------
# --- Custom Exceptions
# ---------------------

class OptimizationError(Exception):
    """Raised when the Gurobi optimization process has not found an optimal solution."""
    
class EmptyVectorError(Exception):
    """Raised when a vector does not contain any samples. E.g. when the training batch does not contain a specific class or sensitive group."""

    
# ---------------
# --- SVM Kernels
# ---------------

class _BaseKernel:
    def __init__(self, **kwargs):
        super().__init__()
    
    def pairwise_distance(self, X, Y=None):
        pass
    
    def __call__(self, X, Y=None):
        return self.pairwise_distance(X, Y)
    
    
class LinearKernel(_BaseKernel):
    def pairwise_distance(self, X, Y=None):
        return linear_kernel(X, Y)
    
    
class SigmoidalKernel(_BaseKernel):
    def pairwise_distance(self, X, Y=None):
        return sigmoid_kernel(X, Y)
    

class GaussianKernel(_BaseKernel):
    def __init__(self, gamma='scale', **kwargs):
        self.gamma = gamma
    
    def pairwise_distance(self, X, Y=None):
        if self.gamma == 'scale':
            gamma = 1. / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            gamma = 1. / X.shape[1]
        else:
            assert isinstance(self.gamma, (float, int))
            gamma = self.gamma
        
        return rbf_kernel(X, Y, gamma=gamma)

    
class PolynomialKernel(_BaseKernel):
    def __init__(self, degree=3, **kwargs):
        self.degree = degree
    
    def pairwise_distance(self, X, Y=None):
        return polynomial_kernel(X, Y, degree=self.degree)


KERNEL_MAP = {
    'linear': LinearKernel,
    'sigmoidal': SigmoidalKernel,
    'gaussian': GaussianKernel,
#    'polynomial': PolynomialKernel,
#    'laplacian': LaplacianKernel,
#    'chi2': Chi2Kernel,
}