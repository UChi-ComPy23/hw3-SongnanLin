"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

from scipy.integrate import OdeSolver

class ForwardEuler(OdeSolver):
    """
    Forward Euler Method using y(t+h) = y(t) + h * f(t, y)
    """
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, h=None, **kwargs):
        y0 = np.atleast_1d(np.array(y0, dtype=float))
        if h is None:
            h = (t_bound - t0) / 100 
        if type(y0) in [int, float]:
            y0 = np.array([y0])
        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound, vectorized)
        self.h = h
        self.direction = 1
        self.f = self.fun(self.t, self.y)
        self.njev = 0
        self.nlu = 0

    def _step_impl(self):
        """
        One step update for Euler Method
        """
        t, y, f = self.t, self.y, self.f
        h = self.h
        self.t_old = t
        self.y_old = y.copy()
        y_new = (y + h * f)
        t_new = t + h
        f_new = self.fun(t_new, y_new)
        self.t = t_new
        self.y = y_new
        self.f = f_new
        return True, None

    def _dense_output_impl(self):
        """
        Return dense output object
        """
        return ForwardEulerDenseOutput(self.t_old, self.t, self.y_old, self.y)

class ForwardEulerDenseOutput(DenseOutput):
    """
    Dense output for the Forward Euler method 
    """
    def __init__(self, t_old, t_new, y_old, y_new):
        super(ForwardEulerDenseOutput, self).__init__(t_old, t_new)
        self.y_old = np.asarray(y_old)
        self.y_new = np.asarray(y_new)

    def __call__(self, t_eval):
        """
        Finding any point from a given line
        """
        alpha = (t_eval - self.t_old) / (self.t - self.t_old)
        y_eval = self.y_old + alpha * (self.y_new - self.y_old)
        return np.atleast_2d(y_eval)
