# utils/models.py
import numpy as np
from scipy.stats import norm, t as student_t
from scipy.special import digamma, polygamma


class GaussianModel:
    def __init__(self, mu=None, sigma=None, fixed_sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.fixed_sigma = fixed_sigma

    def logpdf(self, y):
        y = np.asarray(y, dtype=float)
        sigma = self.sigma if self.fixed_sigma is None else self.fixed_sigma
        sigma = max(float(sigma), 1e-8)
        return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((y - self.mu)**2)/(sigma**2)

    def m_update(self, y, w):
        y = np.asarray(y, float); w = np.asarray(w, float)
        wsum = np.sum(w) + 1e-12
        self.mu = float(np.sum(w*y)/wsum)
        if self.fixed_sigma is None:
            var = np.sum(w*(y - self.mu)**2)/wsum
            self.sigma = float(np.sqrt(max(var, 1e-12)))

    def interval(self, q_low=0.2, q_high=0.8):
        sigma = self.sigma if self.fixed_sigma is None else self.fixed_sigma
        z_low = norm.ppf(q_low); z_high = norm.ppf(q_high)
        return self.mu + z_low*sigma, self.mu + z_high*sigma


class StudentTModel:
    def __init__(self, mu=None, sigma=None, nu=5.0, nu_bounds=(2.1, 100.0)):
        self.mu = mu
        self.sigma = sigma
        self.nu = float(nu)
        self.nu_bounds = nu_bounds

    def logpdf(self, y):
        y = np.asarray(y, dtype=float)
        nu = self.nu
        sigma = max(float(self.sigma), 1e-8)
        return student_t.logpdf((y - self.mu)/sigma, df=nu) - np.log(sigma)

    def m_update(self, y, w, max_inner=3):
        y = np.asarray(y, float); w = np.asarray(w, float)
        if self.mu is None:
            self.mu = float(np.sum(w*y)/(np.sum(w)+1e-12))
        if self.sigma is None:
            var = np.sum(w*(y - self.mu)**2)/(np.sum(w)+1e-12)
            self.sigma = float(np.sqrt(max(var, 1e-12)))
        for _ in range(max_inner):
            d = ((y - self.mu)**2) / (self.sigma**2 + 1e-12)
            Ew = (self.nu + 1.0) / (self.nu + d + 1e-12)
            w_eff = w * Ew
            wsum = np.sum(w_eff) + 1e-12
            self.mu = float(np.sum(w_eff * y)/wsum)
            d = ((y - self.mu)**2) / (self.sigma**2 + 1e-12)
            Ew = (self.nu + 1.0) / (self.nu + d + 1e-12)
            w_eff = w * Ew
            wsum = np.sum(w_eff) + 1e-12
            self.sigma = float(np.sqrt(np.sum(w_eff*(y - self.mu)**2)/wsum + 1e-12))
            Elogw = digamma((self.nu + 1.0)/2.0) - np.log((self.nu + d + 1e-12)/2.0)
            c = (np.sum(w*Ew) - np.sum(w*Elogw)) / (np.sum(w) + 1e-12)

            def F(nu):
                return np.log(nu/2.0) - digamma(nu/2.0) + 1.0 - c

            def Fp(nu):
                return 1.0/nu - 0.5*polygamma(1, nu/2.0)

            for _it in range(2):
                nu_old = self.nu
                val = F(nu_old)
                der = Fp(nu_old)
                step = val / (der + 1e-12)
                self.nu = float(np.clip(nu_old - step, self.nu_bounds[0], self.nu_bounds[1]))

    def interval(self, q_low=0.05, q_high=0.95):
        ql = student_t.ppf(q_low, df=self.nu)
        qh = student_t.ppf(q_high, df=self.nu)
        return self.mu + ql*self.sigma, self.mu + qh*self.sigma
