import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import norm
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver

# Parameter statistics estimated from Chris Smith's calibrated parameter ensemble
# https://zenodo.org/records/13142999/files/calibrated_constrained_parameters.csv
LOG_MEANS_3 = np.array([
    1.77475666,  1.37614666,  2.7670545 ,  4.41765518,  0.21476718,
    0.94670492,  0.04308742,  0.17697343, -0.17338284, -0.8145181 ,
    2.04040769
])
LOG_STDS_3 = np.array([
    0.78287494, 0.25966234, 0.47488502, 0.5293674 , 0.29054064,
    0.42407117, 0.32690524, 0.29560772, 0.58624183, 0.33045455,
    0.13291872
])

def log_means(k):
    """Return ensemble means of log-transformed parameters for k-box model."""
    if k == 3:
        return LOG_MEANS_3
    else:
        raise ValueError("Number of boxes must be 3.")

def log_stds(k):
    """Return ensemble standard deviations of log-transformed parameters for k-box model."""
    if k == 3:
        return LOG_STDS_3
    else:
        raise ValueError("Number of boxes must be 3.")

def standardise(parameters):
    """Standardise parameters using means and standard deviations from Chris Smith's ensemble."""
    if len(parameters) == 11:
        k = 3
    else:
        raise ValueError("Number of parameters must be 11 (3-box model).")
    return (np.log(parameters) - log_means(k)) / log_stds(k)

def unstandardise(parameters):
    """Unstandardise parameters using means and standard deviations from Chris Smith's ensemble."""
    if len(parameters) == 11:
        k = 3
    else:
        raise ValueError("Number of parameters must be 11 (3-box model).")
    return np.exp(parameters * log_stds(k) + log_means(k))

def unpack_parameters(parameters):
    """Unpack parameters from a 1D array."""
    if len(parameters) == 11:
        gamma = parameters[0]
        C = parameters[1:4]
        kappa = parameters[4:7]
        epsilon = parameters[7]
        sigma_eta = parameters[8]
        sigma_xi = parameters[9]
        F_4xCO2 = parameters[10]
    else:
        raise ValueError("Number of parameters must be 11 (3-box model).")
    return gamma, C, kappa, epsilon, sigma_eta, sigma_xi, F_4xCO2

def objective(standardised_parameters, y, regularisation_factor):
    """Objective function for optimisation."""
    parameters = unstandardise(standardised_parameters)
    gamma, C, kappa, epsilon, sigma_eta, sigma_xi, F_4xCO2 = unpack_parameters(parameters)
    model = EnergyBalanceModel(gamma, C, kappa, epsilon, sigma_eta, sigma_xi, F_4xCO2)
    penalty = -np.sum(norm.logpdf(standardised_parameters)) * regularisation_factor
    return model.negative_log_likelihood(y) + penalty

def maximise_likelihood(y, regularisation_factor, n_attempts, **kwargs):
    """Maximise likelihood for observations using the Kalman filter."""
    for attempt in range(n_attempts):
        print(f'Attempt {attempt + 1}:')
        initial_guess = np.random.randn(11)
        try:
            best_value = objective(initial_guess, y, regularisation_factor)
        except ValueError:
            print('  Initial guess failed. Trying again...')
            continue
        print(f'  Initial guess value: {best_value}')
        for i in range(200):
            standardised_parameters = np.random.randn(11)
            try:
                objective_value = objective(standardised_parameters, y, regularisation_factor)
            except ValueError:
                continue
            if objective_value < best_value:
                initial_guess = standardised_parameters
                best_value = objective_value
                print(f'  New best value:      {best_value}')
        print('  Optimising...')
        try:
            result = minimize(objective, initial_guess, args=(y, regularisation_factor), **kwargs)
            if result.success:
                print('  Optimisation successful.')
                break
            else:
                print('  Optimisation failed. Trying again...')
                continue
        except ValueError:
            if attempt < n_attempts - 1:
                print('  Optimisation failed. Trying again...')
                continue
            else:
                print('  Optimisation failed. Maximum number of attempts reached. Returning most recent result.')
                return result
    return result

def fit_ebm(y, regularisation_factor=1, n_attempts=10, **kwargs):
    """Fit the energy balance model to observations using the Kalman filter.
    
    Arguments
    ---------
    y : np.ndarray
        Array of shape (n, 2) containing the observed noisy step response.
        The first column is the surface temperature and the second column is
        the top-of-atmosphere net downward radiative flux.
    regularisation_factor : float
        Positive number determining the amount of regularisation. Zero means no
        regularisation, i.e. maximum likelihood estimation. Values greater than
        zero correspond to maximum a posteriori estimation with a Gaussian prior
        on the standardised parameters.
    n_attempts : int
        Number of attempts to make if optimisation fails.
    **kwargs : dict
        Additional keyword arguments to pass to scipy.optimize.minimize. Good
        results can be obtained by setting method='BFGS' and options={'gtol': 1e-3}.
    
    Returns
    -------
    results : EstimationResults
        Results of fitting the energy balance model to observations.
    """
    result = maximise_likelihood(y, regularisation_factor, n_attempts, **kwargs)
    return EstimationResults(result)

def build_A(gamma, C, kappa, epsilon, k):
        """Build continuous-time dynamics matrix A."""
        if k == 2:
            A = np.array([
                 [-gamma,                                   0,                     0],
                 [1/C[0], -(kappa[0] + epsilon*kappa[1])/C[0], epsilon*kappa[1]/C[0]],
                 [     0,                       kappa[1]/C[1],        -kappa[1]/C[1]]
            ])
        elif k == 3:
            A = np.array([
                 [-gamma,                           0,                                   0,                     0],
                 [1/C[0], -(kappa[0] + kappa[1])/C[0],                       kappa[1]/C[0],                     0],
                 [0,                    kappa[1]/C[1], -(kappa[1] + epsilon*kappa[2])/C[1], epsilon*kappa[2]/C[1]],
                 [0,                                0,                       kappa[2]/C[2],        -kappa[2]/C[2]]
            ])
        else:
            raise ValueError("Number of boxes must be 2 or 3.")
        return A

def build_B(gamma, k):
    """Build continuous-time input matrix B."""
    B = np.zeros((k + 1, 1))
    B[0, 0] = gamma
    return B

def build_Q(C, sigma_eta, sigma_xi, k):
    """Build continuous-time process noise covariance matrix Q."""
    Q = np.zeros((k + 1, k + 1))
    Q[0, 0] = sigma_eta**2
    Q[1, 1] = (sigma_xi/C[0])**2
    return Q

def build_A_d(A):
    """Build discrete-time dynamics matrix A_d."""
    A_d = expm(A)
    return A_d

def build_B_d(A, A_d, B, k):
    """Build discrete-time input matrix B_d."""
    B_d = np.linalg.solve(A, (A_d - np.eye(k + 1)) @ B)
    return B_d

def Build_C_d(kappa, epsilon, k):
    """Build discrete-time observation matrix C_d."""
    if k == 2:
        C_d = np.array([
            [0,                                  1,                       0],
            [1, -kappa[0] + (1 - epsilon)*kappa[1], -(1 - epsilon)*kappa[1]]
        ])
    elif k == 3:
        C_d = np.array([
            [0,         1,                      0,                       0],
            [1, -kappa[0], (1 - epsilon)*kappa[2], -(1 - epsilon)*kappa[2]]
        ])
    else:
        raise ValueError("Number of boxes must be 2 or 3.")
    return C_d

def build_Q_d(A, Q, k):
    """Build discrete-time process noise covariance matrix Q_d.
    
    WARNING: Can return invalid covariance matrices for certain combinations of parameters.
    """
    H = np.block([
        [                      -A,   Q],
        [np.zeros((k + 1, k + 1)), A.T]
    ])
    G = expm(H)
    Q_d = G[k + 1:, k + 1:].T @ G[:k + 1, k + 1:]
    return Q_d

def build_Gamma_0(Ad, Qd, k):
    """Build discrete-time marginal covariance matrix Gamma_0."""
    Gamma_0 = np.linalg.solve(np.eye((k + 1)**2) - np.kron(Ad, Ad), Qd.flatten())
    Gamma_0 = Gamma_0.reshape((k + 1, k + 1))
    return Gamma_0

def confidence_intervals(parameters, standard_errors, alpha=0.05):
    """Compute confidence intervals for parameters using normal quantiles.
    
    WARNING: Can violate physical constraints if applied to unstandardised parameters.
    """
    z = norm.ppf(1 - alpha/2)
    return np.array([parameters - z*standard_errors, parameters + z*standard_errors])

class EnergyBalanceModel:
    """k-box stochastic energy balance model.

    Attributes
    ----------
    gamma : float
        Stochastic forcing continuous-time autocorrelation parameter.
    C : array_like
        Ocean heat capacity of each layer (top first) in W yr m-2 K-1.
    kappa : array_like
        Heat exchange coefficient between ocean layers (top first) in W m-2 K-1.
    epsilon : float
        Efficacy of deepest ocean layer.
    sigma_eta : float
        Standard deviation of stochastic forcing component in W m-2.
    sigma_xi : float
        Standard deviation of stochastic disturbance applied to surface layer in W m-2.
    F_4xCO2 : float
        Effective radiative forcing from a quadrupling of CO2 in W m-2.
    
    Raises
    ------
    ValueError
        If number of boxes is not 2 or 3.
    ValueError
        If lengths of C and kappa are not equal.
    """
    def __init__(self, gamma, C, kappa, epsilon, sigma_eta, sigma_xi, F_4xCO2):
        self.gamma = gamma
        self.C = C
        self.kappa = kappa
        self.epsilon = epsilon
        self.sigma_eta = sigma_eta
        self.sigma_xi = sigma_xi
        self.F_4xCO2 = F_4xCO2

        # Determine number of boxes
        self.k = len(self.C)
        if not (self.k == 2 or self.k == 3):
            raise ValueError("Number of boxes must be 2 or 3.")
        if not len(self.kappa) == self.k:
            raise ValueError("C and kappa must have the same length.")
        
        # Build continuous-time matrices
        self.A = build_A(self.gamma, self.C, self.kappa, self.epsilon, self.k)
        self.B = build_B(self.gamma, self.k)
        self.Q = build_Q(self.C, self.sigma_eta, self.sigma_xi, self.k)

        # Build discrete-time matrices
        self.A_d = build_A_d(self.A)
        self.B_d = build_B_d(self.A, self.A_d, self.B, self.k)
        self.C_d = Build_C_d(self.kappa, self.epsilon, self.k)
        self.Q_d = build_Q_d(self.A, self.Q, self.k)
        self.Gamma_0 = build_Gamma_0(self.A_d, self.Q_d, self.k)
    
    def step_response(self, n):
        """Calculate first n years of step response to 4xCO2 forcing.
        
        Returns
        -------
        x : np.ndarray
            Array of shape (n, k + 1) containing the step response.
        """
        d = self.B_d @ np.array([self.F_4xCO2])
        x = np.zeros((n + 1, self.k + 1))
        x[0, 0] = self.F_4xCO2
        for i in range(n):
            x[i + 1] = self.A_d @ x[i] + d
        return x[1:]

    def simulate_noise(self, n):
        """Simulate n years of process noise.
        
        Returns
        -------
        x : np.ndarray
            Array of shape (n, k + 1) containing the noise.
        """
        chol_Gamma_0 = np.linalg.cholesky(self.Gamma_0)
        chol_Q_d = np.linalg.cholesky(self.Q_d)
        x = np.zeros((n + 1, self.k + 1))
        x[0] = chol_Gamma_0 @ np.random.randn(self.k + 1)
        for i in range(n):
            x[i + 1] = self.A_d @ x[i] + chol_Q_d @ np.random.randn(self.k + 1)
        return x[1:]
    
    def noisy_step_response(self, n):
        """Simulate n years of step response to 4xCO2 forcing with process noise.
        
        Returns
        -------
        x : np.ndarray
            Array of shape (n, k + 1) containing the noisy step response.
        """
        step_response = self.step_response(n)
        noise = self.simulate_noise(n)
        return step_response + noise
    
    def observe_noisy_step_response(self, n):
        """Simulate observed component of n-year noisy step response to 4xCO2 forcing.
        
        Returns
        -------
        y : np.ndarray
            Array of shape (n, 2) containing the observed noisy step response. The
            first column is the surface temperature and the second column is the
            top-of-atmosphere net downward radiative flux.
        """
        x = self.noisy_step_response(n)
        y = self.observe(x)
        return y
    
    def kalman_filter(self, y):
        """Kalman filter for observations of noisy step response.
        
        Arguments
        ---------
        y : np.ndarray
            Array of shape (n, 2) containing the observed noisy step response.
            The first column is the surface temperature and the second column is
            the top-of-atmosphere net downward radiative flux.
        
        Returns
        -------
        saver : filterpy.common.Saver
            Saver object containing the Kalman filter state at each time step.
        """
        n = y.shape[0]
        kf = KalmanFilter(dim_x=self.k + 1, dim_z=2, dim_u=1)
        kf.x = np.zeros(self.k + 1)
        kf.x[0] = self.F_4xCO2
        kf.F = self.A_d
        kf.H = self.C_d
        kf.P = self.Gamma_0
        kf.R = np.eye(2) * 1e-12
        kf.Q = self.Q_d
        kf.B = self.B_d
        u = np.array([self.F_4xCO2])
        saver = Saver(kf)
        for i in range(n):
            kf.predict(u)
            kf.update(y[i])
            saver.save()
        return saver
    
    def negative_log_likelihood(self, y):
        """Compute negative log-likelihood using the Kalman filter.
        
        Arguments
        ---------
        y : np.ndarray
            Array of shape (n, 2) containing the observed noisy step response.
            The first column is the surface temperature and the second column is
            the top-of-atmosphere net downward radiative flux.
        """
        saver = self.kalman_filter(y)
        log_likelihood = np.array(saver['log_likelihood'])
        return -np.sum(log_likelihood)
    
    def print(self):
        """Print physical parameters of the model."""
        print("gamma =", self.gamma)
        print("C =", self.C)
        print("kappa =", self.kappa)
        print("epsilon =", self.epsilon)
        print("sigma_eta =", self.sigma_eta)
        print("sigma_xi =", self.sigma_xi)
        print("F_4xCO2 =", self.F_4xCO2)
    
    def get_parameters(self, format='tuple', standardised=False):
        """Return parameters of the model in specified format.
        
        Arguments
        ---------
        format : str
            Format of the output. Must be 'tuple', 'dict', or 'array'.
        standardised : bool
            If True, return standardised parameters (only for 'array' format).
        
        Raises
        ------
        ValueError
            If format is not 'tuple', 'dict', or 'array'.
        """
        if format == 'tuple':
            return self.gamma, self.C, self.kappa, self.epsilon, self.sigma_eta, self.sigma_xi, self.F_4xCO2
        elif format == 'dict':
            parameter_dict = {
                'gamma': self.gamma,
                'C': self.C,
                'kappa': self.kappa,
                'epsilon': self.epsilon,
                'sigma_eta': self.sigma_eta,
                'sigma_xi': self.sigma_xi,
                'F_4xCO2': self.F_4xCO2
            }
            return parameter_dict
        elif format == 'array':
            parameters = np.array([self.gamma, *self.C, *self.kappa, self.epsilon, self.sigma_eta, self.sigma_xi, self.F_4xCO2])
            if standardised:
                return standardise(parameters)
            else:
                return parameters
        else:
            raise ValueError("Format must be 'tuple', 'dict', or 'array'.")
    
    def observe(self, x):
        """Observe surface temperature and top-of-atmosphere net downward radiative flux.
        
        Arguments
        ---------
        x : np.ndarray
            Array of shape (n, k + 1) containing the state. The first column is the
            forcing and the remaining columns are the temperatures of the ocean layers.

        Returns
        -------
        y : np.ndarray
            Array of shape (n, 2) containing the observations. The first column is the
            surface temperature and the second column is the top-of-atmosphere net
            downward radiative flux.
        """
        y = x @ self.C_d.T
        return y

class EstimationResults:
    """Results of fitting the energy balance model to observations.
    
    Attributes
    ----------
    result : scipy.optimize._optimize.OptimizeResult
        Technical details of the numerical optimisation.
    parameters : np.ndarray
        Array of shape (p,) containing the fitted parameters (standardised).
    covariance : np.ndarray
        Covariance matrix of shape (p, p) of the fitted parameters (standardised).
    standard_errors : np.ndarray
        Array of shape (p,) containing the standard errors of the fitted parameters (standardised).
    log_likelihood : float
        Log-likelihood of the fitted model.
    AIC : float
        Akaike Information Criterion of the fitted model.
    """
    def __init__(self, result):
        self.result = result
        self.parameters = result.x
        self.covariance = result.hess_inv
        self.standard_errors = np.sqrt(np.diag(self.covariance))
        self.log_likelihood = -result.fun
        self.AIC = 2*len(self.parameters) - 2*self.log_likelihood
    
    def get_parameters(self, standardised=False):
        """Return the fitted parameters.
        
        Arguments
        ---------
        standardised : bool
            If True, return standardised parameters.
        
        Returns
        -------
        parameters : np.ndarray
            Array of shape (p,) containing the fitted parameters.
        """
        if standardised:
            return self.parameters
        else:
            return unstandardise(self.parameters)
    
    def get_confidence_intervals(self, alpha=0.05, standardised=False):
        """Return confidence intervals for the fitted parameters.
        
        Arguments
        ---------
        alpha : float
            Significance level for confidence intervals.
        standardised : bool
            If True, return confidence intervals for standardised parameters.
        
        Returns
        -------
        intervals : np.ndarray
            Array of shape (2, p) containing the confidence intervals
        """
        intervals = confidence_intervals(self.parameters, self.standard_errors, alpha)
        if standardised:
            return intervals
        else:
            return unstandardise(intervals)

    def get_model(self):
        """Return an instance of the fitted model."""
        parameters = unstandardise(self.parameters)
        model = EnergyBalanceModel(*unpack_parameters(parameters))
        return model