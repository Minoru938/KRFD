# coding: utf-8
# author: Minoru Kusaba (kusabam6812@gmail.com)
# last update: 2024/09/22

"""
This module provides a functional output regression model based on the kernel method called KRFD. KRFD constructs a model directly from vector input values and discrete functional output values. If the functional output values are observed at the same measurement point set for each input, use the KRFD_model class. If the functional output values are observed at different measurement point sets for each input, use the KRSFD_model class.
"""

# Import libraries.
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# Define functions.
def Gram_matrix(X, sigma = 1, kernel_type = "Gaussian"):
    """
    Calculate a Gram matrix with a Gaussian or Laplacian kernel.

    Args
    ----
    X: array-like of shape (n_systems, n_features)
          Features for each input system.
    sigma: float, default = 1
          Scale parameter for a kernel function (1/(2*sigma**2) for Gaussian, and 1/sigma for Laplacian).
    kernel_type: str, default = "Gaussian"
          Type of a kernel function for creating a gram matrix.
          Only "Gaussian" and "Laplacian" are supported.
    Returns
    ----
    G: numpy array of shape (n_systems, n_systems)
    gamma: float
    """  
    if kernel_type == "Gaussian":
    # Gaussian kernel.
        gamma = 1/(2*sigma**2)
        if X.ndim == 1:
            d = np.array([(X[i] - X)**2 for i in range(len(X))]) # square of L2 distances
            G = np.exp(-d * gamma)
        else:
            d = distance_matrix(X, X)**2 # square of L2 distances
            G = np.exp(-d * gamma)
            
    elif kernel_type == "Laplacian":
    # Laplacian kernel.
        gamma = 1/sigma
        if X.ndim == 1:
            d = np.array([np.abs(X[i] - X) for i in range(len(X))]) # L1 distances
            G = np.exp(-d * gamma)
        else:
            d = distance_matrix(X, X, p=1) # L1 distances
            G = np.exp(-d * gamma)
    
    else:
        print("kernel_type must be 'Gaussian' or 'Laplacian'.")
            
    return G, gamma

def kernel_for_test(X, X_test, gamma, kernel_type = "Gaussian"):
    """
    Calculate a Gram matrix for test set with a Gaussian or Laplacian kernel.

    Args
    ----
    X: array-like of shape (n_systems, n_features)
          Features for training systems.
    X_test: array-like of shape (n_test_systems, n_features)
          Features for test systems.    
    gamma: float
          Gamma parameter for a kernel function (gamma = 1/(2*sigma**2) for Gaussian, and gamma = 1/sigma for Laplacian).
    kernel_type: str, default = "Gaussian"
          Type of a kernel function for creating a Gram matrix.
          Only "Gaussian" and "Laplacian" are supported.
    Returns
    ----
    np.exp(-d_X * gamma): numpy array of shape (n_test_systems, n_systems)
    """  
    if kernel_type == "Gaussian":
        # more than 1D?
        if X.ndim == 1:
            d_X = np.array([(X - X_test[i])**2 for i in range(len(X_test))])
        else:
            d_X = distance_matrix(X_test, X)**2
    elif kernel_type == "Laplacian":
        # more than 1D?
        if X.ndim == 1:
            d_X = np.array([np.abs(X - X_test[i]) for i in range(len(X_test))])
        else:
            d_X = distance_matrix(X_test, X, p=1)
    else:
        print("kernel_type must be 'Gaussian' or 'Laplacian'.")
        
    return np.exp(-d_X * gamma) 

def KRFD_const_solve(Y, G, T, M, lambda_G = 1/16, lambda_T = 1/16, lambda_M = 1/16):
    """
    Solve the KRFD model.

    Args
    ----
    Y: array-like of shape (n_systems, n_measurement_points)
          Functional output values for each system and measurement point.
    G: array-like of shape (n_systems, n_systems)
          Gram matrix for X (X is features for each input system).
    T: array-like of shape (n_measurement_points, n_measurement_points)
          Gram matrix for measurement points.
    M: array-like of shape (n_systems, n_systems)
           Gram matrix for X. This Gram matrix is used for the X-dependent constant term.
    lambda_G: float, default = 1/16
          Regularization parameter for controlling the complexity of the model in X-space.
    lambda_T: float, default = 1/16
          Regularization parameter for controlling the complexity of the model in t-space.
    lambda_M: float, default = 1/16
          Regularization parameter for controlling the complexity of the X-dependent constant term.
    Returns
    ----
    c_opt: numpy array of shape (n_systems, 1)
    Theta_opt: numpy array of shape (n_systems, n_measurement_points)
    """  
    # Set A, B, and C.
    num_N = G.shape[0]
    num_T = T.shape[0]
    
    I_N = np.eye(num_N)
    I_T = np.eye(num_T)
    
    Inv_G = np.linalg.inv(G + lambda_G*I_N)
    Inv_T = np.linalg.inv(T + lambda_T*I_T)
    Inv_M = np.linalg.inv(M + lambda_M*I_N)
    
    A = G @ Inv_G
    B = Inv_T @ T
    C = Inv_M/num_T
    
    # Solve model.
    one_T = np.array([np.ones(num_T)]).T
    
    sum_B = (one_T.T @ B @ one_T)[0][0]
    F = np.linalg.inv(I_N - sum_B * C @ A @ M)
    L = C @ (Y - A @ Y @ B) @ one_T
    # c_opt.
    c_opt = F @ L
    # Theta_opt.
    Center = Y - M @ c_opt @ one_T.T
    Theta_opt = Inv_G @ Center @ Inv_T
    
    return c_opt, Theta_opt

class KRFD_model():
    """
    A class for KRFD (Kernel Regression for functional data) model.
    """
    def __init__(self, sigmas = [1, 1, 1], lambdas = [1/16, 1/16, 1/16], IG = [1, 1],
                 kernel_types = ["Gaussian", "Gaussian"]):
        """
        Parameters
        ----
        sigmas: a list of float, default = [1, 1, 1]
              The scale parameters for the gram matrices G, T, and M.
        lambdas: a list of float, default = [1/16, 1/16, 1/16]
              The Regularization parameters corresponding to lambda_G, lambda_T, and lambda_M.
        IG: a list of float, default = [1, 1]
              The shape and scale parameters for inverse gamma distribution, which is a prior distribution for observation noise.
        kernel_types: a list of str, default = ["Gaussian", "Gaussian"]
              The first argument defines the type of kernel functions defined on X-space.
              The second argument defines the type of kernel functions defined on t-space.
        ----
        """
        # Set hyper-parameters.
        self.sigma_G = sigmas[0]
        self.sigma_T = sigmas[1]
        self.sigma_M = sigmas[2]
        
        self.lambda_G = lambdas[0]
        self.lambda_T = lambdas[1]
        self.lambda_M = lambdas[2]
        
        self.alpha = IG[0]
        self.beta = IG[1]
        
        self.kernel_X = kernel_types[0]
        self.kernel_t = kernel_types[1]
    
    def fit(self, X, t, Y, add_epsilon_X = False):
        """
        Fit KRFD model.
        
        Parameters
        ----
        X: array-like of shape (n_systems, n_features)
              Features for each input system.
        t: array-like of shape (n_measurement_points,) for 1D, and (n_measurement_points, n_dimensions) for more than 1D
              Features for each measurement point.
        Y: array-like of shape (n_systems, n_measurement_points)
              Functional output values for each system and measurement point.
        add_epsilon_X: bool or float, default = False
              If X contains multiple identical sample data, G may become a singular matrix,
              making it impossible to train the KRFD. In that case, G can be made nonsingular 
              by eliminating or averaging duplicate data, or setting a small value (e.g., 0.01) to add_epsilon_X.
        ----
        """
        # Add random noises to X for making G nonsingular.
        if type(add_epsilon_X) == float:
            X = X + np.random.normal(0, add_epsilon_X, X.shape)
        else:
            pass
        
        # Get Gram matrices.
        G, gamma_G = Gram_matrix(X, self.sigma_G, self.kernel_X)
        T, gamma_T = Gram_matrix(t, self.sigma_T, self.kernel_t)
        M, gamma_M = Gram_matrix(X, self.sigma_M, self.kernel_X)
        
        # Get optimal solutions.
        c_opt, Theta_opt = KRFD_const_solve(Y, G, T, M, lambda_G = self.lambda_G,
                                      lambda_T = self.lambda_T, lambda_M = self.lambda_M)
        
        self.c_opt = c_opt
        self.Theta_opt = Theta_opt
        
        self.X = X
        self.t = t
        self.gamma_G = gamma_G
        self.gamma_T = gamma_T
        self.gamma_M = gamma_M
        
        # MAP estimation of sigma2.
        error = Y - ( G @ Theta_opt @ T + M @ c_opt @ np.array([np.ones(len(t))]) )
        
        sigma2_map = (np.sum(error**2) + 2*self.beta)/(2*self.alpha + 2 + error.shape[0]*error.shape[1])
        
        self.sigma2_map = sigma2_map
        
    def predict(self, X_test, t_test, std = False, epsilon = 1e-012):
        """
        Perform prediction with the trained KRFD model for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.
        t_test: array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D
              Features for test measurement points.  
        std: bool, default = False
              If True, the standard deviation of the prediction (Y_pred_std) is also returned.
        epsilon: float, default = 1e-012
              This is used to stabilize the computation of the inverse matrices,
              which is required to derive the predictive distribution's covariance matrix (thus only relevant when std = True).
        Returns
        ----
        Y_pred_mean: numpy array of shape (n_test_systems, n_test_measurement_points)
        Y_pred_std (only if std = True): numpy array of shape (n_test_systems, n_test_measurement_points)
        """  
        # Outputs prediction means and stds.
        if std:
            # Re-calculation of G, T, M (avoiding to save G, T, M in self for data size saving).
            G, gamma_G = Gram_matrix(self.X, self.sigma_G, self.kernel_X)
            T, gamma_T = Gram_matrix(self.t, self.sigma_T, self.kernel_t)
            M, gamma_M = Gram_matrix(self.X, self.sigma_M, self.kernel_X)

            num_N = G.shape[0]
            num_T = T.shape[0]

            I_N = np.eye(num_N)
            I_T = np.eye(num_T)

            Inv_G2 = np.linalg.inv(G@G + self.lambda_G*G + epsilon*I_N)
            Inv_T2 = np.linalg.inv(T@T + self.lambda_T*T + epsilon*I_T)
            Inv_M2 = np.linalg.inv(M@M + self.lambda_M*M + epsilon*I_N)

            # For test data.
            one_T_test = np.array([np.ones(len(t_test))]).T
            
            # Calculating kernels on test data.
            G_test = kernel_for_test(self.X, X_test, gamma_G, self.kernel_X)
            M_test = kernel_for_test(self.X, X_test, gamma_M, self.kernel_X)
            T_test = kernel_for_test(self.t, t_test, gamma_T, self.kernel_t)
            
            Y_pred_mean = G_test @ self.Theta_opt @ T_test.T + M_test @ self.c_opt @ one_T_test.T # prediction.

            pred_stds = []
            for i in range(X_test.shape[0]):
                M_oneT_test = M_test[i,:] * one_T_test

                cov_theta = self.sigma2_map * G_test[i,:] @ Inv_G2 @ G_test[i,:] * T_test @ Inv_T2 @ T_test.T
                cov_c = self.sigma2_map/num_T  * M_oneT_test @ Inv_M2 @ M_oneT_test.T

                pred_std = np.sqrt(np.diag(cov_theta) + np.diag(cov_c))
                pred_stds.append(pred_std)

            Y_pred_std = np.array(pred_stds)

            return Y_pred_mean, Y_pred_std
              
        # outputs prediction means (lower calculation cost).
        else:
            one_T_test = np.array([np.ones(len(t_test))]).T
            
            # Calculating kernels on test data.
            G_test = kernel_for_test(self.X, X_test, self.gamma_G, self.kernel_X)
            M_test = kernel_for_test(self.X, X_test, self.gamma_M, self.kernel_X)
            T_test = kernel_for_test(self.t, t_test, self.gamma_T, self.kernel_t)

            Y_pred_mean = G_test @ self.Theta_opt @ T_test.T + M_test @ self.c_opt @ one_T_test.T # prediction.

            return Y_pred_mean
    
    def predict_sampling(self, X_test, t_test, size = 10, epsilon = 1e-012):
        """
        Perform sampling of functions from the predictive distribution for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.
        t_test: array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D
              Features for test measurement points.  
        size: int, default = 10
              Sample size for each system.
        epsilon: float, default = 1e-012
              This is used to stabilize the computation of the inverse matrices,
              which is required to derive the predictive distribution's covariance matrix.
        Returns
        ----
        pred_samplings: a list of numpy arraies of shape (size, n_test_measurement_points);
                        The length of the list is n_test_systems
        """  
        # Re-calculation of G, T, M (avoiding to save G, T, M in self for data size saving).
        G, gamma_G = Gram_matrix(self.X, self.sigma_G, self.kernel_X)
        T, gamma_T = Gram_matrix(self.t, self.sigma_T, self.kernel_t)
        M, gamma_M = Gram_matrix(self.X, self.sigma_M, self.kernel_X)
        
        num_N = G.shape[0]
        num_T = T.shape[0]

        I_N = np.eye(num_N)
        I_T = np.eye(num_T)
        
        Inv_G2 = np.linalg.inv(G@G + self.lambda_G*G + epsilon*I_N)
        Inv_T2 = np.linalg.inv(T@T + self.lambda_T*T + epsilon*I_T)
        Inv_M2 = np.linalg.inv(M@M + self.lambda_M*M + epsilon*I_N)
        
        # For test data.
        one_T_test = np.array([np.ones(len(t_test))]).T
        
        # Calculating kernels on test data.
        G_test = kernel_for_test(self.X, X_test, gamma_G, self.kernel_X)
        M_test = kernel_for_test(self.X, X_test, gamma_M, self.kernel_X)
        T_test = kernel_for_test(self.t, t_test, gamma_T, self.kernel_t)
        
        Y_pred_mean = G_test @ self.Theta_opt @ T_test.T + M_test @ self.c_opt @ one_T_test.T # prediction.
        
        pred_samplings = []
        for i in range(X_test.shape[0]):
            M_oneT_test = M_test[i,:] * one_T_test
        
            cov_theta = self.sigma2_map * G_test[i,:] @ Inv_G2 @ G_test[i,:] * T_test @ Inv_T2 @ T_test.T
            cov_c = self.sigma2_map/num_T  * M_oneT_test @ Inv_M2 @ M_oneT_test.T
            pred_cov = cov_theta + cov_c
            pred_mean = Y_pred_mean[i,:]
            
            pred_sampling = np.random.multivariate_normal(pred_mean, pred_cov, size) # random sampling.
            pred_samplings.append(pred_sampling)
        
        return pred_samplings
    
### Kernel Regression for Sparse Functional Data (KRSFD).

# import additional libraries for KRSFD.
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spilu # for calculating a covariance matrix of the posterior distribution of theta.

def truncated_gram_matrix(X, sigma = 1, zero_ratio = 0.2, kernel_type = "Gaussian"):
    """
    Calculate a Gram matrix with a truncated Gaussian or Laplacian kernel.

    Args
    ----
    X: array-like of shape (n_systems, n_features)
          Features for each input system.
    sigma: float, default = 1
          Scale parameter for a kernel function (1/(2*sigma**2) for Gaussian, and 1/sigma for Laplacian).
    zero_ratio: float, default = 0.2
          The proportion of zeros in the Gram matrix created from this function.
    kernel_type: str, default = "Gaussian"
          Type of a kernel function for creating a gram matrix.
          Only "Gaussian" and "Laplacian" are supported.
    Returns
    ----
    G: numpy array of shape (n_systems, n_systems)
    gamma: float
    thd: float
    """  
    
    if kernel_type == "Gaussian":
        gamma = 1/(2*sigma**2)
        if X.ndim == 1:
            d = np.array([(X[i] - X)**2 for i in range(len(X))])
            G = np.exp(-d * gamma)
            # Get the threshold value for a given zero_ratio.
            zero_position = int(G.shape[0]**2 * zero_ratio - 1)
            thd = np.sort(G.flatten())[zero_position]
            G[G<=thd] = 0 # make truncation.
        else:
            d = distance_matrix(X, X)**2
            G = np.exp(-d * gamma)
            # Get the threshold value for a given zero_ratio.
            zero_position = int(G.shape[0]**2 * zero_ratio - 1)
            thd = np.sort(G.flatten())[zero_position]        
            G[G<=thd] = 0 # make truncation.
            
    elif kernel_type == "Laplacian":
        gamma = 1/sigma
        if X.ndim == 1:
            d = np.array([np.abs(X[i] - X) for i in range(len(X))])
            G = np.exp(-d * gamma)
            # Get the threshold value for a given zero_ratio.
            zero_position = int(G.shape[0]**2 * zero_ratio - 1)
            thd = np.sort(G.flatten())[zero_position]
            G[G<=thd] = 0 # make truncation.
        else:
            d = distance_matrix(X, X, p=1)
            G = np.exp(-d * gamma)
            # Get the threshold value for a given zero_ratio.
            zero_position = int(G.shape[0]**2 * zero_ratio - 1)
            thd = np.sort(G.flatten())[zero_position]        
            G[G<=thd] = 0 # make truncation.
         
    else:
        print("kernel_type must be 'Gaussian' or 'Laplacian'.")
        
    return G, gamma, thd

def truncated_kernel_for_test(X, X_test, gamma, thd, kernel_type = "Gaussian"):
    """
    Calculate a Gram matrix with a truncated Gaussian or Laplacian kernel on test data.

    Args
    ----
    X: array-like of shape (n_systems, n_features)
          Features for training systems.
    X_test: array-like of shape (n_test_systems, n_features)
          Features for test systems.    
    gamma: float
          Gamma parameter for a kernel function (gamma = 1/(2*sigma**2) for Gaussian, and gamma = 1/sigma for Laplacian).
    thd: float
          Threshold for truncating elements of a Gram matrix.
    kernel_type: str, default = "Gaussian"
          Type of a kernel function for creating a gram matrix.
          Only "Gaussian" and "Laplacian" are supported.
    Returns
    ----
    K: numpy array of shape (n_test_systems, n_systems)
    """  
    K = kernel_for_test(X, X_test, gamma, kernel_type)
    K[K<=thd] = 0
    
    return K 

def get_size_of_sparsematrix(sparsematrix):
    """
    Calculate data size of a sparse_matrix.

    Args
    ----
    sparsematrix: a sparse matrix in the Compressed Sparse Row format (scipy.sparse._csr.csr_matrix)
    Returns
    ----
    return: int
    """  
    return sparsematrix.data.nbytes + sparsematrix.indices.nbytes + sparsematrix.indptr.nbytes

def conjugate_gradient_sparse(init, G, T, H, Ys, lambda_H = 1/16, tol = 0.001,
                              max_iter = 500, verbose = True):
    """
    Solve a KRSFD model with conjugate gradient method.

    Args
    ----
    init: array-like of shape (n_systems * n_grid_points,)
          Initial value of the vector x.
    G: array-like of shape (n_systems, n_systems)
          Truncated Gram matrix for X (X is features for each input system).
    T: array-like of shape (n_grid_points, n_grid_points)
          Truncated Gram matrix for grid points.
    H: csr_matrix of shape (n_all_observations, n_systems * n_grid_points)
          H = sparse.vstack([sparse.kron(G[i,:], Ts[i], format="csr") for i in range(n_systems)], format="csr").
    Ys: a list of array-like of shape (n_ith_observations,); The length of the list is n_systems
          The observations for each system.
    lambda_H: float, default = 1/16
          Regularization parameter for controlling the complexity of the KRSFD model.
    tol: float, default = 0.001
          Tolerance for the MSE of the residual vector r.
    max_iter: int, default = 500
          The number of maximum iterations.
    verbose: bool, default = True
          If True, print(f"Optimization finished at {i}th iteration; MSE = {r_sse/N}").
    Returns
    ----
    x.reshape(num_N, num_T): numpy array of shape (n_systems, n_grid_points)
    """  
    # Setting.
    num_N, num_T = G.shape[0], T.shape[0]
    N = num_N*num_T
    b = H.T.dot(np.concatenate(Ys)) # Vector multiplication.

    # Start iteration
    x = init
    vec = H.dot(x) # pre-calculation for HHp
    r = b - (H.T.dot(vec) + lambda_H*x)
    r_sse = np.sum(r**2)
    p = r
    i = 0

    while (r_sse/N) > tol:

        vec = H.dot(p) # pre-calculation for HHp
        Ap = H.T.dot(vec) + lambda_H*p # vector.
        alpha = r_sse/np.sum(Ap * p)
        x += alpha*p
        r += -alpha*Ap
        beta = np.sum(r**2)/r_sse
        p = r + beta*p
        r_sse = np.sum(r**2)
        i += 1

        if i == max_iter:
            break
            
    if verbose:
        print(f"Optimization finished at {i}th iteration; MSE = {r_sse/N}")
 
    return x.reshape(num_N, num_T)

class KRSFD_model():
    """
    A class for KRSFD (Kernel Regression for sparse functional data) model.
    """
    def __init__(self, zero_ratios = [0.9, 0.1], sigmas = [1, 1], lambda_H = 1/16, IG = [1, 1],
                 kernel_types = ["Gaussian", "Gaussian"]):
        """
        Parameters
        ----
        zero_ratios: a list of float, default = [0.9, 0.1]
          The proportions of zeros in the Gram matrices G and T.
        sigmas: a list of float, default = [1, 1]
              The scale parameters for the Gram matrices G and T.
        lambda_H: float, default = 1/16
              Regularization parameter for controlling the complexity of the KRSFD model.
        IG: a list of float, default = [1, 1]
              The shape and scale parameters for inverse gamma distribution, which is a prior distribution for observation noise.
        kernel_types: a list of str, default = ["Gaussian", "Gaussian"]
              The first argument defines the type of kernel functions defined on X-space.
              The second argument defines the type of kernel functions defined on t-space.
        ----
        """
        # Set hyper-parameters.
        self.zero_ratios = zero_ratios
        self.sigmas = sigmas
        self.lambda_H = lambda_H
        
        self.alpha = IG[0]
        self.beta = IG[1]
        
        self.kernel_X = kernel_types[0]
        self.kernel_t = kernel_types[1]
        
    def fit(self, X, ts, Ys, t_grid, init = False, tol = 0.001, max_iter = 500, verbose = True, cov_Theta = False,
            add_epsilon_X = False):
        """
        Fit KRSFD model.
        
        Parameters
        ----
        X: array-like of shape (n_systems, n_features)
              Features for each input system.
        ts: a list of array-like of shape (n_ith_observations,) for 1D,
            and (n_ith_observations, n_dimensions) for more than 1D; The length of the list is n_systems
              Features of the measurement points for each system.
        Ys: a list of array-like of shape (n_ith_observations,); The length of the list is n_systems
              The observations for each system.  
        t_grid: array-like of shape (n_grid_points,) for 1D, and (n_grid_points, n_dimensions) for more than 1D
              Features for each grid point.
        init: array-like of shape (n_systems * n_grid_points,)
              Initial value of the vector x in the conjugate gradient method part.
        tol: float, default = 0.001
              Tolerance for the MSE of the residual vector r in the conjugate gradient method part.
        max_iter: int, default = 500
              The number of maximum iterations in the conjugate gradient method part.
        verbose: bool, default = True
              If True, print(f"Optimization finished at {i}th iteration; MSE = {r_sse/N}") in the conjugate gradient method part.
        cov_Theta: bool, default = False
              If True, a covariance matrix for the posterior distribution of Theta is also calculated.
              This increases the computational cost significantly.
        add_epsilon_X: bool or float, default = False
              If X contains multiple identical sample data, G may become a singular matrix,
              making it impossible to train the KRFD. In that case, G can be made nonsingular 
              by eliminating or averaging duplicate data, or setting a small value (e.g., 0.01) to add_epsilon_X.
        ----
        """
        # Add random noises to X for making G nonsingular.
        if type(add_epsilon_X) == float:
            X = X + np.random.normal(0, add_epsilon_X, X.shape)
        else:
            pass
        
        # Get numbers.
        num_N, num_T = X.shape[0], len(t_grid)

        # Dealing with default parameters.
        if type(init) == bool:
            init = np.ones(num_N*num_T)*0.1
            
        G, gamma_G, thd_G = truncated_gram_matrix(X, self.sigmas[0], self.zero_ratios[0], self.kernel_X)
        T, gamma_T, thd_T = truncated_gram_matrix(t_grid, self.sigmas[1], self.zero_ratios[1], self.kernel_t)
        
        Ts = [truncated_kernel_for_test(t_grid, ts[i], gamma_T, thd_T, self.kernel_t) for i in range(num_N)]
        H = sparse.vstack([sparse.kron(G[i,:], Ts[i], format="csr") for i in range(num_N)], format="csr")
        del Ts
        
        if verbose:
            print(f"thd_G = {thd_G}, thd_T = {thd_T}")
            print(f"Data-size reduction rate of H: {get_size_of_sparsematrix(H)/(H.shape[0] * H.shape[1] * 8)}")
            
        Theta = conjugate_gradient_sparse(init, G, T, H, Ys, self.lambda_H, tol,
                              max_iter, verbose)
        
        # Save results.
        self.Theta = Theta
        self.t_grid = t_grid
        self.X = X
        self.gamma_G = gamma_G
        self.gamma_T = gamma_T
        self.thd_G = thd_G
        self.thd_T = thd_T
        
        # MAP estimation of sigma2.
        SSE = np.sum((np.concatenate(Ys) - H @ Theta.flatten())**2)
        num_M = H.shape[0]
        
        if verbose:
            print(f"Train MSE = {SSE/num_M}") 
            
        sigma2_map = (2*self.beta + SSE)/(2*self.alpha + 2 + num_M)
        self.sigma2_map = sigma2_map
        
        # Calculate a covariance matrix for the posterior discribution of Theta (requires significant computational cost).
        if cov_Theta:
            # Perform the incomplete LU decomposition on a sparse matrix with default setting.
            self.inv_H2 = spilu(H.T.dot(H) + self.lambda_H * sparse.eye(H.shape[1]))
        else:
            self.inv_H2 = None # added (2024/07/21).
                  
    def predict(self, X_test, t_test, std = False):
        """
        Perform prediction with the trained KRSFD model for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.       
        t_test: (option 1): a list of array-like of shape (n_ith_test_observations,) for 1D,
                and (n_ith_test_observations, n_dimensions) for more than 1D; The length of the list is n_test_systems
                (option 2): array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D
              Features for test measurement points.  
        std: bool, default = False
              If True, the standard deviation of the prediction (Y_pred_std) is also returned.
              To calculate std, cov_Theta need to be True when using the KRSFD_model.fit.
        Returns
        ----
        Y_pred: when t_test is a list, numpy array of shape (n_all_test_observations,);
                when t_test is not a list, numpy array of shape (n_test_systems, n_test_measurement_points)
        Y_pred_std (only if std = True and t_test is not a list): numpy array of shape (n_test_systems, n_test_measurement_points)
        """  
        # Sparse t_test.
        if type(t_test) == list:
            
            # Calculating kernels on test data.
            G_test = truncated_kernel_for_test(self.X, X_test, self.gamma_G, self.thd_G, self.kernel_X)
            Ts_test = [truncated_kernel_for_test(self.t_grid, t_test[i], self.gamma_T, self.thd_T,
                                                self.kernel_t) for i in range(len(t_test))]
            
            H_test = sparse.vstack([sparse.kron(G_test[i,:], Ts_test[i], format="csr") for i in range(len(t_test))], format="csr")
            
            Y_pred_mean = H_test @ self.Theta.flatten()
            
            return Y_pred_mean
        # Dense t_test.
        else:
            
            # Calculating kernels on test data.
            G_test = truncated_kernel_for_test(self.X, X_test, self.gamma_G, self.thd_G, self.kernel_X) 
            T_test = truncated_kernel_for_test(self.t_grid, t_test, self.gamma_T, self.thd_T, self.kernel_t)

            Y_pred_mean = G_test @ self.Theta @ T_test.T
            
            # Prediction mean + std (requires significant computational cost).
            if std:
                if type(self.inv_H2) != sparse.linalg.SuperLU:
                    print("To calculate std, cov_Theta need to be True when using the KRSFD_model.fit.")
                else:
                    pass
                
                pred_stds = []
                for i in range(G_test.shape[0]):
                    GT_test = np.kron(G_test[i,:].reshape(G_test.shape[1], 1), T_test.T)
                    pred_cov = self.sigma2_map * GT_test.T @ self.inv_H2.solve(GT_test)
                    pred_stds.append(np.sqrt(np.diag(pred_cov)))
                
                Y_pred_std = np.array(pred_stds)
                
                return Y_pred_mean, Y_pred_std 
            
            # Prediction mean only.
            else:
                return Y_pred_mean
        
    def predict_sampling(self, X_test, t_test, size = 10): 
        """
        Perform sampling of functions from the predictive distribution for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.
        t_test: array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D
              Features for test measurement points.  
        size: int, default = 10
              Sample size for each system.
        Returns
        ----
        pred_samplings: a list of numpy arraies of shape (size, n_test_measurement_points);
                        The length of the list is n_test_systems.
        """  
        if type(self.inv_H2) != sparse.linalg.SuperLU:
            print("To calculate std, cov_Theta need to be True when using the KRSFD_model.fit.")
        else:
            pass
        
        # Calculating kernels on test data.
        G_test = truncated_kernel_for_test(self.X, X_test, self.gamma_G, self.thd_G, self.kernel_X) 
        T_test = truncated_kernel_for_test(self.t_grid, t_test, self.gamma_T, self.thd_T, self.kernel_t)

        Y_pred_mean = G_test @ self.Theta @ T_test.T
        
        # Perform functional sampling.
        pred_samplings = []
        for i in range(G_test.shape[0]):
            GT_test = np.kron(G_test[i,:].reshape(G_test.shape[1], 1), T_test.T)
            pred_cov = self.sigma2_map * GT_test.T @ self.inv_H2.solve(GT_test)
            pred_mean = Y_pred_mean[i,:]
            pred_sampling = np.random.multivariate_normal(pred_mean, pred_cov, size) # random sampling.
            pred_samplings.append(pred_sampling)
        
        return pred_samplings      
        
##################################
## for comparison to KRFD.
##################################

class LFOS():
    """
    A class for Linear Function-On-Scalars (LFOS) model.　
    """
    def __init__(self, sigma = 1, lambda_H = 1, kernel_type = "Gaussian"):
        """
        Parameters
        ----
        sigma: float, default = 1
              Scale parameter for a kernel function (1/(2*sigma**2) for Gaussian, and 1/sigma for Laplacian).
        lambda_H: float, default = 1
              Regularization parameter for controlling the complexity of the LFOS model.
        kernel_type: str, default = "Gaussian"
              Type of kernel functions defined on t-space.
        ----
        """
        # Set hyper-parameters.
        self.sigma = sigma
        self.lambda_H = lambda_H
        self.kernel_type = kernel_type
    def fit(self, X, t, Y):
        """
        Fit LFOS model.
        
        Parameters
        ----
        X: array-like of shape (n_systems, n_features)
              Features for each input system.
        t: array-like of shape (n_measurement_points,) for 1D, and (n_measurement_points, n_dimensions) for more than 1D
              Features for each measurement point.
        Y: array-like of shape (n_systems, n_measurement_points)
              Functional output values for each system and measurement point.
        ----
        """
        # Add constant term.
        X_aug = np.concatenate([np.ones(X.shape[0]).reshape(X.shape[0], 1), X], 1)
        # Gram matrix for t.
        T, gamma_T = Gram_matrix(t, self.sigma, self.kernel_type)
        H = np.kron((X_aug.T @ X_aug), (T @ T)) 
        # Solve LFOS.
        L = (X_aug.T @ Y @ T).flatten()
        F = np.linalg.inv(H + self.lambda_H * np.eye(H.shape[0]))
        Theta = (F @ L).reshape(X_aug.shape[1], len(t))
        # Save results.
        self.Theta = Theta
        self.gamma_T = gamma_T
        self.X_aug = X_aug
        self.t = t
    def predict(self, X_test, t_test):
        """
        Perform prediction with the trained LFOS model for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.
        t_test: array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D.
              Features for test measurement points.  
        Returns
        ----
        Y_pred: numpy array of shape (n_test_systems, n_test_measurement_points)
        """  
        # Add constant term.
        X_test_aug = np.concatenate([np.ones(X_test.shape[0]).reshape(X_test.shape[0], 1), X_test], 1)
        # Get T_test.
        T_test = kernel_for_test(self.t, t_test, self.gamma_T, self.kernel_type)
        # Make prediction.
        Y_pred = X_test_aug @ self.Theta @ T_test.T
        
        return Y_pred

class LFOS_sparse():
    """
    A class for LFOS model on sparse functional data.　
    """
    def __init__(self, sigma = 1, lambda_H = 1, kernel_type = "Gaussian"):
        """
        Parameters
        ----
        sigma: float, default = 1
              Scale parameter for a kernel function (1/(2*sigma**2) for Gaussian, and 1/sigma for Laplacian).
        lambda_H: float, default = 1
              Regularization parameter for controlling the complexity of the LFOS model.
        kernel_type: str, default = "Gaussian"
              Type of kernel functions defined on t-space.
        ----
        """
        # Set hyper-parameters.
        self.sigma = sigma
        self.lambda_H = lambda_H
        self.kernel_type = kernel_type
    def fit(self, X, ts, Ys, t_grid, **kwargs):
        """
        Fit LFOS model on sparse functional data.
        
        Parameters
        ----
        X: array-like of shape (n_systems, n_features)
              Features for each input system.
        ts: a list of array-like of shape (n_ith_observations,) for 1D,
            and (n_ith_observations, n_dimensions) for more than 1D; The length of the list is n_systems
              Features of the measurement points for each system.
        Ys: a list of array-like of shape (n_ith_observations,); The length of the list is n_systems
              The observations for each system.  
        t_grid: array-like of shape (n_grid_points,) for 1D, and (n_grid_points, n_dimensions) for more than 1D
              Features for each grid point.
        ----
        """
        # Add constant term.
        X_aug = np.concatenate([np.ones(X.shape[0]).reshape(X.shape[0], 1), X], 1)
        # Gram matrix for t, then calculate H.
        T, gamma_T = Gram_matrix(t_grid, self.sigma, self.kernel_type)
        Ts = [kernel_for_test(t_grid, ts[i], gamma_T, self.kernel_type) for i in range(len(ts))]
        H = np.concatenate([np.kron(X_aug[i,:], Ts[i]) for i in range(len(ts))])
        # Solve LFOS.
        F = np.linalg.inv(H.T @ H + self.lambda_H * np.eye(H.shape[1]))
        L = H.T @ np.concatenate(Ys)
        Theta = F @ L
        # Save results.
        self.Theta = Theta
        self.gamma_T = gamma_T
        self.t_grid = t_grid
    def predict(self, X_test, t_test):
        """
        Perform prediction with the trained LFOS model for the test set.

        Args
        ----
        X_test: array-like of shape (n_test_systems, n_features)
              Features for test systems.
        t_test: (option 1): a list of array-like of shape (n_ith_test_observations,) for 1D,
                and (n_ith_test_observations, n_dimensions) for more than 1D; The length of the list is n_test_systems
                (option 2): array-like of shape (n_test_measurement_points,) for 1D,
                and (n_test_measurement_points, n_dimensions) for more than 1D
              Features for test measurement points.  
        Returns
        ----
        Y_pred: when t_test is a list, numpy array of shape (n_all_test_observations,);
                when t_test is not a list, numpy array of shape (n_test_systems, n_test_measurement_points)
        """  
        # Add constant term.
        X_test_aug = np.concatenate([np.ones(X_test.shape[0]).reshape(X_test.shape[0], 1), X_test], 1)
        # Sparse t_test.
        if type(t_test) == list:
            # Calculating kernels on test data.
            Ts_test = [kernel_for_test(self.t_grid, t_test[i], self.gamma_T, 
                                       self.kernel_type) for i in range(len(t_test))]
            H_test = np.concatenate([np.kron(X_test_aug[i,:], Ts_test[i]) for i in range(len(t_test))])
            # Prediction.
            Y_pred = H_test @ self.Theta
            return Y_pred
        # Dense t_test.
        else:
            T_test = kernel_for_test(self.t_grid, t_test, self.gamma_T, self.kernel_type)
            # Prediction.
            Y_pred = X_test_aug @ self.Theta.reshape(X_test_aug.shape[1], len(self.t_grid)) @ T_test.T
            return Y_pred