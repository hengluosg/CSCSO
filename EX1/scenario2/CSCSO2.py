import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import kv
import seaborn as sns
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
import ray
plt.rcParams['font.family'] = 'Times New Roman'  
import time
ray.init(num_cpus=40)  
import os

def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



# SGD optimization function
def project(theta, lowbound, upbound):
    """
    Projects theta back into the valid domain [lowbound, upbound] element-wise.
    """
    return np.clip(theta, lowbound, upbound)



def grad_f(theta, x , num_samples):
    x_sqrt = np.sqrt(x)
    x = x.reshape(-1, 1)
    d = theta.shape[0]
    W = np.diag(np.arange(1, d+1))  # Diagonal matrix W
    grad1 = 2 * W @ (theta - x_sqrt)  # Matrix multiplication
    part2 = 2 *lambda1 * x @ x.T @ theta
    
    grad = grad1 + part2
    gradients = np.zeros((num_samples, d))
    for rep in range(num_samples):
        noise = np.random.normal(0, std, d)  # Generate Gaussian noise
        gradients[rep,:] = grad  + noise
    avg_gradient = np.mean(gradients, axis=0)
    return avg_gradient



# SGD with Learning Rate Decay


def sgd_with_lr_decay( theta_0, x, eta_0, gamma, T):
    theta = theta_0
    theta_values = []  
    for t in range(1, T + 1):
        grad = grad_f(theta, x, num_samples)
        eta_t = eta_0 
        theta = theta - eta_t * grad
        theta_values.append(theta)
        
    
    return theta_values




def grid_sample(d, lowbound, upbound, point):
    n = int(round(point ** (1.0 / d))) +1 
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points



def offline_stage(n, T, eta_0, gamma, covariate_dim):
    
    
    covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
    n = len(covariates_points)
    theta_estimates = np.zeros((n,covariate_dim))
    for i in range(n):
        x_i = covariates_points[i]
        
        theta_init = np.random.randn(covariate_dim)  
        theta_values = sgd_with_lr_decay( theta_init, x_i, eta_0, gamma, T)
        theta_bar = np.mean(theta_values, axis=0)
        theta_estimates[i] = theta_bar
        
    return  covariates_points,theta_estimates




def gaussian_kernel(x, x_i, h):
    distance = np.linalg.norm(x - x_i)  
    return np.exp(-(distance ** 2) / (2 * h ** 2))




def true_theta_function(x):
    Ax_sqrt =  np.sqrt(x).T
    x = x.reshape(-1, 1)
    L = np.diag(np.arange(1, covariate_dim + 1))  
    L_plus_W_inv = np.linalg.inv(L + lambda1* x@ x.T)
    theta = L_plus_W_inv @ L  @ Ax_sqrt
    #print(theta.shape)
    return theta.T



def prepare_data_for_plot(n_values, mse_knn, mse_krr, mse_ks,mse_lr):
    data = []
    # for k in k_values:
    #     for i, n in enumerate(n_values):
    #         data.append({"n": n, "MSE": mse_by_k[k][i], "Method": f"k-NN (k={k})"})
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_knn[i]), "Method": "KNN"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_krr[i]), "Method": "KRR"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_ks[i]), "Method": "KS"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_lr[i]), "Method": "LR"})
    return pd.DataFrame(data)

def prepare_plot_data_updata(budget_values, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr,n_knn,n_krr,n_ks,n_lr,  T_knn, T_krr, T_ks, T_lr):
    data = []
    
    for i, n in enumerate(budget_values):
        
        data.append({"T": T_knn[i],"n": n_knn[i],"y": n, "MSE": mse_knn[i], "Variance": variance_knn[i], "Bias^2": bias_knn[i], "Method": f"k-NN"})


    for i, n in enumerate(budget_values):
        data.append({"T": T_krr[i],"n": n_krr[i],"y": n, "MSE": mse_krr[i], "Variance": variance_krr[i], "Bias^2": bias_krr[i], "Method": "KRR"})
    

    for i, n in enumerate(budget_values):
        data.append({"T": T_ks[i],"n": n_ks[i],"y": n, "MSE": mse_ks[i], "Variance": variance_ks[i], "Bias^2": bias_ks[i], "Method": "KS"})
    
    for i, n in enumerate(budget_values):
        data.append({"T": T_lr[i],"n": n_lr[i],"y": n, "MSE": mse_lr[i], "Variance": variance_lr[i], "Bias^2": bias_lr[i], "Method": "LR"})
    

    return pd.DataFrame(data)

def compute_bias_variance_mse(true_value, predicted_values):
    predicted_values = np.array(predicted_values)
    mean_predicted = np.mean(predicted_values, axis=0)  
    bias = mean_predicted - true_value  
    
    bias_squared = np.mean(bias ** 2)  
    variance = np.mean(np.var(predicted_values, axis=0))  
    mse = np.mean(np.mean((predicted_values - true_value) ** 2, axis=1))  
    return bias_squared, variance, mse





def gaussian_kernel_krr(X, Y, length_scale=1.0):
    X = np.array(X)
    Y = np.array(Y)
    dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
    K = np.exp(-0.5 * dists**2)
    
    return K

def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, length_scale=1.0):
    """Compute the inverse of the kernel matrix with regularization."""
    K_phi = gaussian_kernel_krr(covariates_points, covariates_points, length_scale)
    return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1.0):
    """Predict theta for a new point using the trained KRR model."""
    k_phi_x = gaussian_kernel_krr([x], covariates_points, length_scale).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x

def krr_cross_validation(covariates_points, theta_estimates, lambda_vals=[0.0001, 0.001, 0.01, 0.1, 1, 10], n_splits=5, length_scale=1.0):
    """Perform K-fold cross-validation for different lambda_param values."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Define KFold split
    best_lambda = None
    best_mse = float('inf')  # Initialize best MSE as infinity

   


    for lambda_param in lambda_vals:
        mse_scores = []  # List to store mean squared errors for each fold

        for train_index, val_index in kf.split(covariates_points):
            # Split data into training and validation sets
            X_train, X_val = covariates_points[train_index], covariates_points[val_index]
            y_train, y_val = theta_estimates[train_index], theta_estimates[val_index]
            
            # Compute the kernel matrix and its inverse for the training set
            K_phi_inv = compute_inverse_kernel_matrix(X_train, lambda_param=lambda_param, length_scale=length_scale)
            
            # Predict on validation set
            predicted_theta_values = np.array([krr_online_stage(X_train, y_train, x, K_phi_inv, length_scale) for x in X_val])
            
            # Compute MSE for the fold
            mse = mean_squared_error(y_val, predicted_theta_values)
            mse_scores.append(mse)

        # Calculate the average MSE across all folds for the current lambda_param
        average_mse = np.mean(mse_scores)
        #print(f"Lambda: {lambda_param}, Average MSE: {average_mse}")

        # Update the best lambda if we found a lower MSE
        if average_mse < best_mse:
            best_mse = average_mse
            best_lambda = lambda_param

    print(f"Best Lambda: {best_lambda} with MSE: {best_mse}")
    return best_lambda



"""
    Linear regression
"""

def linear_basis(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.hstack([np.ones((x.shape[0], 1)), x - 1])

def quadratic_basis(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.hstack([np.ones((x.shape[0], 1)), x - 1, (x - 1) ** 2])

def cubic_basis(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.hstack([np.ones((x.shape[0], 1)), x - 1, (x - 1) ** 2, (x - 1) ** 3])

    
    
    # return np.hstack([np.ones((x.shape[0], 1)),x-1]) #
    # #return np.hstack([np.ones((x.shape[0], 1)),x-1,(x-1)**2]) #
    # #return np.hstack([np.ones((x.shape[0], 1)),x-1,(x-1)**2,(x-1)**3]) #



def linear_regression_train(x_train, theta_train, basis_fn):
    Phi = basis_fn(x_train)
    Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi)
    beta_hat = Phi_T_Phi_inv @ Phi.T @ theta_train
    return beta_hat


def cross_validate(x_train, theta_train, basis_fn):
    return linear_regression_train(x_train, theta_train, basis_fn)


def linear_regression_on_stage(x, beta_hat, basis_fn):
    phi_x = basis_fn(x)
    return phi_x @ beta_hat


  
def picture_plot1(df):
    sns.set_theme()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = {
    "Linear Basis Function": "#b04040", 
    "Quadratic Basis Function": "#dd8452",  
    "Cubic Basis Function": "#55a868" }

    for i, column in enumerate(df):
       
        sns.lineplot(
                    data=column,
                    ax=axes[i],
                    x='n',
                    y='MSE',
                    hue='Method',
                    style='Method',
                    markers=['^', '^', '^'],
                    dashes=True,
                    palette=palette,         
                    linewidth=3,
                    markersize=10)
        # sns.lineplot(data=column, ax=axes[i], x='n', y='MSE', hue='Method', style='Method', markers=['^', '^', '^'], dashes=True, markersize=10, linewidth=3)
        
        axes[i].set_xscale('log', base=2)
        axes[i].set_xlabel(r'Total budget $(\Gamma)$', fontsize=15)
        axes[i].set_ylabel(r'$log_{2}(MSE)$', fontsize=15)
        axes[i].set_title(r'$d = {} $'.format(covariate_dim1[i]), fontsize=15)

        # Set x-axis tick positions to unique 'n' values and format them as log_2(n)
        xticks = sorted(column['n'].unique())
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in xticks], fontsize=15)

        #axes[i].tick_params(axis='y', labelsize=15)
        axes[i].grid(True)
        
       
    
    plt.tight_layout()
    plt.savefig('ex_2.pdf', format='pdf')  # Save as PDF   
    

@ray.remote
def run_replication_lr(n, T, eta_0, gamma1, covariate_dim, test_points, basis_fn, seed=3):
    set_global_seed(seed)
    covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
    beta_hat = cross_validate(covariates_points, theta_estimates, basis_fn)
    predicted_theta_value = np.array([
        linear_regression_on_stage(x, beta_hat, basis_fn) for x in test_points
    ])
    return predicted_theta_value





if __name__ == '__main__':
    lowbound = 0
    upbound = 2

    eta_0 = 0.1
    gamma1 = 0.01 
    # covariate_dim = 2
    num_samples = 1  # Number of samples for true theta calculation
    lambda1 = 0.1 
    std = 4
    num_test_points = 1  #Online stage
    set_global_seed(3)
    total_budget = 2 ** np.arange(10, 17)
  
    basis_functions_list = {
    "Linear Basis Function": linear_basis,
    "Quadratic Basis Function": quadratic_basis,
    "Cubic Basis Function": cubic_basis}
  
  
    
    data1 = []
    covariate_dim1 = [2,5,8]
    for covariate_dim in covariate_dim1:

      
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
        true_theta_values = true_theta_function(test_points)
        bias_lr,variance_lr, mse_lr = [],[],[]
        n_lr,T_lr = [],[]
        replication = 100 #30 50
        "Linear Regression (LR)" #linear basis function
        df  = pd.DataFrame()
        
           
        for method, basis_fn in basis_functions_list.items():
            mse_list = []
            for gamma in total_budget:
                n = int(gamma ** 0.5)
                T = int(gamma ** 0.5)
                results = ray.get([
                    run_replication_lr.remote(n, T, eta_0, gamma1, covariate_dim, test_points, basis_fn, seed=j)
                    for j in range(replication)
                ])
                predicted_theta_values = np.array(results)
                bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
                mse_list.append(np.log2(mse))
                print(f"LR, n = {n}, T = {T}, bias_squared: {bias}, variance: {variance}, mse: {mse}")
            df[method] = mse_list
            df['n'] = total_budget 
        df_long = df.melt(id_vars='n', var_name='Method', value_name='MSE')
        
        
        data1.append(df_long)
        df.to_csv('dim{}.csv'.format(covariate_dim), index=False)
        methods = ['linear', 'quadratic','cubic']
       
      
   
    print("over")
    picture_plot1(data1)
    
    ray.shutdown()
    print("Ray has been shutdown.")
   