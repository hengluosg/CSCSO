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


def project(theta, lowbound, upbound):
    return np.clip(theta, lowbound, upbound)



def grad_f(theta, x , num_samples):
    x_sqrt = np.sqrt(x)
    x = x.reshape(-1, 1)
    d = theta.shape[0]
    
    W = np.diag(np.arange(1, d+1))  
    grad1 = 2 * W @ (theta - x_sqrt)  
    part2 = 2 *lambda1 * x @ x.T @ theta
    
    grad = grad1 + part2
    gradients = np.zeros((num_samples, d))
    for rep in range(num_samples):
        noise = np.random.normal(0, std, d)  
        gradients[rep,:] = grad  + noise
    
    avg_gradient = np.mean(gradients, axis=0)
    
    return avg_gradient

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



def knn_online_stage(covariates_points, theta_estimates, x, k):
   
    distances = euclidean_distances([x], covariates_points).flatten()
    nearest_neighbors_indices = np.argsort(distances)[:k]
    nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    theta_hat = np.mean(nearest_theta_estimates, axis=0)
    return theta_hat


def gaussian_kernel(x, x_i, h):
    distance = np.linalg.norm(x - x_i)  
    return np.exp(-(distance ** 2) / (2 * h ** 2))




def ks_online_stage(x, x_train, theta_train, h):
    kernel_values = np.array([gaussian_kernel(x, x_i, h) for x_i in x_train])
    kernel_values = np.expand_dims(kernel_values, axis=1)  # (n, 1)
  
    numerator = np.sum(kernel_values * theta_train, axis=0)
    denominator = np.sum(kernel_values)
    if denominator == 0:
        print("Warning: Denominator is zero. Returning 0 for theta_hat.")
        return 0
    theta_hat = numerator / denominator
    return theta_hat




def true_theta_function(x):
    Ax_sqrt =  np.sqrt(x).T
    x = x.reshape(-1, 1)
    L = np.diag(np.arange(1, covariate_dim + 1))  
    L_plus_W_inv = np.linalg.inv(L + lambda1* x@ x.T)
    theta = L_plus_W_inv @ L  @ Ax_sqrt
    
    return theta.T



def prepare_data_for_plot(n_values, mse_knn, mse_krr, mse_ks,mse_lr):
    data = []
   
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_knn[i]), "Method": "KNN"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_krr[i]), "Method": "KRR"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_ks[i]), "Method": "KS"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_lr[i]), "Method": "LR"})
    return pd.DataFrame(data)


def prepare_plot_data(n_values, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr):
    data = []
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_knn[i], "Variance": variance_knn[i], "Bias^2": bias_knn[i], "Method": f"k-NN"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Variance": variance_krr[i], "Bias^2": bias_krr[i], "Method": "KRR"})

    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_ks[i], "Variance": variance_ks[i], "Bias^2": bias_ks[i], "Method": "KS"})
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_lr[i], "Variance": variance_lr[i], "Bias^2": bias_lr[i], "Method": "LR"})
    

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



def plot_metrics_for_each_method(df, methods ,d):
    sns.set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  
    axes = axes.flatten()  
    for i, method in enumerate(methods):
        method_df = df[df['Method'] == method].copy()
        method_df['log_n'] = np.log2(method_df['n'])  
        sns.lineplot(data=method_df, x='n', y='MSE', label='MSE', marker='o',  markersize=10,ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Bias^2', label=r"Bias$^2$", marker='D', markersize=10, ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Variance', label='Variance', marker='H', markersize=10, ax=axes[i], linewidth=3)
        axes[i].set_title(f"{method} Performance Metrics")
        
        axes[i].set_xlabel(r'Total budget $(\Gamma)$')
        axes[i].set_ylabel('Value')
        axes[i].set_xscale('log', base=2)  
        axes[i].tick_params(labelsize=20)
        axes[i].text(
        0.95, 0.05, 
        f'Covariate Dimension: {covariate_dim}',  
        transform=axes[i].transAxes,  
        fontsize=14, 
        verticalalignment='bottom', 
        horizontalalignment='right')
        axes[i].grid(True)
        
    plt.tight_layout()  
    plt.savefig('dex_1_d={}.pdf'.format(d), format='pdf')  


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
    best_mse = float('inf') 

   


    for lambda_param in lambda_vals:
        mse_scores = [] 
        for train_index, val_index in kf.split(covariates_points):
           
            X_train, X_val = covariates_points[train_index], covariates_points[val_index]
            y_train, y_val = theta_estimates[train_index], theta_estimates[val_index] 
            K_phi_inv = compute_inverse_kernel_matrix(X_train, lambda_param=lambda_param, length_scale=length_scale)
            predicted_theta_values = np.array([krr_online_stage(X_train, y_train, x, K_phi_inv, length_scale) for x in X_val])
            mse = mean_squared_error(y_val, predicted_theta_values)
            mse_scores.append(mse)
        average_mse = np.mean(mse_scores)
        if average_mse < best_mse:
            best_mse = average_mse
            best_lambda = lambda_param

    print(f"Best Lambda: {best_lambda} with MSE: {best_mse}")
    return best_lambda



"""
    Linear regression
"""

def basis_functions(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    return np.hstack([np.ones((x.shape[0], 1)),x-1]) #
   
def linear_regression_train(x_train, theta_train):
   
    Phi = basis_functions(x_train)
    
    print(Phi.shape[0])
    
    n = Phi.shape[1]
    # Compute (Phi.T @ Phi + lambda_reg * I)^(-1) @ Phi.T @ theta_train
    Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi)  # Regularized inversion
    # Compute the final beta_hat using the regularized solution
    beta_hat = Phi_T_Phi_inv @ Phi.T @ theta_train
    return beta_hat

def cross_validate(x_train, theta_train):

    beta_hat = linear_regression_train(x_train, theta_train)

    return beta_hat






def linear_regression_on_stage(x, beta_hat):
    
    phi_x = basis_functions(x)  # (s,)
    theta_hat = phi_x @ beta_hat  # (d,)
    return theta_hat



def picture_plot1(df):
    #sns.set(font_scale=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, column in enumerate(df):
        sns.set_theme()
        sns.lineplot(data=column, ax=axes[i], x='n', y='MSE', hue='Method', style='Method', 
                     markers=['o', 's', 'D', '^'], dashes=False, markersize=10, linewidth=3)
        
        axes[i].set_xscale('log', base=2)
        axes[i].set_xlabel(r'Total budget $(\Gamma)$', fontsize=15)
        axes[i].set_ylabel(r'$log_{2}(MSE)$', fontsize=15)
        axes[i].set_title(r'$d = {} $'.format(covariate_dim1[i]), fontsize=15)

        # Set x-axis tick positions to unique 'n' values and format them as log_2(n)
        xticks = sorted(column['n'].unique())
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in xticks], fontsize=15)

        axes[i].tick_params(axis='y', labelsize=15)
        axes[i].grid(True)
        
       
    
    plt.tight_layout()
    plt.savefig('ex_1.pdf', format='pdf')  
    


@ray.remote
def run_replication(n, T, covariate_dim, eta_0, gamma1, test_points, lambda_vals,seed = 3):
    set_global_seed(seed)
    covariates_points,theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
    
    # Perform cross-validation to determine best lambda for KRR
    best_lambda = krr_cross_validation(covariates_points, theta_estimates, lambda_vals=lambda_vals)
    
    # Compute inverse kernel matrix with the best lambda
    K_phi_inv = compute_inverse_kernel_matrix(covariates_points, lambda_param=best_lambda, length_scale=1)
    
    # Compute predictions for test points using KRR online stage
    predicted_theta_value = np.array([krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1) for x in test_points])
    
    return predicted_theta_value





@ray.remote
def run_replication_lr(n, T, eta_0, gamma1, covariate_dim, test_points,seed = 3):
    
    set_global_seed(seed)
    
    covariates_points,theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
    
    
    # Perform cross-validation to obtain the beta_hat
    beta_hat = cross_validate(covariates_points, theta_estimates)

    # Predict theta values using linear regression on stage
    predicted_theta_value = np.array([linear_regression_on_stage(x, beta_hat) for x in test_points])
    

    
    return predicted_theta_value

@ray.remote
def run_knn_replication(n, T, eta_0, gamma1, covariate_dim, test_points, k,seed = 3):
    set_global_seed(seed)
    covariates_points,theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
    predicted_theta_value = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
    return predicted_theta_value




@ray.remote
def run_ks_replication(n, T, eta_0, gamma1, covariate_dim, test_points,  h, seed =3):
   
    set_global_seed(seed)
    covariates_points,theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
    predicted_theta_value = np.array([ks_online_stage(x, covariates_points, theta_estimates, h) for x in test_points])

    return predicted_theta_value

if __name__ == '__main__':
    lowbound = 0
    upbound = 2

    eta_0 = 0.1
    gamma1 = 0.01 
    
    num_samples = 1  # Number of samples for true theta calculation
    lambda1 = 0.1 
    std = 4
    num_test_points = 1  #Online stage
    set_global_seed(3)
    total_budget = 2 ** np.arange(10, 17)
  
  
  
    timing_results = {}
    
    data1 = []
    covariate_dim1 = [2,5,8]
    for covariate_dim in covariate_dim1:
        # if covariate_dim == covariate_dim1[2]:
        #     eta_0 = 0.05
        # elif covariate_dim == covariate_dim1[1]:
        #     eta_0 = 0.03
        # else:
        #     eta_0 = 0.01
            
      
      
        
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
        true_theta_values = true_theta_function(test_points)
        #true_theta_values = true_theta_function(test_points)
        
        
        bias_knn,variance_knn, mse_knn = [],[],[]
        n_knn,T_knn = [],[]
        bias_krr,variance_krr, mse_krr = [],[],[]
        n_krr,T_krr = [],[]
        bias_ks,variance_ks, mse_ks = [],[],[]
        n_ks,T_ks = [],[]
        bias_lr,variance_lr, mse_lr = [],[],[]
        n_lr,T_lr = [],[]
        replication = 100 #30 50
        
        
        timing_results[covariate_dim] = {}
        time_lr, time_knn, time_krr, time_ks = [],[],[],[]
        "Linear Regression (LR)"
      
        start = time.time()
        for i, gamma in enumerate(total_budget):
            start = time.time()
            n = int(gamma ** (1 / 2))
            T = int(gamma ** (1 / 2))


            #covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
            # Call the remote function for parallel execution
            replication_results = ray.get([run_replication_lr.remote(n, T, eta_0, gamma1, covariate_dim, test_points ,seed = j)  for j in range(replication)])
            predicted_theta_values = np.array(replication_results)
            
            # Get the results from the parallel execution (this is a blocking call)
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            
            # Store the results
            bias_lr.append(bias)
            variance_lr.append(variance)
            mse_lr.append(mse)
            n_lr.append(n)
            T_lr.append(T)
            print(f"LR, n = {n}, T = {T}, bias_squared: {bias}, variance: {variance}, mse: {mse}")

            t_time = time.time() - start
            time_lr.append(t_time)


        timing_results[covariate_dim]["LR"] = time_lr
        # "Kernel Ridge Regression (KRR)"
        
    

        "Kernel Ridge Regression (KRR)"
        for i, gamma in enumerate(total_budget):
            start = time.time()
            n = int(gamma ** (2 / 3))
            T = int(gamma / n)
           
            
            #covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
            
            # Define replication lambda values
            if covariate_dim==covariate_dim1[0]:
                
                lambda_vals = np.arange(1, 40, 5)
                
            elif covariate_dim==covariate_dim1[1]:
                lambda_vals = np.arange(0.1, 10, 0.5)
            else:
                
                lambda_vals = np.arange(0, 2, 0.05)
            
           
            # Parallelize the replication process
            replication_results = ray.get([run_replication.remote(n, T, covariate_dim, eta_0, gamma1, test_points, lambda_vals,seed = j) for j in range(replication)])
            
            # Combine results from all replications
            predicted_theta_values = np.array(replication_results)
            
            # Continue with the calculation of bias, variance, and MSE
            
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            bias_krr.append(bias)
            variance_krr.append(variance)
            mse_krr.append(mse)
            n_krr.append(n)
            T_krr.append(T)
            print(f"KRR, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")
            t_time = time.time() - start
            time_krr.append(t_time)
    

        timing_results[covariate_dim]["KRR"] = time_krr

        "Kernel Smoothing (KS)"
        
        for i, gamma in enumerate(total_budget):
           
            start = time.time()
            h = gamma**(-1/(covariate_dim+2))
            T = 3*int(gamma**((1)/(covariate_dim+2)))
            n = int(gamma / T)
            
           
            replication_results = ray.get([run_ks_replication.remote(n, T, eta_0, gamma1, covariate_dim, test_points, h,seed = j) for j in range(replication)])
            
            # Get the results once all replications are done
            predicted_theta_values = np.array(replication_results)
            
            # Continue with the calculation of bias, variance, and MSE
            
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)

            # Store the results
            bias_ks.append(bias)
            variance_ks.append(variance)
            mse_ks.append(mse)
            n_ks.append(n)
            T_ks.append(T)

            # Print the results
            print(f"KS, n = {n}, T = {T}, bias_squared: {bias}, variance: {variance}, mse: {mse}")
            t_time = time.time() - start
            time_ks.append(t_time)

        timing_results[covariate_dim]["KS"] = time_ks
            
            

        "k-Nearest Neighbors (kNN)"

        
        for i, gamma in enumerate(total_budget):
           
            start = time.time()
            k = round(gamma ** (1 / (covariate_dim + 2)))
            T = 3*int(gamma ** (2 / (covariate_dim + 2))/k) 
            n = int(gamma / T)
            #covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
            # Launch the remote task for this gamma
            replication_results = ray.get([ run_knn_replication.remote(n, T, eta_0, gamma1, covariate_dim, test_points, k,seed = j) for j in range(replication)])
    
            # Wait for the result from Ray and append it to the lists
            predicted_theta_values = np.array(replication_results)
            
            # Continue with the calculation of bias, variance, and MSE
            
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            bias_knn.append(bias)
            variance_knn.append(variance)
            mse_knn.append(mse)
            n_knn.append(n)
            T_knn.append(T)

            # Print the results
            print(f"KNN, n = {n}, T = {T}, bias_squared: {bias}, variance: {variance}, mse: {mse}")

            t_time = time.time() - start
            time_knn.append(t_time)


        timing_results[covariate_dim]["KNN"] = time_knn
        



        df = prepare_data_for_plot(total_budget, mse_knn, mse_krr, mse_ks,mse_lr)
        data1.append(df)
        df = prepare_plot_data_updata(total_budget, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr,n_knn,n_krr,n_ks,n_lr,  T_knn, T_krr, T_ks, T_lr)
        df.to_csv('dim{}.csv'.format(covariate_dim), index=False)
        methods = ['k-NN', 'KRR','KS','LR']
        plot_metrics_for_each_method(df, methods,covariate_dim)
    
    print(timing_results)
    print("over")
    picture_plot1(data1)
    
    dfs_by_dim = {}
    for dim, methods in timing_results.items():
        rows = []
        for i in range(len(next(iter(methods.values())))):  
            row = {'Run': i + 1}
            for method, values in methods.items():
                row[method] = round(values[i], 2)
            rows.append(row)
        df_dim = pd.DataFrame(rows)
        dfs_by_dim[dim] = df_dim
        df_dim.to_csv(f"timing_results_d{dim}.csv", index=False)


    ray.shutdown()
    print("Ray has been shutdown.")

