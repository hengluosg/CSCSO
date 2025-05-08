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
plt.rcParams['font.family'] = 'Times New Roman' 
# import ray
# ray.init(num_cpus=30)  
def assign_regions(base_stations, grid_size=100, boundary_min=0, boundary_max=10):
    x = np.linspace(boundary_min, boundary_max, grid_size)
    y = np.linspace(boundary_min, boundary_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid into list of (x, y) points

    num_base_stations = base_stations.shape[0]
    distances = np.zeros((grid_points.shape[0], num_base_stations))
    for i, (x_bs, y_bs) in enumerate(base_stations):
        distances[:, i] = np.linalg.norm(grid_points - [x_bs, y_bs], axis=1)

    nearest_base_station = np.argmin(distances, axis=1)
    regions = []
    for i in range(num_base_stations):
        region_points = grid_points[nearest_base_station == i]
        regions.append(region_points)
    return regions




def compute_stochastic_gradient(a, G, user_weights, user_noise):
    gradients = np.zeros(L)
    total_weight = sum(np.sum(user_weights[l]) for l in range(L))
    
    for l_star in range(L):
        # First term: \\sum_{u \\in \\mathcal{U}_{l^*}} w_u
        first_term = np.sum(user_weights[l_star])
        
        # Second term: \\sum_{l \\neq l^*} \\sum_{u \\in \\mathcal{U}_l} w_u \\frac{G_{u,l^*} e^{a_{l^*}}}{\\eta_u + \\sum_{l' \\in \\mathcal{N}_l} G_{u,l'} e^{a_{l'}}}
        second_term = 0.0
        for l in range(L):
            if l != l_star:
                for u, w_u in enumerate(user_weights[l]):
                    
                    numerator = G[l][u][l_star] * np.exp(a[l_star])
                    denominator = user_noise[l][u] + np.sum([G[l][u][l_prime] * np.exp(a[l_prime]) for l_prime in [i for i in range(L) if i != l]]) 
                    second_term += w_u * (numerator / denominator)
        
        # Compute gradient for l_star
        gradients[l_star] = -(c1 / total_weight) * (first_term - second_term)+ c2 * np.exp(a[l_star])
    return gradients


def objective_function(a, G_list, user_weights_list, user_noise_list, c1, c2, regions):
    
    L = len(regions)  # Number of base stations
    total_weight = sum(np.sum(weights) for weights in user_weights_list)  # Total weight across all regions

    # First term: weighted log-sum-exp (with f = 1, no feature loop)
    first_term = 0
    for l in range(L):  # Iterate over all base stations
        region_users = regions[l]  # Users in base station l
        G = G_list[l]  # Gain matrix for region l
        user_weights = user_weights_list[l]
        user_noise = user_noise_list[l]

        for u in range(G.shape[0]):  # Iterate over users in region l
            numerator = G[u, l] * np.exp(a[l])
            denominator = user_noise[u] + np.sum([
                G[u, l_prime] * np.exp(a[l_prime]) for l_prime in range(L) if l_prime != l
            ])
            # if numerator > 0 and denominator > 0:
            #     first_term += user_weights[u] * np.log(numerator / denominator)
            
            first_term += user_weights[u] * np.log(numerator / denominator)

    first_term *= c1 / total_weight

    # Second term: regularization penalty (with f = 1)
    second_term = c2 * np.sum(np.exp(a))

    # Combine the terms
    h_a = -first_term + second_term
    return h_a

def simulation_algorithm(covariates, base_stations, regions, L, n, T, eta, sigma):
    
    # Initialization
    ln_p_min = np.log(p_lowbound)  # -2.3026
    ln_p_max = np.log(p_upbound)   # 2.3026
    a_params = [np.random.uniform(ln_p_min, ln_p_max, L) for _ in range(n)]  # Initial parameters
    averaged_params = []

    # Main algorithm
    for i, covariate in enumerate(covariates):
        a_t = a_params[i]
        a_t1 = []

        for t in range(T):
            G_list, user_weights_list, user_noise_list = [], [], []
            user_locations_total =[]
            for l, region_points in enumerate(regions):
                num_users = np.random.poisson(covariate[l])
                user_weights = 2 * np.ones(num_users)
                user_weights_list.append(user_weights)
                user_noise = np.full(num_users, sigma)
                user_noise_list.append(user_noise)

                if num_users > 0:
                    
                    user_locations = region_points[np.random.choice(region_points.shape[0], num_users, replace=False)]
                    #print("User Locations:", user_locations)
                    user_locations_total.append(user_locations)
                    G = []
                    for user in user_locations:
                        row = []
                        for bs in base_stations:
                            distance = np.linalg.norm(user - bs)
                            min_distance = 0.0000001
                            distance = max(distance, min_distance)
                            if distance <= 7:
                                gain = np.random.uniform(10**-14.4 * distance**-5 * 0.75, 10**-14.4 * distance**-5 * 1.25)
                                row.append(gain)
                            else:
                                row.append(0)
                        G.append(row)
                    G = np.array(G)

                G_list.append(G)

            # Compute stochastic gradient
            gradient = compute_stochastic_gradient(a_t, G_list, user_weights_list, user_noise_list)
            a_t = a_t - eta * gradient
            a_t1.append(a_t)

            # if t == 1:
            #     h_a = objective_function(a_t, G_list, user_weights_list, user_noise_list, c1=4, c2=0.1, regions=regions)
            #     print(f"Objective Function Value for Iteration {i}, Time Step {t}: {h_a}")

        # Compute averaged parameter
        averaged_a = np.mean(a_t1, axis=0)
        #print(f"Iteration {i}, covariate is {covariate}: {averaged_a}")
        averaged_params.append(averaged_a)

        # h_a = objective_function(averaged_a, G_list, user_weights_list, user_noise_list, c1=4, c2=0.1, regions=regions)
        # print(f"Objective Function Value for Iteration {i}: {h_a}")

    # Final averaged parameters
    averaged_params = np.array(averaged_params)
    return averaged_params,user_locations_total

def plot_regions(base_stations):
    
    regions = assign_regions(base_stations)
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'green', 'yellow', 'orange']

    for i, region_points in enumerate(regions):
        plt.scatter(region_points[:, 0], region_points[:, 1], s=1, color=colors[i % len(colors)], label=f"Region {i+1}")

    plt.scatter(base_stations[:, 0], base_stations[:, 1], color='red', s=100, label='Base Stations', zorder=5)
    for i, (x, y) in enumerate(base_stations):
        plt.text(x, y, f"BS{i+1}", color='black', fontsize=12)

    plt.title("2D Space Partitioned by Nearest Base Station")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def grid_sample(d, lowbound, upbound, point):
    n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]

    #grid_points = (grid_points - lowbound) / (upbound - lowbound) 
    return grid_points
    
def compute_bias_variance_mse(true_value, predicted_values):
    predicted_values = np.array(predicted_values)
    mean_predicted = np.mean(predicted_values, axis=0)  
    bias = mean_predicted - true_value  
    
    bias_squared = np.mean(bias ** 2)  
    variance = np.mean(np.var(predicted_values, axis=0))  
    mse = np.mean(np.mean((predicted_values - true_value) ** 2, axis=1))  
    return bias_squared, variance, mse

"""
    Linear regression
"""

def basis_functions(x):
  
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    return np.hstack([np.ones((x.shape[0], 1)),x-1.5]) #
def linear_regression_train(x_train, theta_train):
    # Apply basis functions to the input data (x_train)
    
    #Phi = np.array([basis_functions(x) for x in x_train])  # Assuming basis_functions is predefined
    Phi = basis_functions(x_train)
    
    print(Phi.shape[0])
    # Number of features (s)
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

# @ray.remote
def run_replication_lr(n, T, L, eta_0, BS_positions, test_points):
    covariates = grid_sample(L, lowbound_cov, upbound_cov, n)
    regions = assign_regions(BS_positions)
    theta_estimates = simulation_algorithm(covariates, BS_positions, regions, L, n, T, eta_0, sigma)
    
    beta_hat = cross_validate(covariates, theta_estimates)
    predicted_theta_value = linear_regression_on_stage(test_points, beta_hat)

    return predicted_theta_value

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


def krr_cross_validation(covariates_points, theta_estimates, lambda_vals=[0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10], n_splits=10, length_scale=1.0):
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
        if average_mse < best_mse:
            best_mse = average_mse
            best_lambda = lambda_param

    print(f"Best Lambda: {best_lambda} with MSE: {best_mse}")
    return best_lambda




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

def gaussian_kernel(x, x_i, h):
    distance = np.linalg.norm(x - x_i)  
    return np.exp(-(distance ** 2) / (2 * h ** 2))




def knn_online_stage(covariates_points, theta_estimates, x, k):
    
    distances = euclidean_distances([x], covariates_points).flatten()
    nearest_neighbors_indices = np.argsort(distances)[:k]
    nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    theta_hat = np.mean(nearest_theta_estimates, axis=0)
    return theta_hat



# @ray.remote
def run_replication(n, T, L, eta_0, BS_positions, test_points):
    covariates = grid_sample(L, lowbound_cov, upbound_cov, n)
    
    regions = assign_regions(BS_positions)
    theta_estimates = simulation_algorithm(covariates, BS_positions, regions, L, n, T, eta_0, sigma)
    

    covariates , theta_estimates = normalize(covariates, lowbound_cov, upbound_cov), normalize(theta_estimates, p_lowbound, p_upbound)
    # Perform cross-validation to determine best lambda for KRR
    best_lambda = krr_cross_validation(covariates, theta_estimates)
    
    # Compute inverse kernel matrix with the best lambda
    K_phi_inv = compute_inverse_kernel_matrix(covariates, lambda_param=best_lambda, length_scale=1)
    
    # Compute predictions for test points using KRR online stage
    test_points = normalize(test_points, lowbound_cov, upbound_cov)
    predicted_theta_value = np.array([krr_online_stage(covariates, theta_estimates, x, K_phi_inv, length_scale=1) for x in test_points])
    

    predicted_theta_value = denormalize(predicted_theta_value, p_lowbound,p_upbound, low=0.0, high=1.0)
    return predicted_theta_value


# @ray.remote
def run_ks_replication(n, T, L, eta_0, BS_positions,  test_points,  h):
   

    covariates = grid_sample(L, lowbound_cov, upbound_cov, n)
    regions = assign_regions(BS_positions)
    theta_estimates = simulation_algorithm(covariates, BS_positions, regions, L, n, T, eta_0, sigma)
    
    # Apply KS online stage for each test point
    predicted_theta_value = np.array([ks_online_stage(x, covariates, theta_estimates, h) for x in test_points])

    return predicted_theta_value


# @ray.remote
def run_knn_replication(n, T, L, eta_0, BS_positions,  test_points, k):

    covariates = grid_sample(L, lowbound_cov, upbound_cov, n)
    regions = assign_regions(BS_positions)
    theta_estimates = simulation_algorithm(covariates, BS_positions, regions, L, n, T, eta_0, sigma)
    
    # Apply KNN on each test point
    predicted_theta_value = np.array([knn_online_stage(covariates, theta_estimates, x, k) for x in test_points])

    return predicted_theta_value







def picture_plot1(df):
    #sns.set( font_scale = 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, column in enumerate(df):
        sns.set_theme()
        sns.lineplot(data=column,ax = axes[i], x='n', y='MSE', hue='Method', style='Method',markers=[ 'o','s', 'D', '^'], dashes=False, markersize=10,linewidth=3)
        axes[i].set_xscale('log', base=2)
        axes[i].set_xlabel('Total budget', fontsize=15)
        axes[i].set_ylabel(r'$log_{2}(MSE)$', fontsize=15)
        axes[i].set_title(r'$d_x = {} $'.format(covariate_dim1[i]), fontsize=15)
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
        #xes[i].set_ylim([-6, 5])
        #axes[i].tick_params(labelsize=20)
        axes[i].grid(True)
        axes[i].text(
            0.95, 0.05, 
            f'Covariate Dimension: {covariate_dim1[i]}', 
            transform=axes[i].transAxes, 
            fontsize=14, 
            verticalalignment='bottom', 
            horizontalalignment='right')
        
        axes[i].tick_params(labelsize=15)
    
    plt.tight_layout()
    plt.savefig('newtotal.png')
    #plt.show()
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


def normalize(data, data_min,data_max,low=0.0, high=1.0):
   
    range_ = data_max - data_min
     
    normalized_data = (data - data_min) / range_  
    normalized_data = normalized_data * (high - low) + low  
    return normalized_data

def denormalize(normalized_data, original_min, original_max, low=0.0, high=1.0):
    range_ = original_max - original_min
    
    original_data = (normalized_data - low) / (high - low)  
    original_data = original_data * range_ + original_min  
    return original_data






def picture_plot_stations(base_stations_list,user_locations1):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define your own color palette
    colors = sns.color_palette("Set2", 9)  # Using 'Set2' palette with 9 colors
    
    for idx, base_stations in enumerate(base_stations_list):
        sns.set_theme()
        regions = assign_regions(base_stations)
        user_locations = user_locations1[idx]
        for i, region_points in enumerate(regions):
            if len(region_points) > 0:
                # Ensure consistent color by manually assigning colors
                color = colors[i % len(colors)]
                df = pd.DataFrame(region_points, columns=["X", "Y"])
                # sns.scatterplot(x="X", y="Y", data=df, ax=axes[idx], color=color, label=f"Region {i+1}", s=30)
                sns.scatterplot(x="X", y="Y", data=df, ax=axes[idx], color=color, s=5)
        


        # Base Stations as red dots
        df_bs = pd.DataFrame(base_stations, columns=["X", "Y"])
        sns.scatterplot(x="X", y="Y", data=df_bs, ax=axes[idx], color='red', s=100, label="Base Stations", zorder=5)
        combined = np.vstack(user_locations) 

        user_label = "Users" 
        df_ser = pd.DataFrame(combined, columns=["X", "Y"])
        sns.scatterplot(x="X", y="Y", data=df_ser, ax=axes[idx],   label=user_label ,color='darkblue', s=30,marker='X')
        


        # Annotate base stations
        for i, (x, y) in enumerate(base_stations):
            axes[idx].text(x, y, f"BS{i+1}", color='black', fontsize=12)

        axes[idx].set_title(f"Base Station Configuration {idx + 1}")
        axes[idx].set_xlabel("X-axis(km)")
        axes[idx].set_ylabel("Y-axis(km)")
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig('stations.pdf', format='pdf') 
    plt.show()



def square_grid_centers(n, s):
    
    step = s / n

    
    centers = [( (2*j + 1) * step / 2, (2*i + 1) * step / 2) 
               for i in range(n) for j in range(n)]
    
    return centers


if __name__ == '__main__':
    
    np.random.seed(2)
    eta_0 = 0.5  # Initial learning rate
    F = 1  # Number of frequencies
    p_lowbound = 0.1
    p_upbound = 10
    sigma = 10**-14.4  # Noise standard deviation
    L = 4  # Number of base stations
    c1=4
    c2=0.1
    lowbound, upbound = 0, 10
    base_stations = np.array([
        [lowbound, lowbound],
        [upbound, lowbound],
        [lowbound, upbound],
        [upbound, upbound]])
    lowbound_cov, upbound_cov =  50 , 100
    total_budget = 2 ** np.arange(8, 14)
    
    base_stations_list = []
    covariate_dim1 = [2,4,9]
    user_locations1 = []
    for covariate_dim in covariate_dim1:
        
        if covariate_dim == covariate_dim1[0]:
            base_stations = np.array([[lowbound, lowbound+5],[upbound, upbound-5]])
        elif covariate_dim == covariate_dim1[1]:
            base_stations = np.array([
                                    [lowbound, lowbound],
                                    [upbound, lowbound],
                                    [lowbound, upbound],
                                    [upbound, upbound]])
        else:

            base_stations =  np.array(square_grid_centers(3, upbound))
           
            print(base_stations.shape)
        L = base_stations.shape[0]
        
        base_stations_list.append(base_stations)
    



        
        ########TEST
        n = 10
        T = 1
        covariates = grid_sample(L, lowbound_cov, upbound_cov, n)
        #print("Covariates:", covariates)
        #covariates = np.random.uniform(10, 20, size=(n, L))
        regions = assign_regions(base_stations)
        averaged_params, user_locations = simulation_algorithm(covariates, base_stations, regions, L, n, T, eta_0, sigma)
        #print("User Locations:", user_locations)
        user_locations1.append(user_locations)

    picture_plot_stations(base_stations_list,user_locations1)
