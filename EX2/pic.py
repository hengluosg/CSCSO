import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
plt.rcParams['font.family'] = 'Times New Roman'
def prepare_plot_data(n_values, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr):
    data = []
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_knn[i], "Variance": variance_knn[i], "Bias^2": bias_knn[i], "Method": f"k-NN"})


    # 处理KRR的MSE、Variance、Bias^2
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Variance": variance_krr[i], "Bias^2": bias_krr[i], "Method": "KRR"})
    

    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_ks[i], "Variance": variance_ks[i], "Bias^2": bias_ks[i], "Method": "KS"})
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_lr[i], "Variance": variance_lr[i], "Bias^2": bias_lr[i], "Method": "LR"})
    
    return pd.DataFrame(data)


def picture_plot1(df, covariate_dim1):
    sns.set_theme()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, column in enumerate(df):
        
        column = column.copy()
        column['log2_MSE'] = np.log2(column['MSE'])

        sns.lineplot(data=column, ax=axes[i], x='n', y='log2_MSE', hue='Method', style='Method', 
                     markers=['o', 's', 'D', '^'], dashes=False, markersize=10, linewidth=3)
        
        axes[i].set_xscale('log', base=2)
        axes[i].set_xlabel(r'Total budget $(\Gamma)$', fontsize=15)
        axes[i].set_ylabel(r'$\log_{2}$(MSE)', fontsize=15)
        axes[i].set_title(r'$d = {} $'.format(covariate_dim1[i]), fontsize=15)

        xticks = sorted(column['n'].unique())
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in xticks], fontsize=15)

        axes[i].tick_params(axis='y', labelsize=15)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('ex2_1.pdf', format='pdf')



total_budget = 2 ** np.arange(8,15)
    
data1 = []
covariate_dim1 = [2,4,9]
for covariate_dim in covariate_dim1:
    
    if covariate_dim == 2:
        df = pd.read_csv('e2dim2.csv')
        
    elif covariate_dim == 4:
        df = pd.read_csv('e2dim4.csv')
    else:
        df = pd.read_csv('e2dim9.csv')
   

  

    df_knn = df[df['Method'].str.strip() == 'k-NN'].sort_values(by='n')
    df_krr = df[df['Method'].str.strip() == 'KRR'].sort_values(by='n')
    df_ks = df[df['Method'].str.strip() == 'KS'].sort_values(by='n')
    df_lr = df[df['Method'].str.strip() == 'LR'].sort_values(by='n')

   

    mse_knn = df_knn['MSE'].values
    variance_knn = df_knn['Variance'].values
    bias_knn = df_knn['Bias^2'].values

    mse_krr = df_krr['MSE'].values
    variance_krr = df_krr['Variance'].values
    bias_krr = df_krr['Bias^2'].values

    mse_ks = df_ks['MSE'].values
    variance_ks = df_ks['Variance'].values
    bias_ks = df_ks['Bias^2'].values

    mse_lr = df_lr['MSE'].values
    variance_lr = df_lr['Variance'].values
    bias_lr = df_lr['Bias^2'].values



    
    
    df = prepare_plot_data(total_budget, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr)
    data1.append(df)
    methods = ['k-NN', 'KRR','KS','LR']
    
picture_plot1(data1, covariate_dim1)