import numpy as np
from scipy.optimize import minimize

def normalizacao(Y, R):
    n_filmes = Y.shape[0]
    
    media = np.zeros([n_filmes, 1])
    norm = np.zeros(Y.shape)
    
    for i in range(n_filmes):
        media[i] = np.sum(Y[i,:])/np.count_nonzero(R[i,:])
        norm[i, R[i,:]==1] = (Y[i, R[i,:]==1] - media[i])
        
    return norm, media

def cost_fun(variables, Y, R, n_pars):
    n_filmes = Y.shape[0]
    n_users = Y.shape[1]
    
    x = variables[:n_filmes * n_pars].reshape([n_filmes, n_pars])
    theta = variables[n_filmes * n_pars:].reshape([n_users, n_pars])
    
    desvios = x @ theta.T - Y
    J = np.sum((desvios**2) * R) / 2
    
    x_grad = (desvios * R) @ theta
    theta_grad = (desvios * R).T @ x
    
    grad = np.concatenate((x_grad.flatten(), theta_grad.flatten()))
    
    return J, grad

def treinamento(Y, R, n_pars, n_iter):
    n_filmes = R.shape[0]
    n_users = R.shape[1]
    
    X = np.random.rand(n_filmes, 100)
    theta = np.random.rand(n_users, 100)
    variables = np.concatenate((X.flatten(), theta.flatten()))
    
    res =  minimize(cost_fun, variables, args=(Y, R, n_pars), method='CG', jac=True, options={'maxiter': n_iter})
    
    x = res['x'][:n_filmes * n_pars].reshape([n_filmes, n_pars])
    theta = res['x'][n_filmes * n_pars:].reshape([n_users, n_pars])
    
    return x, theta, res