import pandas as pd
import numpy as np
import copy
from scipy.optimize import minimize

def zero_col(df):
    zero_cols = []
    for i in range(df.shape[1]):
        test = np.zeros([df.shape[0]])
        if np.array_equal(df[i], test):
            zero_cols.append(i)
            df.pop(i)
    return np.asmatrix(df), zero_cols

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid*(1-sigmoid)


def computeCost(theta, X, y, input_layer_size, hidden_layer_size, num_labels, Lambda, conj, regularizada):

    thetan = []
    if conj:
        thetan = reshapeTheta(theta, input_layer_size, hidden_layer_size, num_labels)
    else:
        thetan = theta
        
    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m,1)),X))

    y10 = np.zeros((m,num_labels))
    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)

    an = []
    an.append(X)
    for i in range(len(thetan)):
        an.append(sigmoid(an[i] @ thetan[i].T))
        if i != len(thetan) - 1:
            an[-1] = np.hstack((np.ones((m,1)),an[-1]))


    for j in range(num_labels):
        J = J + sum(-y10[:,j]*np.log(an[-1][:,j])-(1-y10[:,j])*np.log(1-an[-1][:,j]))

    cost = 1/m*J
    
    if regularizada:
        aux = 0
        for i in range(len(thetan)):
            aux2 = thetan[i][:,1:].reshape([1, thetan[i][:,1:].shape[0] * thetan[i][:,1:].shape[1]])
            aux = aux + np.sum(aux2**2)
        cost = cost + Lambda/(2*m)*(aux)

    grad = []
    for i in range(len(thetan)):
        grad.append(np.zeros((thetan[i].shape)))           

    for i in range(m):
        xi = X[i,:]
        ani = []
        dn = []
        for j in range(len(an)):
            ani.append(an[j][i,:])   
     
        dn.append(np.array(ani[-1] - y10[i,:]))
        for j in range(len(thetan)-1):
            z = np.array(np.hstack((np.ones([1,1]),(ani[len(thetan)-2-j].reshape(1,ani[len(thetan)-2-j].size) @ thetan[len(thetan)-2-j].T))))
            aux = np.array(dn[0] @ thetan[len(thetan)-1-j]) * sigmoidGradient(z)
            dn.insert(0,aux[:,1:])

        
        for j in range(len(thetan)):
            grad[j] = grad[j] + (ani[j].reshape(1,ani[j].size).T @ dn[j].reshape(1,dn[j].size)).T

    for j in range(len(grad)):                        
        grad[j] = np.array(1/m*grad[j])
        if regularizada:
            grad[j] = grad[j] + (Lambda/m)*np.hstack((np.zeros((thetan[j].shape[0],1)),thetan[j][:,1:]))
            
            
    if conj:
        return float(cost)
    else:
        return float(cost), grad
    
def randInitializeWeights(L_in,L_out):
    epi = (6**1/2)/(L_in+L_out)**1/2
    W = np.random.rand(L_out,L_in+1)*(2*epi)-epi
    return W

def gradientDescent(theta,X, y, alpha, nbr_iter, Lambda, input_layer_size, hidden_layer_size, num_labels, regularizada):
    thetan = theta
    J_history = []
    conj = False
    for i in range(nbr_iter):
        cost, grad = computeCost(thetan,X,y,input_layer_size,hidden_layer_size,num_labels,Lambda,conj,regularizada)
        for j in range(len(thetan)): 
            thetan[j] = thetan[j] - (alpha*grad[j])
        J_history.append(cost)
        
    return thetan,J_history

def prediction(X,thetan):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    an = [X]
    for i in range(len(thetan)):
        an.append(sigmoid(an[i] @ thetan[i].T))
        if i != len(thetan) - 1:
            an[-1] = np.hstack((np.ones((m,1)),an[-1]))  

    return np.argmax(an[-1],axis=1)+1

def backpropagation(X, y, num_labels, hidden_layer_size, Lambda, alpha, nbr_iter, regularizada=True):
    input_layer_size = X.shape[1]
    n_layers = hidden_layer_size.shape[0]

    theta = []
    theta.append(randInitializeWeights(input_layer_size,hidden_layer_size[0]))
    if(n_layers > 1):
        for i in range(hidden_layer_size.shape[0]-1):
            theta.append(randInitializeWeights(hidden_layer_size[i],hidden_layer_size[i+1]))
    theta.append(randInitializeWeights(hidden_layer_size[-1],num_labels))

    theta, J_history = gradientDescent(theta, X, y, alpha, nbr_iter, Lambda, input_layer_size, hidden_layer_size, num_labels, regularizada)
    
    return theta, J_history

def checkGradient(X, y, num_labels, input_layer_size, hidden_layer_size, Lambda, alpha, eps, regularizada):
   
    conj = False
    n_layers = hidden_layer_size.shape[0]

    theta = []
    theta.append(randInitializeWeights(input_layer_size,hidden_layer_size[0]))
    if(n_layers > 1):
        for i in range(hidden_layer_size.shape[0]-1):
            theta.append(randInitializeWeights(hidden_layer_size[i],hidden_layer_size[i+1]))
    theta.append(randInitializeWeights(hidden_layer_size[-1],num_labels))
    
    theta_minus = copy.deepcopy(theta)
    theta_plus = copy.deepcopy(theta)

    grad_approx = []
    for k in range(len(theta)):
        _G= []
        for r in range(theta[k].shape[0]):
            _g = []
            for s in range(theta[k][r].shape[0]):
                theta_minus[k][r][s] = theta[k][r][s] - eps
                theta_plus[k][r][s] = theta[k][r][s] + eps
                c_minus, c_plus = 0, 0
                c_minus, grad = computeCost(theta_minus,X,y,input_layer_size,hidden_layer_size,num_labels,Lambda,conj,regularizada)
                c_plus, grad = computeCost(theta_plus,X,y,input_layer_size,hidden_layer_size,num_labels,Lambda,conj,regularizada)
                
                _g.append(((np.asarray(c_plus)) - (np.asarray(c_minus)))/(2*eps))
                
                theta_minus[k][r][s] = theta[k][r][s]
                theta_plus[k][r][s] = theta[k][r][s]
            _G.append(_g)
        grad_approx.append(np.asarray(_G))
   
    
    cost, grad = computeCost(theta,X,y,input_layer_size,hidden_layer_size,num_labels,Lambda,conj,regularizada)
    
    _grad = []
    _grad_approx = []
    for i in range(len(grad)):
        for j in range(np.asarray(grad[i]).shape[0]):
            _grad = np.append(_grad,grad[i][j])
            _grad_approx = np.append(_grad_approx,grad_approx[i][j])
    
    return _grad, _grad_approx

def gradientConjugate(X, y, alpha, nbr_iter, Lambda, input_layer_size, hidden_layer_size, num_labels, regularizada):
    conj = True
    n_layers = hidden_layer_size.shape[0]
    theta = []
    theta.append(randInitializeWeights(input_layer_size,hidden_layer_size[0]))
    if(n_layers > 1):
        for i in range(hidden_layer_size.shape[0]-1):
            theta.append(randInitializeWeights(hidden_layer_size[i],hidden_layer_size[i+1]))
    theta.append(randInitializeWeights(hidden_layer_size[-1],num_labels))
    thetan = theta
    for j in range(len(thetan)-1):
        X0 = np.append(thetan[0].flatten(),thetan[j+1].flatten())
   
    
    res = minimize(computeCost, X0, args=(X, y, input_layer_size, hidden_layer_size, num_labels, Lambda,conj,regularizada),options={'maxiter': nbr_iter}) 
                                      
    return res

def reshapeTheta(theta, input_layer_size, hidden_layer_size, num_labels):
     
        n_layers = hidden_layer_size.shape[0]
        thetan = []
        aux_begin = ((input_layer_size+1)*hidden_layer_size[0])
        thetan.append(theta[:aux_begin].reshape(hidden_layer_size[0],input_layer_size+1))
        if(n_layers > 1):
            for i in range(hidden_layer_size.shape[0]-1):
                aux_end = aux_begin + (hidden_layer_size[i]+1)*hidden_layer_size[i+1]
                thetan.append(theta[aux_begin:aux_end].reshape(hidden_layer_size[i+1],hidden_layer_size[i]+1))                                                 
                aux_begin = aux_end
        else:
            aux_end = aux_begin
        thetan.append(theta[aux_end:].reshape(num_labels,hidden_layer_size[-1]+1))
        return thetan
    