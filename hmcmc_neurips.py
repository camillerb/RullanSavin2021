# code implementing hamiltonian MCMC
import numpy as np

def grad_U(x, cov_inv, m):
    du = np.dot(cov_inv, x - m)
    return du

def U(x, cov_inv, m):
    u_x = 0.5*np.dot((x - m).T, np.dot(cov_inv,x-m))
    return u_x

def hmcmc(epsilon, L, current_q, params, skipAccept):
    q = np.copy(current_q)
    p = np.random.normal(size = q.shape)

    #make a half step
    p = p - epsilon * grad_U(q, params['cov_inv'], params['mu']) / 2
    for i in range(L):
        q = q + epsilon*p

        if i != L:
            p = p  - epsilon*grad_U(q, params['cov_inv'], params['mu'])

    p = p - epsilon * grad_U(q, params['cov_inv'], params['mu']) / 2
    p_real  = np.copy(-1*p)

    return q, -1*p

def langevin(params, epsilon, L, Tmax, init = None):
    #generate langevin sample from a specified distribution
    if init is not None:
        p0 = init
    else:
        p0 = params['mu']
    p, q = hmcmc(epsilon, 1, p0, params, 1)

    samp = np.zeros(shape = (Tmax, p0.shape[0]))
    for t in range(Tmax):
        p, q = hmcmc(epsilon, 1, p, params, 1)
        samp[t,:] = p
    return samp

def get_posterior(x_trial, mu_prior, sigma_prior, sigma_noise):
    mu_noise = np.array([0., 0.]) #observation noise mean
    A = np.array([[1 , 0], [0,  1]]) #observation model 
    
    strial = np.dot(A, x_trial)
    
    inv_l = np.einsum('ij,jk,kl->il', A.T, np.linalg.inv(sigma_noise), A)
    sigma_post = np.linalg.inv(np.linalg.inv(sigma_prior) + inv_l)
    a = np.dot(np.linalg.inv(sigma_prior), mu_prior)
    b = np.dot(A.T, np.dot(np.linalg.inv(sigma_noise), strial - mu_noise))
    mu_post = np.dot(sigma_post, a + b)
    
    return mu_post, sigma_post