
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
#import scipy.integrate as intgr
import scipy.optimize as opt
import scipy.linalg


data = np.load('Bivariatesup_OU(nu=10,a=3,b=0.05,B=-0.1,an=1.95).npz')
selected_x_components = data['selected_x_components']
selected_y_components = data['selected_y_components']

def vech(A):
    mask = np.tril(np.ones_like(A, dtype=bool))
    return A[mask]
def data_moments(xvals, yvals):
    mean_x = np.mean(xvals)
    mean_y = np.mean(yvals)

    var_x = np.var(xvals)
    var_y = np.var(yvals)
    cov_xy = np.cov(xvals, yvals)[0, 1]
    cov_yx = np.cov(xvals, yvals)[1, 0]

    mean_data = np.array([mean_x, mean_y])
    variance_data = np.array([[var_x, cov_xy],
                              [cov_yx, var_y]])
    xp_x = xvals - mean_x
    lags = [1, 2, 3, 4, 5]
   # cov_x = np.sum(xp_x[l:] * xp_x[:-l]) / len(xp_x)
    yp_y = yvals - mean_y

    lags = [1, 2, 3, 4, 5]
    autocovariance_data = []

    for l in lags:
        cov_x = np.sum(xp_x[l:] * xp_x[:-l]) / len(xp_x)
        cov_y = np.sum(yp_y[l:] * yp_y[:-l]) / len(yp_y)
        autocov_model = np.array([[cov_x, cov_xy],
                                   [cov_yx, cov_y]])
        autocovariance_data.append(autocov_model)

    autocovariance_data = np.array(autocovariance_data)

    return mean_data, variance_data, autocovariance_data

#print(data_moments(selected_x_components, selected_y_components))

mean_data, var_data, autocov_data = data_moments(selected_x_components, selected_y_components)
print("var_data",var_data)


vech_am=[vech(A) for A in autocov_data ]

#print(vech_am)


def model_moments(mu, sigma,an):
    Adj = np.array([[0, 1],
                    [1, 0]])  # in K
    c=0.5
    an=1.95
    cAdj = c * Adj
    I2 = np.identity(2)
    cAdj += I2
    k = cAdj
 #   mu= np.array([3, 3])
  #  sigma=np.array([[0.6, 0.225],
  #                                 [0.225, 0.6]])
    mm=np.dot(np.linalg.inv(k), mu)
    mean_model=(1/(an-1))*mm

    def solve_system_equations(A, W):
        # Size of A and W
        n = A.shape[0]
        # Identity matrix
        I = np.eye(n)
        # Kronecker products
        kron_IA = np.kron(I, A)
        kron_AI = np.kron(A, I)

        # Sum of Kronecker products
        sum_kron = kron_IA + kron_AI

        # Vectorize matrices X and W
        vec_W = W.ravel()
        # Solve the system of equations
        vec_X = np.linalg.solve(sum_kron, vec_W)
        # Reshape the result to obtain matrix X
        X = vec_X.reshape((n, -1))
        return X
    result_X = solve_system_equations(-k, sigma)


    var_model=-(1/(an-1))*result_X
    lags = [1, 2, 3, 4, 5]
    autocov_model = [-((1**an)/(an-1))*np.dot(scipy.linalg.fractional_matrix_power(I2+l*k, 1-an),result_X) for l in lags]
    autocovariance_model=np.array(autocov_model)
    return mean_model, var_model,autocovariance_model

mu_init=np.array([3,3])
sigma_init=np.array([[0.6,0.225],[0.225,0.6]])
an_init=1.95
W_hat = np.eye(20)
#print("Model moments:", model_moments(mu_init,sigma_init, an_init))

mean_model, var_model, autocov_model =  model_moments(mu_init,sigma_init, an_init)
print("var_model",var_model)

vech_amm=[vech(A) for A in autocov_model ]

def err_vec(xvals,yvals, mu, sigma,an):
    mean_data, var_data, autocovariance_data = data_moments(xvals,yvals)
    moms_data = np.concatenate([mean_data.ravel(), vech(var_data)] + vech_am)
    mean_model, var_model,autocovariance_model = model_moments(mu, sigma,an)
    moms_model = np.concatenate([mean_model.ravel(), vech(var_model)] + vech_amm)
    err_vec = (moms_model - moms_data)
    return err_vec

print(err_vec(selected_x_components, selected_y_components, mu_init, sigma_init,an_init))

params_init = np.concatenate([mu_init, vech(sigma_init), [an_init]])
W_hat = np.eye(20)
def criterion(mu, sigma, an, xvals, yvals, W):
    err = err_vec(xvals, yvals, mu, sigma, an)
    crit_val = (1/1000) * np.dot(np.dot(err.T, W), err)  # 1000 is the number of observations
    return crit_val

args=(selected_x_components, selected_y_components, W_hat)

#print(criterion(mu_init,sigma_init,an_init,selected_x_components, selected_y_components, W_hat))

def criterion_wrapper(params, *args):
    mu = params[:2]
    sigma = np.array([[params[2], params[3]], [params[3], params[4]]])
    an = params[5]
    return criterion(mu, sigma, an, *args)

#print(criterion_wrapper(params_init, *args))

bounds = [(-5, None)] * len(params_init)

results = opt.minimize(criterion_wrapper, params_init, args=args, method='L-BFGS-B', bounds=bounds)

mu_GMM1 = results.x[:2]
sigma_flat_GMM1 = results.x[2:5]
sigma_GMM1 = np.array([[sigma_flat_GMM1[0], sigma_flat_GMM1[1]], [sigma_flat_GMM1[1], sigma_flat_GMM1[2]]])
an_GMM1 = results.x[5]

print('mu_GMM1=', mu_GMM1, ' sigma_GMM1=', sigma_GMM1, ' an_GMM1=', an_GMM1)

mean_data, var_data, autocovariance_data = data_moments(selected_x_components, selected_y_components)
err1 = err_vec(selected_x_components, selected_y_components, mu_init, sigma_init, an_init)
print('Mean of points =', mean_data, ', Variance of points =', var_data, ', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model, ', Variance of model =', var_model, ', Autocovariance of model =', autocov_model)
print('Error vector=', err1)
params_GMM1 = results.x
results_final = opt.minimize(criterion_wrapper, params_GMM1, args=args, method='L-BFGS-B', bounds=bounds)

mu_GMM2 = results_final.x[:2]
sigma_flat_GMM2 = results_final.x[2:5]
sigma_GMM2 = np.array([[sigma_flat_GMM2[0], sigma_flat_GMM2[1]], [sigma_flat_GMM2[1], sigma_flat_GMM2[2]]])
an_GMM2 = results_final.x[5]

print('mu_GMM2=', mu_GMM2, ' sigma_GMM2=', sigma_GMM2, ' an_GMM2=', an_GMM2)

err2 = err_vec(selected_x_components, selected_y_components, mu_GMM2, sigma_GMM2, an_GMM2)
print('Error vector=', err2)


'''
bounds = [(-5, None)] * len(params_init)

#xvals, yvals, W = (selected_x_components, selected_y_components, W_hat)
results = opt.minimize(criterion, mu_init,sigma_init,an_init,selected_x_components, selected_y_components, W_hat, method='L-BFGS-B', bounds=bounds)

mu_GMM1 = results.x[:2]
sigma_flat_GMM1 = results.x[2:5]
sigma_GMM1 = np.array([[sigma_flat_GMM1[0], sigma_flat_GMM1[1]], [sigma_flat_GMM1[1], sigma_flat_GMM1[2]]])
an_GMM1 = results.x[5]

print('mu_GMM1=', mu_GMM1, ' sigma_GMM1=', sigma_GMM1, ' an_GMM1=', an_GMM1)

mean_data, var_data, autocovariance_data = data_moments(selected_x_components, selected_y_components)
err1 = err_vec(selected_x_components, selected_y_components, mu_init, sigma_init, an_init)
print('Mean of points =', mean_data, ', Variance of points =', var_data, ', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model, ', Variance of model =', var_model, ', Autocovariance of model =', autocov_model)
print('Error vector=', err1)

params_GMM1 = results.x
results_final = opt.minimize(criterion, params_GMM1, xvals, yvals, W, method='L-BFGS-B', bounds=bounds)

mu_GMM2 = results_final.x[:2]
sigma_flat_GMM2 = results_final.x[2:5]
sigma_GMM2 = np.array([[sigma_flat_GMM2[0], sigma_flat_GMM2[1]], [sigma_flat_GMM2[1], sigma_flat_GMM2[2]]])
an_GMM2 = results_final.x[5]

print('mu_GMM2=', mu_GMM2, ' sigma_GMM2=', sigma_GMM2, ' an_GMM2=', an_GMM2)

err2 = err_vec(selected_x_components, selected_y_components, mu_GMM2, sigma_GMM2, an_GMM2)
print('Error vector=', err2)
'''