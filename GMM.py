'''


def data_moments(xvals):
    mean_data = xvals.mean()
    var_data = xvals.var()
    xp = xvals - mean_data
    lags=[1,2,3,4,5]
    cov = [np.sum(xp[l:] * xp[:-l]) / len(xp) for l in lags]
    autocovariance_data=np.array(cov)
    return mean_data, var_data,autocovariance_data


def model_moments(mu, sigma,B,an):
    mean_model=-mu/(B*(an-1))
    var_model=-sigma/(2*B*(an-1))
    autocovdem = 2 * B * (an - 1)
    lags = [1, 2, 3, 4, 5]
    autocov_model = [(-sigma * (1 - B * l) ** (1 - an)) / autocovdem for l in lags]
    autocovariance_model=np.array(autocov_model)
    return mean_model, var_model,autocovariance_model
def err_vec(xvals, mu, sigma,B,an):
    mean_data, var_data, autocovariance_data = data_moments(xvals)
    moms_data = np.array([mean_data, var_data]+list(autocovariance_data))
    mean_model, var_model,autocovariance_model = model_moments(mu, sigma, B,an)
    moms_model = np.array([mean_model ,var_model]+list(autocovariance_model))
    err_vec = (moms_model - moms_data)**2
    return err_vec


def criterion(params, *args):'''
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
#import scipy.integrate as intgr
import scipy.optimize as opt
import scipy.linalg


data = np.load('Bivariatesup_OU(nu=10,a=3,b=0.05,B=-0.1,an=1.95).npz')
selected_x_components = data['selected_x_components']
selected_y_components = data['selected_y_components']

def data_moments(xvals):
    mean_data = xvals.mean()
    var_data = xvals.var()
    xp = xvals - mean_data
    lags=[1,2,3,4,5]
    cov = [np.sum(xp[l:] * xp[:-l]) / len(xp) for l in lags]
    autocovariance_data=np.array(cov)
    return mean_data, var_data,autocovariance_data


#def model_moments():

mean_model=2.101
var_model=0.3419

lags = [1, 2, 3, 4, 5]
    # theoretical
D1_values = []
D2_values = []
b = np.array([[0.3419, -0.0526],
                  [-0.0526, 0.3419]])

lags = [1, 2, 3, 4, 5]
for h in lags:
    a = np.array([[1 + h, 0.5 * h],
                      [0.5 * h, 1 + h]])
    C = np.dot(scipy.linalg.fractional_matrix_power(a, -0.95), b)
    D = np.diag(C)

    D1_values.append(D[0])
    D2_values.append(D[1])


    def model_moments(mu, sigma, B, an):
        mean_model = -mu / (B * (an - 1))
        var_model = -sigma / (2 * B * (an - 1))
        autocovdem = 2 * B * (an - 1)
        lags = [1, 2, 3, 4, 5]
        autocov_model = [(-sigma * (1 - B * l) ** (1 - an)) / autocovdem for l in lags]
        autocovariance_model = np.array(autocov_model)
        return mean_model, var_model, autocovariance_model

  #  return mean_model, var_model, D1_values, D2_values
def err_vec(mu, sigma, B, an,xvals):
    mean_data, var_data, autocovariance_data = data_moments(xvals)
    moms_data = np.array([mean_data, var_data]+list(autocovariance_data))
   # mean_model, var_model,autocovariance_model = model_moments(mu, sigma, B,an)
    moms_model = np.array([mean_model ,var_model]+list(D1_values))
    err_vec = (moms_model - moms_data)**2
    return err_vec


def criterion(params, *args):
    mu, sigma, B, an = params
    xvals, W = args
    err = err_vec(mu, sigma, B, an,xvals)
    crit_val = (1/1000) * np.dot(np.dot(err.T, W), err)  # 1000 is the number of observations
    return crit_val

mu_init = 2.101
sig_init = 0.3
B_init=-0.1
a_n_init=1.95
params_init = np.array([mu_init, sig_init,B_init,a_n_init])
W_hat = np.eye(7)

gmm_args = (selected_x_components, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1 = results.x
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1,' B_GMM1=', B_GMM1,' a_n_GMM1=', a_n_GMM1)
mean_data, var_data,autocovariance_data = data_moments(selected_x_components)
#mean_model, var_model,autocovariance_model = model_moments(mu_init, sig_init,B_init,a_n_init)
err1 = err_vec(selected_x_components, mu_init, sig_init,B_init,a_n_init)
print('Mean of points =', mean_data, ', Variance of points =', var_data,', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model, ', Variance of model =', var_model,', Autocovariance of model =', D1_values)
print('Error vector=', err1)
results
params_GMM1 = np.array([mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1])
resultsfinal = opt.minimize(criterion, params_GMM1, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
mu_GMM2, sig_GMM2,B_GMM2,a_n_GMM2 = resultsfinal.x
resultsfinal
print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2,' B_GMM2=', B_GMM2,' a_n_GMM2=', a_n_GMM2)
mean_data, var_data,autocovariance_data = data_moments(sup_OU)
#mean_model2, var_model2,autocovariance_model2 = model_moments(mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
err2 = err_vec(sup_OU, mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
print('Mean of points =', mean_data, ', Variance of points =', var_data,', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model, ', Variance of model =', var_model,', Autocovariance of model =', D1_values)
print('Error vector=', err2)


gmm_args = (sup_OU, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1 = results.x
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1,' B_GMM1=', B_GMM1,' a_n_GMM1=', a_n_GMM1)
mean_data, var_data,autocovariance_data = data_moments(sup_OU)
mean_model, var_model,autocovariance_model = model_moments(mu_init, sig_init,B_init,a_n_init)
err1 = err_vec(sup_OU, mu_init, sig_init,B_init,a_n_init)
print('Mean of points =', mean_data, ', Variance of points =', var_data,', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model, ', Variance of model =', var_model,', Autocovariance of model =', autocovariance_model)
print('Error vector=', err1)
results
params_GMM1 = np.array([mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1])
resultsfinal = opt.minimize(criterion, params_GMM1, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
mu_GMM2, sig_GMM2,B_GMM2,a_n_GMM2 = resultsfinal.x
resultsfinal
print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2,' B_GMM2=', B_GMM2,' a_n_GMM2=', a_n_GMM2)
mean_data, var_data,autocovariance_data = data_moments(sup_OU)
mean_model2, var_model2,autocovariance_model2 = model_moments(mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
err2 = err_vec(sup_OU, mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
print('Mean of points =', mean_data, ', Variance of points =', var_data,', Autocovariance of points =', autocovariance_data)
print('Mean of model =', mean_model2, ', Variance of model =', var_model2,', Autocovariance of model =', autocovariance_model2)
print('Error vector=', err2)


