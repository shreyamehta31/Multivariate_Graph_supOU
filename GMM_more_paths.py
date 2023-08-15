import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
#import scipy.integrate as intgr
import scipy.optimize as opt


from More_paths import sup_OU_paths
num_paths = 3

for i in range(num_paths):
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


    def criterion(params, *args):
        mu, sigma,B,an = params
        xvals, W = args
        err = err_vec(xvals, mu, sigma, B,an)
        crit_val = (1/1000)*np.dot(np.dot(err.T, W), err)#1000 is the number of observations
        return crit_val
    mu_init = 1.5
    sig_init = 0.3
    B_init=-0.1
    a_n_init=1.95
    params_init = np.array([mu_init, sig_init,B_init,a_n_init])
    W_hat = np.eye(7)

    gmm_args = (sup_OU_paths[i], W_hat)
    results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
    mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1 = results.x
    print('mu_GMM1',i+1,'=', mu_GMM1, ' sig_GMM1',i+1,'=', sig_GMM1,' B_GMM1',i+1,'=', B_GMM1,' a_n_GMM1=', a_n_GMM1)
    mean_data, var_data,autocovariance_data = data_moments(sup_OU_paths[i])
    mean_model, var_model,autocovariance_model = model_moments(mu_init, sig_init,B_init,a_n_init)
    err1 = err_vec(sup_OU_paths[i], mu_init, sig_init,B_init,a_n_init)
    print('Mean of points ',i+1,'=', mean_data, ', Variance of points ',i+1,'=', var_data,', Autocovariance of points',i+1,'=', autocovariance_data)
    print('Mean of model =', mean_model, ', Variance of model =',var_model,', Autocovariance of model =', autocovariance_model)
    print('Error vector=', err1)
    results
    params_GMM1 = np.array([mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1])
    resultsfinal = opt.minimize(criterion, params_GMM1, args=(gmm_args),
                       method='L-BFGS-B', bounds=((-5, None), (-5, None),(-5, None),(-5, None)))
    mu_GMM2, sig_GMM2,B_GMM2,a_n_GMM2 = resultsfinal.x
    resultsfinal
    print('mu_GMM2',i+1,'=', mu_GMM2, ' sig_GMM2',i+1,'=', sig_GMM2,' B_GMM2',i+1,'=', B_GMM2,' a_n_GMM2',i+1,'=', a_n_GMM2)
    mean_data, var_data,autocovariance_data = data_moments(sup_OU_paths[i])
    mean_model2, var_model2,autocovariance_model2 = model_moments(mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
    err2 = err_vec(sup_OU_paths[i], mu_GMM1, sig_GMM1,B_GMM1,a_n_GMM1)
    print('Mean of points',i+1,'=', mean_data, ', Variance of points ',i+1,'=', var_data,', Autocovariance of points ',i+1,'=', autocovariance_data)
    print('Mean of model ',i+1,'=', mean_model2, ', Variance of model',i+1,'=', var_model2,', Autocovariance of model ',i+1,'=', autocovariance_model2)
    print('Error vector=', err2)


