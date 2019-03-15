import numpy as np
import pandas as pd
from astropy.table import Table
import emcee
import time
import glob
import sys
import scipy.optimize as op
from multiprocessing import Queue, Process, Pool


def mixture_lnlike(theta, x, y, sigma_y):
    m, b, lnp_b, lnv_b, y_b = theta
    model = m * x + b
    bad_noise = np.exp(lnv_b) + sigma_y*sigma_y
    ln_l = np.sum( np.log( (1 - np.exp(lnp_b))/np.sqrt(2*np.pi*sigma_y*sigma_y)*np.exp(-1/2*((y - model)/sigma_y)**2) + 
                    (np.exp(lnp_b))/np.sqrt(2*np.pi*(bad_noise))*np.exp(-1/2*(y - y_b)**2/(bad_noise))
                          )
                  )
    
    return ln_l

def systematic_lnlike(theta, x, y, sigma_y):
    m, b, lnsig_0 = theta
    model = m * x + b
    sig_0 = np.exp(lnsig_0)

    chi2_term = -1/2*np.sum((y - model)**2/(sigma_y**2 + sig_0**2))
    error_term = np.sum(np.log(1/np.sqrt(2*np.pi*(sigma_y**2 + sig_0**2))))
    ln_l = chi2_term + error_term

    return ln_l

def mixture_lnprior(theta):
    m, b, lnp_b, lnv_b, y_b = theta
    if (-1e6 < m < 1e6 and -1e6 < b < 1e6 and 
        np.log(1e-3) < lnp_b < np.log(0.75) and 
        -50 < lnv_b < 50 and -1e4 < y_b < 1e4):
        return 0.0
    return -np.inf

systematic_nll = lambda *args: -systematic_lnlike(*args)
mixture_nll = lambda *args: -mixture_lnlike(*args)

def systematic_lnprior(theta):
    m, b, lnsig_0 = theta
    if (-1e6 < m < 1e6 and -1e6 < b < 1e6 and 
        -50 < lnsig_0 < 50):
        return 0.0
    return -np.inf

# The full log-probability function is
def mixture_lnprob(theta, x, y, yerr):
    lp = mixture_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + mixture_lnlike(theta, x, y, yerr)

def systematic_lnprob(theta, x, y, yerr):
    lp = systematic_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + systematic_lnlike(theta, x, y, yerr)

def small_print_func(i):
    print("Running a single process")
    print(i)
    return

def pool_sys_process(df, i):
    print(i)
    small_print_func(i)
    subdf = df.iloc[np.where(df['index']==i)]
    x = subdf['x'].values
    y = subdf['y'].values
    yerr = subdf['ey'].values
        
    result = op.minimize(systematic_nll, [0, 0, 2], 
                         method='Powell', args=(x, y, yerr))
    ml_guess = result["x"]

    if ml_guess[-1] < -49:
        ml_guess[-1] = -35
    ndim = len(ml_guess)
    nwalkers = 250
    
    pos = [ml_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    # set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, systematic_lnprob, 
                                    args=(x, y, yerr))

    max_samples = 25000

    autocorr = np.empty(max_samples)
    old_tau = np.inf
    sys_tstart = time.time()
    for sample in sampler.sample(pos, iterations=max_samples):
        if ((sampler.iteration % 250) and
            (sampler.iteration < 5000)):
            continue
        elif ((sampler.iteration % 1000) and
              (5000 <= sampler.iteration < 15000)):
            continue
        elif ((sampler.iteration % 2500) and
              (15000 <= sampler.iteration)):
            continue
    #     tau = sampler.get_autocorr_time(tol=0)
    #     autocorr[sampler.iteration-1] = np.mean(tau)
    #
    #     # Check convergence
    #     converged = np.all(tau * 100 < sampler.iteration)
    #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    #     if converged:
    #         break
    #     old_tau = tau
    # sys_tend = time.time()
    # print('Epcoh {} took {:.2f} s'.format(i, sys_tend - sys_tstart))
    # samples = sampler.get_chain(discard=int(10*tau[0]), flat=True)
    samples = sampler.get_chain(discard=5000, flat=True)
    
    Fmcmc_low, Fmcmc_med, Fmcmc_high = np.percentile(samples[:,0], 
                                                     (15.87, 50, 84.13))
    amcmc_low, amcmc_med, amcmc_high = np.percentile(samples[:,1], 
                                                     (15.87, 50, 84.13))
    result = np.array([i, # position argument
                       Fmcmc_med, (Fmcmc_high - Fmcmc_low)/2.,
                       amcmc_med, (amcmc_high - amcmc_low)/2.])
    return result

def pool_mix_process(df, i):
    subdf = df.iloc[np.where(df['index']==i)]
    x = subdf['x'].values
    y = subdf['y'].values
    yerr = subdf['ey'].values

    nwalkers = 250
    ndim = 5
    
    # posterior is highly multi-modal, run a "burn-in" ensemble first
    grid_pos = list(np.reshape(np.mgrid[-250:251:125, 
                                        -50:51:50, 
                                        -5.2:-1.1:2, 
                                        -5:6:5, 
                                        -100:101:100].T, (405,5)))
    grid_walkers = 405
    grid_sampler = emcee.EnsembleSampler(grid_walkers, ndim, mixture_lnprob, 
                                         args=(x, y, yerr))
    grid_samples = 2500
    grid_sampler.run_mcmc(grid_pos, grid_samples, progress=False)

    idx1, idx2 = np.unravel_index(np.argmax(grid_sampler.lnprobability), 
                                  np.shape(grid_sampler.lnprobability))

    emcee_guess = grid_sampler.get_chain()[idx1,idx2,:]
    pos = [emcee_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    
    mix_sampler = emcee.EnsembleSampler(nwalkers, ndim, mixture_lnprob, 
                                args=(x, y, yerr))

    max_samples = 25000

    autocorr = np.empty(max_samples)
    old_tau = np.inf
    for sample in mix_sampler.sample(pos, iterations=max_samples):
        if ((mix_sampler.iteration % 250) and
            (mix_sampler.iteration < 5000)):
            continue
        elif ((mix_sampler.iteration % 1000) and
              (5000 <= mix_sampler.iteration < 15000)):
            continue
        elif ((mix_sampler.iteration % 2500) and
              (15000 <= mix_sampler.iteration)):
            continue
    #     mix_tau = mix_sampler.get_autocorr_time(tol=0)
    #     autocorr[mix_sampler.iteration-1] = np.mean(mix_tau)
    #
    #     # Check convergence
    #     converged = np.all(mix_tau * 100 < mix_sampler.iteration)
    #     converged &= np.all(np.abs(old_tau - mix_tau) / mix_tau < 0.01)
    #     if converged:
    #         break
    #     old_tau = mix_tau
    #
    # mix_samples = mix_sampler.get_chain(discard=int(5*tau[0]), flat=True)
    mix_samples = mix_sampler.get_chain(discard=5000, flat=True)
    
    Fmcmc_low, Fmcmc_med, Fmcmc_high = np.percentile(mix_samples[:,0], 
                                                     (15.87, 50, 84.13))
    amcmc_low, amcmc_med, amcmc_high = np.percentile(mix_samples[:,1], 
                                                     (15.87, 50, 84.13))
    result = np.array([i, # position argument
                       Fmcmc_med, (Fmcmc_high - Fmcmc_low)/2.,
                       amcmc_med, (amcmc_high - amcmc_low)/2.])
    return result

def get_force_photometry(ztf_name, 
                         info_path="/projects/p30796/ZTF/early_Ia/2018/info/",
                         xy_path="/projects/p30796/ZTF/early_Ia/2018/xydata/",
                         mixture=False):
    '''Perform MCMC fit to PSF model to produce forced phot
                         
    Parameters
    ----------
    ztf_name : str
        Name of the ZTF source that requires forced photometry
    
    info_path : str (default="/projects/p30796/ZTF/early_Ia/2018/info/")
        file path to the info*fits file for the source
    
    xy_path : str (default="/projects/p30796/ZTF/early_Ia/2018/xydata/")
        file path to the xy*fits file for the source
    
    mixture : Bool (default=False)
        Boolean to determine if the fit should be done using a model with 
        unknown systematic uncertainty (default), or a model that incorporates
        a Gaussian mixture model to describe "background" outliers
    '''
    
    info_file = info_path + 'force_phot_{}_info_refcut.fits'.format(ztf_name)
    xy_file = xy_path + 'xydata_{}_refcut.fits'.format(ztf_name)

    info_tbl = Table.read(info_file)
    xy_tbl = Table.read(xy_file)
    info_df = info_tbl.to_pandas()
    xy_df = xy_tbl.to_pandas()
    
    if mixture:
        pool = Pool()
        tstart = time.time()
        results = [pool.apply_async(pool_mix_process, args=(xy_df, i,)) 
                   for i in xy_df['index'].unique()]
        output = [p.get() for p in results]
        tend = time.time()

        print("Mixture model took {:.4f} sec".format(tend-tstart))
    else:
        pool = Pool()
        tstart = time.time()
        print('tstart = {:.4f}'.format(tstart))
        results = [pool.apply_async(pool_sys_process, args=(xy_df, i,)) 
                   for i in xy_df['index'].unique()]
        output = [p.get() for p in results]
        tend = time.time()

        print("Pool map took {:.4f} sec".format(tend-tstart))
    
    output_arr = np.array(output)
    print(output_arr)

    Fmcmc = np.zeros(len(xy_df['index'].unique()))
    Fmcmc_unc = np.zeros_like(Fmcmc)
    amcmc = np.zeros_like(Fmcmc)
    amcmc_unc = np.zeros_like(Fmcmc)

    for res_idx in xy_df['index'].unique():
        diff_filename = xy_df['path'].iloc[np.where(xy_df['index'] == res_idx)].unique()[0]
        info_idx = np.where(info_df['diffimgname'] == diff_filename)[0][0]
        output_idx = np.where(output_arr[:,0].astype(int) == int(res_idx))[0]
        print(res_idx, info_idx, output_idx)
        Fmcmc[info_idx] = output_arr[output_idx, 1]                
        Fmcmc_unc[info_idx] = output_arr[output_idx, 2]
        amcmc[info_idx] = output_arr[output_idx, 3]
        amcmc_unc[info_idx] = output_arr[output_idx, 4]
    
    # calculate flux
    f0 = 10**(info_df['zp'].values/2.5)
    f0_unc = f0 / 2.5 * np.log(10) * info_df['ezp']
    Fratio =  Fmcmc/f0
    Fratio_unc = np.hypot(Fmcmc_unc/f0, Fmcmc*f0_unc/f0**2)
    
    info_df['Fmcmc'] = Fmcmc
    info_df['Fmcmc_unc'] = Fmcmc_unc
    info_df['Fratio'] = Fratio
    info_df['Fratio_unc'] = Fratio_unc

    info_df.to_hdf(info_path + '{}_force_phot.h5'.format(ztf_name), 'lc')

if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    print('Got the ZTF name: {}'.format(ztf_name))
    if 2 < len(sys.argv) < 4:
        get_force_photometry(ztf_name, mixture=True)
    else:
        print('... running systematic term ...'.format(ztf_name))
        get_force_photometry(ztf_name)
        