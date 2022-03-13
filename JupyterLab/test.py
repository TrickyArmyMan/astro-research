import numpy
import matplotlib.pyplot
import pandas
import glob
import emcee

import eztao
import eztao.ts

import celerite

import multiprocessing as mp
import time

# chi-sqared
def chisqg(y_data, y_model, sd=None):
    chisq = numpy.nansum(((y_data-y_model)/sd)**2)
    return chisq

################################
# Define the prior and log-probability functions for MCMC
################################

# prior function for tau_perturb
def lnprior_perturb(theta):
    """Prior on perturbation timescale. Note: this is a wedge like prior."""

    # determine DHO timescales
    log10_tau_perturb = (theta[-1] - theta[-2])/numpy.log(10)
    if -3 <= log10_tau_perturb <= 5:
        prior = 0
    else:
        prior = -(numpy.abs(log10_tau_perturb - 1) - 4)

    return prior


def lnprior_bounds(theta):
    """Prior on AR and MA parameters. This is a flat prior."""

    # Place some bounds on the parameter space
    bounds_low = numpy.array([-15, -15, -20, -20])
    bounds_high = numpy.array([15, 15, 10, 10])

    log_a1, log_a2, log_b0, log_b1 = theta
    if (
        bounds_low[0] < log_a1 < bounds_high[0]
        and bounds_low[1] < log_a2 < bounds_high[1]
        and bounds_low[2] < log_b0 < bounds_high[2]
        and bounds_low[3] < log_b1 < bounds_high[3]
       ):
        return 0.0
    return -numpy.inf


# We'll use the eztao version which effectively returns "gp.log_likelihood" from the GP and np.inf otherwise
def lnlike(theta, y, gp):
    return -eztao.ts.neg_param_ll(theta, y, gp)


def lnprob(theta, y, gp):
    lp_bounds = lnprior_bounds(theta)
    lp_perturb = lnprior_perturb(theta)
    if not numpy.isfinite(lp_bounds):
        return -numpy.inf
    return lp_bounds + lp_perturb + lnlike(theta, y, gp)


def getCARMAstats(file):
    ################################
    # setup
    ################################

    file_name = file[5:-4]
    #file_name = file[5:-8]

    # read-in light curve
    df = pandas.read_csv(file)

    # obtain values from df
    ra = df['ra'].values[0]
    dec = df['dec'].values[0]
    t = df['mjd'].values
    y_real = df['mag'].values
    yerr_real = df['magerr'].values
    lc_length = len(t)

    # invert the magnitudes
    y_real_inverted = (min(y_real) - y_real)

    # normalize to unit standard deviation and zero mean
    y = (y_real_inverted - numpy.mean(y_real_inverted)) / numpy.std(y_real_inverted)
    yerr = yerr_real / numpy.std(y_real_inverted)

    ################################
    ################################
    #
    # DRW Process
    #
    ################################
    ################################

    # obtain best-fit
    best_drw = eztao.ts.drw_fit(t, y, yerr)

    # define celerite GP model
    # drw_gp = celerite.GP(eztao.carma.DRW_term(*numpy.log(best_drw)), mean=numpy.median(y_real))
    # drw_gp.compute(t, yerr_real)

    # define log prob function
    # def param_ll(*args):
    #    return -eztao.ts.neg_param_ll(*args)

    # initialize the walker, specify number of walkers, prob function, args and etc.
    # initial = numpy.array(numpy.log(best_drw))
    # ndim, nwalkers = len(initial), 32
    # sampler_drw = emcee.EnsembleSampler(nwalkers, ndim, param_ll, args=[y_real, drw_gp])

    # run a burn-in surrounding the best-fit parameters obtained above
    # p0 = initial + 1e-8 * numpy.random.randn(nwalkers, ndim)
    # p0, lp, _ = sampler_drw.run_mcmc(p0, 500)

    # clear up the stored chain from burn-in, rerun the MCMC
    # sampler_drw.reset()
    # sampler_drw.run_mcmc(p0, 2000);

    # remove points with low prob for the sake of making good corner plot
    # prob_threshold_drw = numpy.percentile(sampler_drw.flatlnprobability, 3)
    # clean_chain_drw = sampler_drw.flatchain[sampler_drw.flatlnprobability > prob_threshold_drw, :]

    ################################
    ################################
    #
    # DHO Process
    #
    ################################
    ################################

    # obtain best-fit
    bounds = [(-15, 15), (-15, 15), (-20, 10), (-20, 10)]
    best_dho = eztao.ts.dho_fit(t, y, yerr, user_bounds=bounds)

    # Create the GP model -- instead of creating a "model" function that is then called by the "lnlike" function from tutorial,
    #  we will create a GP that will be passed as an argument to the MCMC sampler. This will be the "gp" that is passed to
    #  the "lnprob" and "param_ll" functions
    dho_kernel = eztao.carma.DHO_term(*numpy.log(best_dho))
    dho_gp = celerite.GP(dho_kernel, mean=numpy.median(y))
    dho_gp.compute(t, yerr)

    ################################
    # MCMC
    ################################

    # Initalize MCMC
    nwalkers = 128
    niter = 2048

    initial = numpy.array(numpy.log(best_dho))
    ndim = len(initial)
    p0 = [numpy.array(initial) + 1e-7 * numpy.random.randn(ndim) for i in range(nwalkers)]

    # Create the MCMC sampler -- note that the GP is passed as an argument in addition to the data
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[y, dho_gp])

    # run a burn-in surrounding the best-fit parameters obtained above
    p0, lp, _ = sampler.run_mcmc(p0, 200)
    sampler.reset()

    # clear up the stored chain from burn-in, rerun the MCMC
    pos, prob, state = sampler.run_mcmc(p0, niter);

    ################################
    # Obtain the Best Fit: theta_max
    ################################

    # put all the samples that explored in MCMC into a single array
    samples = sampler.flatchain

    # find the parameters that have the best fit
    theta_max_index = numpy.argmax(sampler.flatlnprobability)
    theta_max_prob = sampler.flatlnprobability[theta_max_index]

    theta_max = samples[theta_max_index]  # these are in log-space
    theta_max_norm = numpy.exp(theta_max)  # take the exponent to get into 'normal' space

    ################################
    ################################
    #
    # Simulate and Return
    #
    ################################
    ################################

    ################################
    # Simulate and plot light curves
    ################################

    # create simulated light curve
    drw_sim_t, drw_sim_y, drw_sim_yerr = eztao.ts.carma_sim.pred_lc(t, y, yerr, best_drw, 1, t)
    dho_sim_t, dho_sim_y, dho_sim_yerr = eztao.ts.carma_sim.pred_lc(t, y, yerr, theta_max_norm, 2, t)

    # directory to save plots to
    plot_dir = 'carma_plots'

    # plot drw
    matplotlib.pyplot.figure()
    matplotlib.pyplot.errorbar(t, y, yerr=yerr, label='data',
                               linestyle="None", marker='.', ms=3., color='purple', ecolor='0.8')
    matplotlib.pyplot.plot(drw_sim_t, drw_sim_y, label='drw best fit')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f'{plot_dir}/{file_name}_drw_fit.png')

    # plot dho
    matplotlib.pyplot.figure()
    matplotlib.pyplot.errorbar(t, y, yerr=yerr, label='data',
                               linestyle="None", marker='.', ms=3., color='purple', ecolor='0.8')
    matplotlib.pyplot.plot(dho_sim_t, dho_sim_y, label='dho best fit')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f'{plot_dir}/{file_name}_dho_fit.png')

    ################################
    # Determine best fit
    ################################

    # get chi-squared from sim light curves
    chisq_drw = chisqg(y, drw_sim_y, yerr)
    chisq_dho = chisqg(y, dho_sim_y, yerr)

    # determine best fit
    best_fit = 'DRW'
    if chisq_drw > chisq_dho and not numpy.isinf(chisq_dho):
        best_fit = 'DHO'

    ################################
    # Return
    ################################

    # Return pandas df of data. Variables put in [] to force everything into the zeroth row of the df.
    return pandas.DataFrame({'Filenames': [file_name], 'RA': [ra], 'DEC': [dec], 'Times (MJD)': [t],
                             'Magnitudes': [y_real], 'Mag Errors': [yerr_real],
                             'Best DRW Fit': [best_drw], 'DRW_chi_sq': [chisq_drw],
                             'Best DHO Fit': [best_dho], 'DHO MCMC Fit': [theta_max_norm],
                             'DHO MCMC Probability': [theta_max_prob], 'DHO_chi_sq': [chisq_dho],
                             'Best Fit': [best_fit], 'LC Length': [lc_length]})


def main():
    # get list of data files
    repository = glob.glob('data/*.csv')

    start_time = time.time()

    with mp.Pool(2) as pool:
        agn_fit_data = pandas.concat(pool.map(getCARMAstats, repository))

    print("Time elapsed: ", time.time() - start_time)

    print(agn_fit_data)


if __name__ == '__main__':
    main()
