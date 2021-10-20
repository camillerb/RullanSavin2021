import numpy as np # math library
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def plot_bsn(time, xh, xx, ss, se, pe, dtStim, N, samp, tcurves, m, params = None):
    f, ax = plt.subplots(1, 3, figsize=(20, 4))

    #plot spikes
    [xax, sps] = np.nonzero(ss[:N, :])
    sns.scatterplot(sps * dtStim, xax, ax=ax[0], s=5, color='k', linewidth = 0)
    sns.despine()
    ax[0].set_xlim((se * dtStim, pe * dtStim))
    ax[0].set_ylim((0, N))
    ax[0].set_title('spikes')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('neuron')

    #plot c variable time course
    for i in range(0, xx.shape[0]):
        sns.lineplot(time[0, se:pe], xh[i, se:pe], ax=ax[1])
        sns.lineplot(time[0, se:pe], xx[i, se:pe], ax=ax[1], color='k')
    sns.despine()
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('readout')
    ax[1].set_title('sampling dynamics')
    ax[1].set_xlim((se*dtStim, pe*dtStim))

    #plot histogram of input samples
    sns.distplot(samp, ax = ax[2], norm_hist=True, kde=False, label='input')   
    #plot output distribution 
    sns.lineplot(np.linspace(-m, m, tcurves.shape[0]), np.dot(tcurves, xh[:, -1]) , ax = ax[2], label='marginal output')
    
    #plot true distribution
    if params is not None:
        mn = params['mu']
        sig = np.sqrt(1/params['cov_inv'])
        x = np.linspace(-m, m, 100)
        distvals = norm(mn, sig)
        sns.lineplot(x, distvals.pdf(x), color='k', label='true distribution')
    sns.despine()
    ax[2].set_xlabel('x')
    ax[2].set_xlim([-m, m])
    ax[2].set_ylabel('magnitude')
    ax[2].set_title('marginal')
    plt.show()