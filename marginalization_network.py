import numpy as np 
#For stability, we update the filtered spike trains through exponential integration separately from the integration of new stimuli. The function kfunct returns a K-dimensional array of the value of each kernel evaluated at the input sample given by the network input vector, stim.

def kfunct(x, bins, m):  # kernel function
    if abs(x) > abs(m):
        # if x is outside the range specified by m, send back zeros
        vals = np.zeros(np.min(bins.shape))
    xax = np.linspace(-m, m, bins.shape[0])
    ind = np.digitize(x, xax)
    if ind != 0 and ind != bins.shape[0] + 1:
        ind = ind - 1
    vals = bins[ind, :]

    return vals

def BSN_marg(stim, dt, params): #marginalization network 
    
    [nd, nt] = stim.shape  # number of dimensions and time bins
    N = params['N']  # number of neurons

    #parameters governing glm nonlinearity
    gamma = params['gamma'] #slope of nonlinearity
    fmax = params['fmax'] #max firing rate
    fmin = params['fmin'] #min firing rate

    #dynamics params
    m = params['m'] #range of tcurves
    beta = params['beta'] #mean of tcurves
    alpha = params['alpha'] #decay of marginal
    kernels = params['kernels'] #centers of kernels

    lambda_d = 1 / params['taud'] #rate decay
    ww = params['ww'] #decoding weights
    Thresh = params['T'] #threshold
    reset = np.dot(ww.T, ww) #voltage reset after a spike
    Rmult = np.exp(-lambda_d * dt) #exponential integrator

    oo = np.zeros([N, nt]) #spikes
    rr = np.zeros([N, nt]) #rates
    cc = np.zeros([nd, nt]) #target
    ch = np.zeros([nd, nt]) #estimate
    vv = np.zeros([N, nt]) #voltage
    cdot = np.zeros([nd, nt]) #derivative
    pspike = np.zeros([N, nt]) #prob of spiking

    for tt in range(1, nt):
        # calculate network estimate of c
        cdot[:, tt] = (1/alpha)*(kfunct(stim[0,tt], kernels, m) - cc[:, tt - 1] * beta)
        cc[:, tt] = cc[:, tt - 1] + cdot[:, tt] * dt #euler update

        # update rates
        rr[:, tt] = Rmult * rr[:, tt - 1] #exponential integration 

        # update voltage
        vv[:, tt] = np.dot(ww.T, cc[:, tt] - np.dot(ww, rr[:, tt]))

        # calculate probability of spiking
        rt = gamma * (vv[:, tt] - Thresh)
        cond = fmax * 1. / (1 + fmax * np.exp(-rt)) + fmin
        pspike[:, tt] = 1 - np.exp(-cond * dt)

        # spiking
        iisp = np.argwhere(np.random.uniform(size=[N]) < pspike[:, tt])
        if len(iisp) > 0:
            oo[iisp, tt] = 1 #update spike train
            rr[iisp, tt] = rr[iisp, tt] + 1 #update filtered sp train
            vv[:, tt] = vv[:, tt] - np.dot(reset, oo[:, tt]) #post-spike reset

        ch[:, tt] = np.dot(ww, rr[:, tt]) #estimate of c variables 

    return oo, rr, ch, cc, vv