import os
import h5py
import dynesty
import cProfile
import numpy as np
import multiprocessing as mp
from numpy.random import Generator, PCG64
from rbf_pe_utils import RBFInterpolatedLikelihood, evaluateRBFList, cdfinv_q, prior_transform, pycbc_log_likelihood, relbin_log_likelihood
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, eta_from_mass1_mass2

def dynesty_sampler(interp_data, boundary, boundary_dL_tc, sampler_params=None, save_to_hdf=False, **params):
    
    """Function to run nested sampling using dynesty sampler (Static Nested sampler)
    
    Parameters
    ----------
    interp_data: dictionary containing relevent interpolants data
    boundary: dictionary containing the boundaries in the intrinsic parameter space
    boundary_dL_tc: dictionary containing the boudaries in distance and time of coalscence (tc)
    sampler_params: None(default), a dictionary containing sampler parameters (nLive, nWalks, nDims, sample)
    save_to_hdf: to save raw samples in a hdf file
    params: dictionary containing required parameters for RBFInterpolatedLikelihood function (lookup rbf_pe_utils.py)
    
    Returns
    --------
    raw samples: a dictionary containing raw samples"""
    
    global RBF_logL
    
    def RBF_logL(q):

        return RBFInterpolatedLikelihood(q, interpolants, basis_vectors, times, **params)

    global prior

    def prior(cube):

        return prior_transform(cube, boundary, boundary_dL_tc)
    
    interpolants = interp_data['interpolants']
    basis_vectors = interp_data['basis_vectors']
    times = interp_data['times']

    print('********** Sampling starts *********\n')

    # -- sampler settings ---
    if(sampler_params):
        
        nLive = sampler_params['nLive']
        nWalks = sampler_params['nWalks']
        nDims = sampler_params['nDims']
        dlogz = sampler_params['dlogz']
        sample = sampler_params['sample']
        seed_PE = sampler_params['seed_PE']
        nProcs = sampler_params['nProcs']
    
    else:
        
        nLive = 500
        nWalks = 100
        nDims = 10
        dlogz = 0.1
        sample = 'rwalk'
        seed_PE = 0
        nProcs = 32
        
    with mp.Pool(nProcs) as pool:

        sampler = dynesty.NestedSampler(RBF_logL, prior, nDims, sample=sample, pool=pool, nlive=nLive, \
                                                   walks=nWalks, queue_size=nProcs, rstate=Generator(PCG64(seed=seed_PE)))
        sampler.run_nested(dlogz=dlogz)

    #-- saving raw (unweighted) samples ---
    result = sampler.results
    print('Evidence:{}'.format(result['logz'][-1]))

    sample_keys = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'ra', 'dec', 'iota', 'pol', 'dL', 'tc']
    
    raw_samples = {}
    i = 0
    for key in sample_keys:
        
        raw_samples[key] = result['samples'][:,i]
        i = i + 1
            
    raw_samples.update(dict(logwt=result['logwt'], logz=result['logz'], logl=result['logl'], seeds=dict(seed_PE=seed_PE, nodes_seed=interp_data['nodes_params']['nodes_seed'])))
          
    if(save_to_hdf):
        
        file = h5py.File('raw_samples_interp_seedPE_{}_nodes_seed_{}.hdf'.format(seed_PE, interp_data['nodes_params']['nodes_seed']), 'w')
        i = 0
        for key in sample_keys:
            file.create_dataset(key, data=result['samples'][:,i])
            i = i + 1
    
        file.create_dataset('logwt', data=result['logwt']) 
        file.create_dataset('logz', data=result['logz']) 
        file.create_dataset('logl', data=result['logl']) 
        
        grp = file.create_group('seeds')
        grp.create_dataset('seed_PE', data=seed_PE)
        grp.create_dataset('nodes_seed', data=interp_data['nodes_params']['nodes_seed'])
            
        file.close()
            
    return raw_samples

def dynesty_sampler_relbin(model, boundary, boundary_dL_tc, sampler_params=None, save_to_hdf=False):
    
    """Function to run nested sampling using dynesty sampler (Static Nested sampler)
    
    Parameters
    ----------
    interp_data: dictionary containing relevent interpolants data
    boundary: dictionary containing the boundaries in the intrinsic parameter space
    boundary_dL_tc: dictionary containing the boudaries in distance and time of coalscence (tc)
    sampler_params: None(default), a dictionary containing sampler parameters (nLive, nWalks, nDims, sample)
    save_to_hdf: to save raw samples in a hdf file
    params: dictionary containing required parameters for RBFInterpolatedLikelihood function (lookup rbf_pe_utils.py)
    
    Returns
    --------
    raw samples: a dictionary containing raw samples"""
    
    global relbin_logL
    
    def relbin_logL(q):

        return relbin_log_likelihood(q, model)

    global prior

    def prior(cube):

        return prior_transform(cube, boundary, boundary_dL_tc)
    
    print('********** Sampling starts *********\n')

    # -- sampler settings ---
    if(sampler_params):
        
        nLive = sampler_params['nLive']
        nWalks = sampler_params['nWalks']
        nDims = sampler_params['nDims']
        dlogz = sampler_params['dlogz']
        sample = sampler_params['sample']
        seed_PE = sampler_params['seed_PE']
        nProcs = sampler_params['nProcs']
    
    else:
        
        nLive = 500
        nWalks = 100
        nDims = 10
        dlogz = 0.1
        sample = 'rwalk'
        seed_PE = 0
        nProcs = 32
        
    with mp.Pool(nProcs) as pool:

        sampler = dynesty.NestedSampler(relbin_logL, prior, nDims, sample=sample, pool=pool, nlive=nLive, \
                                                   walks=nWalks, queue_size=nProcs, rstate=Generator(PCG64(seed=seed_PE)))
        sampler.run_nested(dlogz=dlogz)

    #-- saving raw (unweighted) samples ---
    result = sampler.results
    print('Evidence:{}'.format(result['logz'][-1]))

    sample_keys = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'ra', 'dec', 'iota', 'pol', 'dL', 'tc']
    
    raw_samples = {}
    i = 0
    for key in sample_keys:
        
        raw_samples[key] = result['samples'][:,i]
        i = i + 1
            
    raw_samples.update(dict(logwt=result['logwt'], logz=result['logz'], logl=result['logl'], seed_PE=seed_PE))
          
    if(save_to_hdf):
        
        file = h5py.File(os.getcwd() + '/raw_samples_relbin/raw_samples_relbin_seedPE_{}_nodes.hdf'.format(seed_PE), 'w')
        i = 0
        for key in sample_keys:
            file.create_dataset(key, data=result['samples'][:,i])
            i = i + 1
    
        file.create_dataset('logwt', data=result['logwt']) 
        file.create_dataset('logz', data=result['logz']) 
        file.create_dataset('logl', data=result['logl']) 
        file.create_dataset('seed_PE', data=seed_PE) 
            
        file.close()
            
    return raw_samples

def dynesty_sampler_pycbc(model, boundary, boundary_dL_tc, sampler_params=None, save_to_hdf=False):
    
    """Function to run nested sampling using dynesty sampler (Static Nested sampler)
    
    Parameters
    ----------
    interp_data: dictionary containing relevent interpolants data
    boundary: dictionary containing the boundaries in the intrinsic parameter space
    boundary_dL_tc: dictionary containing the boudaries in distance and time of coalscence (tc)
    sampler_params: None(default), a dictionary containing sampler parameters (nLive, nWalks, nDims, sample)
    save_to_hdf: to save raw samples in a hdf file
    params: dictionary containing required parameters for RBFInterpolatedLikelihood function (lookup rbf_pe_utils.py)
    
    Returns
    --------
    raw samples: a dictionary containing raw samples"""
    
    global pycbc_logL
    
    def pycbc_logL(q):

        return pycbc_log_likelihood(q, model)

    global prior

    def prior(cube):

        return prior_transform(cube, boundary, boundary_dL_tc)
    
    print('********** Sampling starts *********\n')

    # -- sampler settings ---
    if(sampler_params):
        
        nLive = sampler_params['nLive']
        nWalks = sampler_params['nWalks']
        nDims = sampler_params['nDims']
        dlogz = sampler_params['dlogz']
        sample = sampler_params['sample']
        seed_PE = sampler_params['seed_PE']
        nProcs = sampler_params['nProcs']
    
    else:
        
        nLive = 500
        nWalks = 100
        nDims = 10
        dlogz = 0.1
        sample = 'rwalk'
        seed_PE = 0
        nProcs = 32
        
    with mp.Pool(nProcs) as pool:

        sampler = dynesty.NestedSampler(pycbc_logL, prior, nDims, sample=sample, pool=pool, nlive=nLive, \
                                                   walks=nWalks, queue_size=nProcs, rstate=Generator(PCG64(seed=seed_PE)))
        sampler.run_nested(dlogz=dlogz)

    #-- saving raw (unweighted) samples ---
    result = sampler.results
    print('Evidence:{}'.format(result['logz'][-1]))

    sample_keys = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'ra', 'dec', 'iota', 'pol', 'dL', 'tc']
    
    raw_samples = {}
    i = 0
    for key in sample_keys:
        
        raw_samples[key] = result['samples'][:,i]
        i = i + 1
            
    raw_samples.update(dict(logwt=result['logwt'], logz=result['logz'], logl=result['logl'], seed_PE=seed_PE))
          
    if(save_to_hdf):
        
        file = h5py.File(os.getcwd() + '/raw_samples_relbin/raw_samples_pycbc_seedPE_{}_nodes.hdf'.format(seed_PE), 'w')
        i = 0
        for key in sample_keys:
            file.create_dataset(key, data=result['samples'][:,i])
            i = i + 1
    
        file.create_dataset('logwt', data=result['logwt']) 
        file.create_dataset('logz', data=result['logz']) 
        file.create_dataset('logl', data=result['logl']) 
        file.create_dataset('seed_PE', data=seed_PE) 
            
        file.close()
            
    return raw_samples

def generate_postSamps(raw_samples, nLive, save_to_hdf=False):
    
    """Function to generate posterior samples using the raw samples obtained from dynesty sampler
    
    Parameters
    ----------
    raw_samples: a dictionary containing raw samples 
    
    Returns
    -------
    eff_samps_dict: a dictionary containing the posterior samples"""
    
    mchirp, q, chi1z = raw_samples['Mc'], raw_samples['mass_ratio'], raw_samples['chi1z']    
    chi2z, iota, ra, dec = raw_samples['chi2z'], raw_samples['iota'], raw_samples['ra'], raw_samples['dec']
    distance, pol, tc =  raw_samples['dL'], raw_samples['pol'], raw_samples['tc']
    seed_PE = raw_samples['seeds']['seed_PE']
    nodes_seed = raw_samples['seeds']['nodes_seed']
        
    logwt = raw_samples['logwt']
    logz = raw_samples['logz']
    logl = raw_samples['logl']
        
    wts =  np.exp(logwt - logz[-1])

    samples = np.zeros((len(mchirp), 10))
    samples[:,0], samples[:,1], samples[:,2], samples[:,3], samples[:,4] = mchirp, q, chi1z, chi2z, iota
    samples[:,5], samples[:,6], samples[:,7], samples[:,8], samples[:,9] = ra, dec, distance, pol, tc
    effective_size = int(len(wts)/(1+(wts/wts.mean() - 1)**2).mean())
    
    np.random.seed(0)
    eff_samps_index = np.random.choice(len(mchirp), effective_size, p=wts, replace=False)
    eff_samps = samples[eff_samps_index]
    eff_samps_llr = logl[eff_samps_index]

    m1, m2 = mass1_from_mchirp_q(eff_samps[:,0], eff_samps[:,1]), mass2_from_mchirp_q(eff_samps[:,0], eff_samps[:,1])
    eta = eta_from_mass1_mass2(m1, m2)
    
    chi_eff = (m1*eff_samps[:,2] + m2*eff_samps[:,3])/(m1 + m2)
    
    eff_samps_dict = {}
    
    params = ['Mc', 'mass_ratio', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
    i = 0
    for p in params: 
        
        if(p=='chi_eff'):
            eff_samps_dict[p] = chi_eff
            i = i + 2
        else:
            eff_samps_dict[p] = eff_samps[:,i]
            i = i + 1
    
    eff_samps_dict.update(dict(pol=pol, tc=tc, eta=eta, chi1z=eff_samps[:,2], chi2z=eff_samps[:,3], seed_PE=seed_PE, nodes_seed=nodes_seed))
            
    if(save_to_hdf):
        
        params = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
        file = h5py.File(os.getcwd() + '/post_samples/post_samples_interp_seedPE_{}_nodes_seed_{}_nLive_{}.hdf'.format(seed_PE, nodes_seed, nLive), 'w')   
        i = 0
        for key in params:

            file.create_dataset(key, data=eff_samps_dict[key])

        file.close()
             
    return eff_samps_dict

def generate_postSamps_pycbc(raw_samples, save_to_hdf=False):
    
    """Function to generate posterior samples using the raw samples obtained from dynesty sampler
    
    Parameters
    ----------
    raw_samples: a dictionary containing raw samples 
    
    Returns
    -------
    eff_samps_dict: a dictionary containing the posterior samples"""
    
    mchirp, q, chi1z = raw_samples['Mc'], raw_samples['mass_ratio'], raw_samples['chi1z']    
    chi2z, iota, ra, dec = raw_samples['chi2z'], raw_samples['iota'], raw_samples['ra'], raw_samples['dec']
    distance, pol, tc =  raw_samples['dL'], raw_samples['pol'], raw_samples['tc']
    seed_PE = raw_samples['seed_PE']
        
    logwt = raw_samples['logwt']
    logz = raw_samples['logz']
    logl = raw_samples['logl']
        
    wts =  np.exp(logwt - logz[-1])

    samples = np.zeros((len(mchirp), 10))
    samples[:,0], samples[:,1], samples[:,2], samples[:,3], samples[:,4] = mchirp, q, chi1z, chi2z, iota
    samples[:,5], samples[:,6], samples[:,7], samples[:,8], samples[:,9] = ra, dec, distance, pol, tc
    effective_size = int(len(wts)/(1+(wts/wts.mean() - 1)**2).mean())
    
    np.random.seed(0)
    eff_samps_index = np.random.choice(len(mchirp), effective_size, p=wts, replace=False)
    eff_samps = samples[eff_samps_index]
    eff_samps_llr = logl[eff_samps_index]

    m1, m2 = mass1_from_mchirp_q(eff_samps[:,0], eff_samps[:,1]), mass2_from_mchirp_q(eff_samps[:,0], eff_samps[:,1])
    eta = eta_from_mass1_mass2(m1, m2)
    
    chi_eff = (m1*eff_samps[:,2] + m2*eff_samps[:,3])/(m1 + m2)
    
    eff_samps_dict = {}
    
    params = ['Mc', 'mass_ratio', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
    i = 0
    for p in params: 
        
        if(p=='chi_eff'):
            eff_samps_dict[p] = chi_eff
            i = i + 2
        else:
            eff_samps_dict[p] = eff_samps[:,i]
            i = i + 1
    
    eff_samps_dict.update(dict(pol=pol, tc=tc, eta=eta, chi1z=eff_samps[:,2], chi2z=eff_samps[:,3], seed_PE=seed_PE))
            
    if(save_to_hdf):
        
        params = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
        file = h5py.File(os.getcwd() + '/post_samples/post_samples_pycbc_seedPE_{}.hdf'.format(seed_PE), 'w')   
        i = 0
        for key in params:

            file.create_dataset(key, data=eff_samps_dict[key])

        file.close()
             
    return eff_samps_dict

def generate_postSamps_relbin(raw_samples, save_to_hdf=False):
    
    """Function to generate posterior samples using the raw samples obtained from dynesty sampler
    
    Parameters
    ----------
    raw_samples: a dictionary containing raw samples 
    
    Returns
    -------
    eff_samps_dict: a dictionary containing the posterior samples"""
    
    mchirp, q, chi1z = raw_samples['Mc'], raw_samples['mass_ratio'], raw_samples['chi1z']    
    chi2z, iota, ra, dec = raw_samples['chi2z'], raw_samples['iota'], raw_samples['ra'], raw_samples['dec']
    distance, pol, tc =  raw_samples['dL'], raw_samples['pol'], raw_samples['tc']
    seed_PE = raw_samples['seed_PE']
        
    logwt = raw_samples['logwt']
    logz = raw_samples['logz']
    logl = raw_samples['logl']
        
    wts =  np.exp(logwt - logz[-1])

    samples = np.zeros((len(mchirp), 10))
    samples[:,0], samples[:,1], samples[:,2], samples[:,3], samples[:,4] = mchirp, q, chi1z, chi2z, iota
    samples[:,5], samples[:,6], samples[:,7], samples[:,8], samples[:,9] = ra, dec, distance, pol, tc
    effective_size = int(len(wts)/(1+(wts/wts.mean() - 1)**2).mean())
    
    np.random.seed(0)
    eff_samps_index = np.random.choice(len(mchirp), effective_size, p=wts, replace=False)
    eff_samps = samples[eff_samps_index]
    eff_samps_llr = logl[eff_samps_index]

    m1, m2 = mass1_from_mchirp_q(eff_samps[:,0], eff_samps[:,1]), mass2_from_mchirp_q(eff_samps[:,0], eff_samps[:,1])
    eta = eta_from_mass1_mass2(m1, m2)
    
    chi_eff = (m1*eff_samps[:,2] + m2*eff_samps[:,3])/(m1 + m2)
    
    eff_samps_dict = {}
    
    params = ['Mc', 'mass_ratio', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
    i = 0
    for p in params: 
        
        if(p=='chi_eff'):
            eff_samps_dict[p] = chi_eff
            i = i + 2
        else:
            eff_samps_dict[p] = eff_samps[:,i]
            i = i + 1
    
    eff_samps_dict.update(dict(pol=pol, tc=tc, eta=eta, chi1z=eff_samps[:,2], chi2z=eff_samps[:,3], seed_PE=seed_PE))
            
    if(save_to_hdf):
        
        params = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'chi_eff', 'iota', 'ra', 'dec', 'dL']
    
        file = h5py.File(os.getcwd() + '/post_samples/post_samples_relbin_seedPE_{}.hdf'.format(seed_PE), 'w')   
        i = 0
        for key in params:

            file.create_dataset(key, data=eff_samps_dict[key])

        file.close()
             
    return eff_samps_dict
    