import os
import time
import copy
import h5py
import pickle
import numpy as np
from tqdm import tqdm

from scipy import special
from scipy.linalg import svd
from scipy.special import hyp2f1
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator

from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor

from rbf.poly import mvmonos
from rbf.interpolate import RBFInterpolant

from gwfast.network import DetNet
from gwfast.signal import GWSignal
import gwfast.gwfastGlobals as glob
from gwfast.waveforms import IMRPhenomD_NRTidalv2, TaylorF2_RestrictedPN, IMRPhenomD
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from multiprocessing import Pool

import pycbc
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.filter import sigmasq
from pycbc.catalog import Merger
from pycbc.filter import highpass
from pycbc.detector import Detector
from gwosc.datasets import event_gps
from pycbc.psd import interpolate, welch
from pycbc.frame.frame import read_frame
from pycbc.pnutils import get_final_freq
from pycbc.waveform import get_fd_waveform
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.filter.resample import resample_to_delta_t
from pycbc.types.array import complex_same_precision_as
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, q_from_mass1_mass2, \
                                    tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3, \
                                    mass1_from_mchirp_eta, mass2_from_mchirp_eta, eta_from_mass1_mass2


#-- data cleaning and psd generation ----
def get_cleaned_data_psd(event, ifos, sampling_frequency, analysis_start_time, analysis_end_time, high_pass_cutoff, \
                                 low_frequency_cutoff, filter_order, seg_len, psd_estimate_method, window, save_to_hdf=False):
    
    """Function to generate cleaned data using .gwf files and estimated the psd
    
    Parameter
    ---------
    event: GW event
    ifos: list of interferometers
    start_time: start time of the analysis (in secs before merger)
    end_time: end time of the analysis (in secs after merger)
    high_pass_cutoff: high pass frequency (Hz)
    low_frequency_cutoff: low frequency cutoff (Hz)
    filter_order: filter order
    psd_estimate_method: 'median' or 'median-mean' or 'mean' """
    
    merger = Merger(event, source='gwtc-3')
    strain, stilde = {}, {}
    psds = {}

    for ifo in ifos:
        
        ts = merger.strain(ifo, duration=4096).time_slice(merger.time - analysis_start_time, merger.time + analysis_end_time)

        strain[ifo] = highpass(ts, high_pass_cutoff, filter_order=filter_order)  
        strain[ifo] = strain[ifo].crop(8,8)
        strain[ifo] = resample_to_delta_t(strain[ifo], 1/sampling_frequency, method='butterworth')
        stilde[ifo] = strain[ifo].to_frequencyseries()
        
        psds[ifo] = interpolate(strain[ifo].psd(seg_len, avg_method=psd_estimate_method), stilde[ifo].delta_f)
        psds[ifo] = inverse_spectrum_truncation(psds[ifo], int(2 * strain[ifo].sample_rate),
                                                    low_frequency_cutoff=low_frequency_cutoff, trunc_method=window)
        
        if(save_to_hdf):
            
            stilde[ifo].save('{}_strain.hdf'.format(event), group='/data/strain_{}'.format(ifo))
            psds[ifo].save('{}_strain.hdf'.format(event), group='/data/psd_{}'.format(ifo))
        
    return stilde, psds

def hdf_to_txt(event, filename, ifos):
    
    """Function to covert .hdf file to .txt file (for gwfast pacakge since it needs psd in .txt format)
    
    Parameters
    ----------
    event: GW event (string)
    filename: hdf file containig psd (saved using function get_cleaned_data_psd)
    ifos: list of ifos
    
    Returns
    -------
    None (saves .txt files in the loop)"""
    
    for ifo in ifos:
    
        fs = pycbc.types.frequencyseries.load_frequencyseries(filename, group='/data/psd_{}'.format(ifo))
        freq, psdvals = fs.sample_frequencies.data, fs.data.data
        np.savetxt(os.getcwd() + '/psd/' + '{}'.format(event) + '_PSD_' + '{}.txt'.format(ifo), np.column_stack((freq, psdvals)))

    return None

#------- Utility functions for the STARTUP STAGE ------------------

#-- subroutine to calculate covariance matrix ---
def generate_covmat(event, fiducial_params, ifos):
    
    """Function to generate covariance matrix using gwfast package developed by Iacovelli et al.
    For more info about gwfast, please visit https://github.com/CosmoStatGW/gwfast
    
    Parameters
    ----------
    fiducial_params: dictionary of parameters at which to calculate the metric (to be given in the following order)
    ['Mc', 'eta', 'chi1z', 'chi2z', 'iota', 'phi', 'theta', 'dL', 'psi', 'tGPS']
    ifos: list of interferometers
    
    Returns
    -------
    cov_mat: a covarince matrix in intrinsic parameters (['Mc', 'eta', 'chi1z', 'chi2z'])"""

    alldetectors = copy.deepcopy(glob.detectors)
    
    # select only LIGO and Virgo
    
    LVdetectors = {}
    
    for ifo in ifos:
        if(ifo=='V1'):
            ifo = 'Virgo'
        LVdetectors[ifo] = alldetectors[ifo]
    
    for ifo in ifos:
        if(ifo=='V1'):
            ifo_mod = 'Virgo'
            LVdetectors[ifo_mod]['psd_path'] = os.getcwd() + '/psd/{}_PSD_{}.txt'.format(event, ifo)
        else:
            LVdetectors[ifo]['psd_path'] = os.getcwd() + '/psd/{}_PSD_{}.txt'.format(event, ifo)

    myLVSignals = {}

    for d in LVdetectors.keys():

        myLVSignals[d] = GWSignal(IMRPhenomD(), 
                    psd_path=LVdetectors[d]['psd_path'],
                    detector_shape = LVdetectors[d]['shape'],
                    det_lat= LVdetectors[d]['lat'],
                    det_long=LVdetectors[d]['long'],
                    det_xax=LVdetectors[d]['xax'], 
                    verbose=False,
                    useEarthMotion = False,
                    fmin=10.,
                    IntTablePath=None, is_ASD=False) 

    myLVNet = DetNet(myLVSignals)
    
    GW170817_dict = {}
    
    for key, val in fiducial_params.items():
        if(key=='tGPS'):
            GW170817_dict[key] = np.array([val])
        elif(key=='dL'):
            GW170817_dict[key] = np.array([val*1e-3])
        elif(key=='theta'):
            GW170817_dict[key] = np.array([np.pi/2 - val])
        else:
            GW170817_dict[key] = np.array([val])
        
    GW170817_dict.update(dict(Phicoal=np.array([0.])))
    GW170817_dict.update(dict(Lambda1=np.array([0.])))
    GW170817_dict.update(dict(Lambda2=np.array([0.])))
    
    SNR = myLVNet.SNR(GW170817_dict)
    print('SNR for GW170817 is %.2f'%SNR)
    
    totF = myLVNet.FisherMatr(GW170817_dict, df=1/64, use_m1m2=True) # replace df=1/360 by an appropriate variable name
    ParNums = IMRPhenomD().ParNums
    newFish, newPars = fixParams(totF, ParNums, ['iota', 'phi', 'theta', 'dL', \
                                                                     'tcoal', 'Phicoal', 'psi'])
    newCov = CovMatr(newFish)[0]
    cov_mat = np.empty((4,4))
    
    for i in range(4):
        
        for j in range(4):
            
            cov_mat[i,j] = newCov.reshape(4,4)[i,j]
                                        
    return cov_mat

#-- subroutine to generate rbf nodes ---
def cov_transf_m1_m2_chi1_chi2_to_Mc_q_chi1_chi2(cov_mat, fiducial_params):
    
    """Function to tranform covariance matrix from m1, m2, chi1, chi2 to mchirp, q, chi1, chi2
    
    Parameters
    -----------
    cov_mat: covariance matrix in m1, m2, chi1, and chi2 coordinates
    fiducial_params: dictionary of fiducial parameters (only 'Mc' and 'eta' are required)
    
    Returns
    --------
    cov_mat_mc_q_chi1_chi2: tranformed covariance matrix in mchirp, mass_ratio, chi1, and chi2 coordinates"""
    
    def dMc_dm1(m1, m2):
        
        """Function to calculate derivative of mchirp wrt. m1¯
        Parameters
        -----------
        m1: primary component mass
        m2: secondary component mass
        
        Returns
        --------
        derivative at given m1 and m2"""
        
        return m2*(2*m1+3*m2)/(5*(m1*m2)**(2/5)*(m1+m2)**(6/5))
    
    m1 = mass1_from_mchirp_eta(fiducial_params['Mc'], fiducial_params['eta'])
    m2 = mass2_from_mchirp_eta(fiducial_params['Mc'], fiducial_params['eta'])
    
    jacob = np.zeros((4,4))
    jacob[0,0] = dMc_dm1(m1, m2)
    jacob[0,1] = dMc_dm1(m2, m1)
    jacob[1,0] = 1/m2
    jacob[1,1] = -m1/m2**2
    jacob[2,2] = 1
    jacob[3,3] = 1
    
    cov_mat_mc_q_chi1_chi2 = (jacob@cov_mat)@jacob.T 
    
    return cov_mat_mc_q_chi1_chi2
    
#-- subroutine to generate rbf nodes ---
def generate_nodes(event, fiducial_params, nodes_params, ifos, fLow):
        
    """Function to generate nodes in the intrinsic parameter space
    
    Parameters
    -----------
    fiducial_params: dictionary of parameters at which to calculate the metric (to be given in the following order)
    ['Mc', 'eta', 'chi1z', 'chi2z', 'iota', 'phi', 'theta', 'dL', 'psi', 'tGPS']
    boundary: dictionary containing the intrinsic parameter boundaries (maxm and min values)
    fLow: seismic cutoff frequency
    Nnodes: no. of nodes
    nodes_gauss_num: fraction of Nnodes to be generated using covariance matrix
    
    Returns
    ---------
    nodes: uniformly sprayed points in intrinsic parameter space"""

    
    # setting seed for generation of a set of nodes    
    boundary = nodes_params['boundary']
    mchirp_min, mchirp_max = boundary['Mc']
    mass_ratio_min, mass_ratio_max = boundary['mass_ratio']
    s1z_min, s1z_max = boundary['chi1z']
    s2z_min, s2z_max = boundary['chi2z']
    
    if(nodes_params['nodes_gauss_num'] != 0):
        
        # np.random.seed(nodes_params['nodes_seed'])
        m1_fid = mass1_from_mchirp_eta(fiducial_params['Mc'], fiducial_params['eta'])
        m2_fid = mass2_from_mchirp_eta(fiducial_params['Mc'], fiducial_params['eta'])
        mass_ratio_fid = m1_fid / m2_fid
        
        mu = np.array([fiducial_params['Mc'], mass_ratio_fid, fiducial_params['chi1z'], \
                                           fiducial_params['chi2z']])
        # generation of covariance matrix
        cov_mat = generate_covmat(event, fiducial_params, ifos)
        cov_mat_Mc_q = cov_transf_m1_m2_chi1_chi2_to_Mc_q_chi1_chi2(cov_mat, fiducial_params)
        
        gauss_nodes = []
        
        np.random.seed(nodes_params['nodes_seed'])
        
        temp = np.random.multivariate_normal(mu, cov_mat_Mc_q, size=1000000)
        
        idx = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(temp[:,0] > mchirp_min, temp[:,0] < mchirp_max), \
                                np.logical_and(temp[:,1] > mass_ratio_min, temp[:,1] < mass_ratio_max)), \
                                np.logical_and(temp[:,2] > s1z_min, temp[:,2] < s1z_max)), \
                                np.logical_and(temp[:,3] > s2z_min, temp[:,3] < s2z_max)))
        
        gauss_nodes = temp[idx, :][0,:,:][0:int(nodes_params['nodes_gauss_num']*nodes_params['Nnodes']), :]
        
        mchirp = (mchirp_max - mchirp_min)*np.random.rand(int(nodes_params['Nnodes'] - nodes_params['nodes_gauss_num']*nodes_params['Nnodes']))+ mchirp_min
        mass_ratio = (mass_ratio_max - mass_ratio_min)*np.random.rand(int(nodes_params['Nnodes'] -  nodes_params['nodes_gauss_num']*nodes_params['Nnodes']))+ mass_ratio_min
        s1z = (s1z_max - s1z_min)*np.random.rand(int(nodes_params['Nnodes'] - nodes_params['nodes_gauss_num']*nodes_params['Nnodes']))+ s1z_min
        s2z = (s2z_max - s2z_min)*np.random.rand(int(nodes_params['Nnodes'] - nodes_params['nodes_gauss_num']*nodes_params['Nnodes']))+ s2z_min
        
        temp = np.append(np.array(gauss_nodes), np.column_stack((mchirp, mass_ratio, s1z, s2z)), axis=0)

        m1 = mass1_from_mchirp_q(temp[:,0], temp[:,1])
        m2 = mass2_from_mchirp_q(temp[:,0], temp[:,1])

        theta0 = 2*np.pi*fLow*tau0_from_mass1_mass2(m1, m2, fLow)
        theta3 = 2*np.pi*fLow*tau3_from_mass1_mass2(m1, m2, fLow)

        nodes = np.column_stack((theta0, theta3, temp[:,2], temp[:,3]))
        
        plt.scatter(temp[:,0], temp[:,1], marker='o', s=50, color='gainsboro')
        plt.scatter(gauss_nodes[:,0], gauss_nodes[:,1], marker='o', s=50, color='crimson')
        plt.savefig('nodes.png', dpi=100, bbox_inches='tight')
        plt.show()
                
    else:
        
        np.random.seed(nodes_params['nodes_seed'])

        mchirp = (mchirp_max - mchirp_min)*np.random.rand(nodes_params['Nnodes'])+ mchirp_min
        mass_ratio = (mass_ratio_max - mass_ratio_min)*np.random.rand(nodes_params['Nnodes'])+ mass_ratio_min
        s1z = (s1z_max - s1z_min)*np.random.rand(nodes_params['Nnodes'])+ s1z_min
        s2z = (s2z_max - s2z_min)*np.random.rand(nodes_params['Nnodes'])+ s2z_min

        m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
        m2 = mass2_from_mchirp_q(mchirp, mass_ratio)

        theta0 = 2*np.pi*fLow*tau0_from_mass1_mass2(m1, m2, fLow)
        theta3 = 2*np.pi*fLow*tau3_from_mass1_mass2(m1, m2, fLow)

        nodes = np.column_stack((theta0, theta3, s1z, s2z))
        
        plt.scatter(mchirp, mass_ratio, marker='o', s=50, color='gainsboro')
        plt.savefig('nodes.png', dpi=100, bbox_inches='tight')
        plt.show()  
        
    return nodes

#-- function to calculate complex snr and template norm at rbf nodes ---
def sigmasq_zc_calc(theta, psd, data, waveform_params):
    
    """Function to calculate template norm square (<h0|h0>) and 
    snr time-series (\vec z(d,h_0)) at the nodes (parameters)
    
    hp (waveform): 'plus' polarization waveform (function of intrinsic parameters)
    d: strain data
    z: snr timeseries
    
    Parameters
    -----------
    approximant: approximant to generate waveform
    Nnodes: Number of nodes
    f_low : low cutoff frequency
    f_high : high cutoff frequency
    data : filtered data from detector
    psd : Power Spectral Density; Default is None if data is whitened 
    
    Returns
    ---------
    <h0|h0>: template norm square
    \vec z(d,h_0): snr timeseries""" 
    
    theta0, theta3, s1z, s2z = theta
    
    m1 = mass1_from_tau0_tau3(theta0/(2*np.pi*waveform_params['fLow']), theta3/(2*np.pi*waveform_params['fLow']), waveform_params['fLow'])
    m2 = mass2_from_tau0_tau3(theta0/(2*np.pi*waveform_params['fLow']), theta3/(2*np.pi*waveform_params['fLow']), waveform_params['fLow'])
    
    htilde, _ = get_fd_waveform(approximant=waveform_params['approximant'], mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
                            delta_f=psd.delta_f, f_lower=waveform_params['fLow'], f_final=waveform_params['fHigh'])


    p = psd.copy()
    stilde = data.copy()

    d_td = stilde.to_timeseries()

    N = (len(stilde)-1) * 2
    kmin, kmax = get_cutoff_indices(waveform_params['fLow'], waveform_params['fHigh'], data.delta_f, N)

    kmax = min(len(htilde), kmax)

    qtilde = zeros(N, dtype=complex_same_precision_as(stilde))
    _q = zeros(N, dtype=complex_same_precision_as(stilde))

    qtilde[kmin:kmax] = htilde[kmin:kmax].conj()*stilde[kmin:kmax]
    qtilde[kmin:kmax] /= p[kmin:kmax]

    ifft(qtilde, _q)

    sigmasq_val = sigmasq(htilde, p, low_frequency_cutoff=waveform_params['fLow'], high_frequency_cutoff=waveform_params['fHigh']).real   
    z = 4.0*stilde.delta_f*_q
        
    z = TimeSeries(z, delta_t = d_td.delta_t, epoch = d_td.start_time, dtype = complex_same_precision_as(d_td))    
    z = z.time_slice(waveform_params['trigTime'] - waveform_params['tau'], waveform_params['trigTime'] + waveform_params['tau'])
    
    sigmasq_val = sigmasq(htilde, p, low_frequency_cutoff=waveform_params['fLow'], high_frequency_cutoff=waveform_params['fHigh']).real   
    
    return sigmasq_val, z.data

#-- Function to calcualte complex snr and sigmasq parallely in all detectors ---
def parallel_sigmasq_zc_cal_over_ifos(ifo, nodes, psd, data, waveform_params):
    
    global dummy_sigmasq_zc_calc
    
    def dummy_sigmasq_zc_calc(theta):
        
        return sigmasq_zc_calc(theta, psd=psd[ifo], data=data[ifo], waveform_params=waveform_params)
    
    with Pool(waveform_params['nProcs']) as executor:
        x = nodes 
        results = list(tqdm(executor.imap(dummy_sigmasq_zc_calc, x, chunksize=1), total=len(nodes)))
    
    data_td = data[ifo].to_timeseries()
    times = data_td.time_slice(waveform_params['trigTime'] - waveform_params['tau'], waveform_params['trigTime'] + waveform_params['tau']).sample_times.data
    idx = np.where(np.logical_and(data_td.sample_times >= times[0], data_td.sample_times <= times[-1]))[0]
    Ns = len(idx)
    
    sigmasq = np.zeros(len(nodes))
    zc = np.zeros((len(nodes), Ns), dtype=complex)
        
    for index, i in enumerate(results):
        
        sigmasq[index] = i[0]
        zc[index,:] = i[1]
            
    return ifo, sigmasq, zc

def createRBFInterpolants(nodes, sigmasq, zc, times, rbf_params, nodes_params, saveinterpolants=False):
  
    """This function creates rbf interpolants for svd coefficients and hh ---
  
    Parameters
    -----------
    nodes: nodes (in theta0, theta3 co-ordinates)
    hh: values of hh at nodes
    C: svd coefficients
    nBasis: no. of retained top basis vectors
    phi: rbf kernel
    order: order of the monomial terms
  
    Returns
    --------
    hh_interpolant (python object)
    C_interpolant (python object)
  
    # for more details regarding rbf interpolation package, please visit: https://rbf.readthedocs.io/en/latest/interpolate.html
  
    """
    
    # dictinaries containing real and imaginary coefficients 
    C = {}
    basis_vectors = {}

    C_interpolants = {}
    sigmasq_interpolants = {}
    sigma = {}

    for ifo in sigmasq.keys():

        u, s, vh = svd(zc[ifo], full_matrices=False)

        sigma[ifo] = np.diag(s)

        C[ifo] = np.matmul(u, sigma[ifo])

        # hnormsq interpolant
        sigmasq_interpolants[ifo] = RBFInterpolant(nodes, sigmasq[ifo], phi=rbf_params['phi'], order=rbf_params['order'], eps=rbf_params['eps'])

        # svd coefficients interpolant
        C_interpolants_list = []

        for i in tqdm(range(rbf_params['Nbasis'])):

            C_interpolants_list.append(RBFInterpolant(nodes, C[ifo][:,i], phi=rbf_params['phi'], order=rbf_params['order'], eps=rbf_params['eps']))

        C_interpolants[ifo] = np.array(C_interpolants_list)

        basis_vectors[ifo] = vh[0:rbf_params['Nbasis'], :]
        
        
    interpolants = {'sigmasq_interpolants': sigmasq_interpolants, 'C_interpolants': C_interpolants}
    basis_vectors = {'basis_vectors': basis_vectors}
    coeffs_matrix = {'C': C}

    # making a dictionary of the data
    interp_data = {'interpolants': interpolants, 'basis_vectors': basis_vectors, 'coeffs_matrix': coeffs_matrix, \
                   'times': times, 'nodes': nodes, 'nodes_params': nodes_params, 'sigma': sigma}
    
    if(saveinterpolants):
        
        with open(os.getcwd() + '/rbf_interpolants/rbfInterpolants_{}_nNodes_nodes_seed_{}.pickle'.format(len(nodes), nodes_params['nodes_seed']), 'wb') as f:

            pickle.dump(interp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return interp_data

#-- A stand alone function to generate the interpolants for a given set of nodes ---
def start_up(event, fiducial_params, data_params, waveform_params, rbf_params, nodes_params, \
                                       ifos, nProcs, savetrainingdata=False, saveinterpolants=False, verbose=False):
    """
    
    Function to calculate sigmasq and zc at the nodes using mulitple cores
  
    Parameters
    -----------
    fiducial_params: dictionary of parameters at which the metric is calculated
    data_params: dictionary containing psd and data
    waveform_params: dictionary containing essential parameters for generating waveforms
    rbf_params: dictionary containing rbf parameters
    nodes_params: dictionary containing nodes parameters
    ifos: list of interferometers
    nProcs: number of processers to use for parallel processing
    savedtriningdata: either save the (unnormalized)snr timeseries and template normsquare ('True' or 'False')
    geninterpolants: either generate RBFInterpolants ('True' or 'False')
    saveinterpolants: either save RBFInterpolants ('True' or 'False')
    
    Return
    -------
    
    sigmasq: dictionary containing sigmasq arrays corresponding to each detector
    zc: dictionary containing zc matrix corresponding to each detector
    
    """
    
    st = time.time()
    nodes = generate_nodes(event, fiducial_params, nodes_params, ifos, waveform_params['fLow'])
    et = time.time()
    
    if(verbose):
        
        print('generation of nodes took: {} seconds'.format(et-st))
    
    trigTime = fiducial_params['tGPS']
    data = data_params['data']
    psd = data_params['psd']
    
    approximant = waveform_params['approximant']
    fLow = waveform_params['fLow']
    fHigh = waveform_params['fHigh']
    
    tau = rbf_params['tau']
    waveform_params.update(dict(tau=tau, nProcs=nProcs, trigTime=trigTime))
    
    global dummy_parallel_over_ifos
    
    def dummy_parallel_over_ifos(ifo):
        
        return parallel_sigmasq_zc_cal_over_ifos(ifo, nodes=nodes, psd=psd, data=data, waveform_params=waveform_params)
    
    with ProcessPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(dummy_parallel_over_ifos, ifos))
        
    data_td = data[ifos[0]].to_timeseries()
    times = data_td.time_slice(waveform_params['trigTime'] - waveform_params['tau'], waveform_params['trigTime'] + waveform_params['tau']).sample_times.data

    sigmasq_dict = {}
    zc_dict = {}
    
    for result in results:
        
        ifo = result[0]
        sigmasq_dict[ifo] = result[1]
        zc_dict[ifo] = result[2]
    
    if(savetrainingdata):
    
        file = h5py.File('sigmasq_zc.hdf', 'w')
        file.create_dataset('nodes_seed', data=np.array([nodes_seed]))
        for ifo in ifos:
            g = file.create_group(ifo)
            g.create_dataset('sigmasq', data=sigmasq_dict[ifo])
            g.create_dataset('zc', data=zc_dict[ifo])
        file.create_dataset('nodes', data=nodes)
        file.create_dataset('times', data=times)
        file.close()
                
    interp_data = createRBFInterpolants(nodes, sigmasq_dict, zc_dict, times, rbf_params, nodes_params, saveinterpolants)
        
    return interp_data

#------- Utility functions for the ONLINE STAGE ------------------

#-- subroutines for evaluating meshfree log-likelihood at query points ---
def evaluateRBFList(Kqy, Pq, sigmasq_interpolant, C_interpolants, bVecs):
    
    """Function to evaluate interpolated values of template norm square (h0_h0) and SVD coefficient (C) for each basis
    
    Parameters
    -----------
    q: query point [mchirp, eta, s1z, s2z]
    hnormsq_interpolant: interpolant for template norm square (h0_h0)
    rList: list of RBF interpolants for SVD coefficients (assume same observation points)
    
    Returns
    --------
    h0_h0: interpolated value of h0_h0 at the query point (q)
    res.flatten(): list interpolated values of SVD coefficents (for each basis) at the query point (q)"""
    
    eval_interp_vals = np.empty((bVecs, 2))
    
    for i in range(bVecs):
                
        eval_interp_vals[i] = (Kqy.dot(C_interpolants[i].phi_coeff) + Pq.dot(C_interpolants[i].poly_coeff))
            
    sigmasq_interp = (Kqy.dot(sigmasq_interpolant.phi_coeff) + Pq.dot(sigmasq_interpolant.poly_coeff))[0]
 
    return sigmasq_interp, eval_interp_vals

# -- RBF Interpolated log-likelihood at a query point ---
def RBFInterpolatedLikelihood(q, interpolants, basis_vectors, times, **params):
    
    """Function to RBF interpolated likelihood value at a query point (q)
    
    Parameters
    -----------
    q: query point; q = [mchirp, mass_ratio, s1z, s2z]
    h0_h0_interpolant: interpolant for template norm square (h0_h0)
    C_interpolant: list of RBF interpolants for SVD coefficients
    basisVecs: basis vectors from SVD
    nBasis: no. of retained top basis vectors
    times: list of times at for which z time-series is interpolated 
    det: python object containing detector information
    fixed_params: fixed parameters (including extrinsic parameters e.g, sky location and inclination)
    
    Returns
    --------
    llr: interpolated value of log likelihood (marginalized phase) at the query point (q)"""
    
    #-- extract intrinsic and extrinsic parameters from q seperately ---
    mchirp, mass_ratio, s1z, s2z, ra, dec, iota, pol, distance, tc = q 
    m1, m2 = mass1_from_mchirp_q(mchirp, mass_ratio), mass2_from_mchirp_q(mchirp, mass_ratio)
    theta0, theta3 = 2*np.pi*params['low_frequency_cutoff']*tau0_from_mass1_mass2(m1, m2, params['low_frequency_cutoff']), \
                                                            2*np.pi*params['low_frequency_cutoff']*tau3_from_mass1_mass2(m1, m2, params['low_frequency_cutoff'])
    q_int = np.array([theta0, theta3, s1z, s2z]).reshape(1,4)
            
    r0 = interpolants['C_interpolants'][params['ifos'][0]][0]
    Kqy = r0.phi(q_int, r0.y, eps=r0.eps, diff=None)
    Pq = mvmonos((q_int - r0.shift)/r0.scale, r0.order, diff=None)
        
    dh_interp = 0j
    hh = 0
    
    C_interp = {}
    #-- calculating interpolated h0_h0 and C_interp at the query point (q) ---
    for ifo in params['ifos']:
        
        C_interpolant_list = interpolants['C_interpolants'][ifo]
        sigmasq_interpolant = interpolants['sigmasq_interpolants'][ifo]
        
        basisVecs = basis_vectors['basis_vectors'][ifo]
        
        sigmasq_interp, C_interp[ifo] = evaluateRBFList(Kqy, Pq, sigmasq_interpolant, \
                                                                            C_interpolant_list, params['bVecs'])
                
        temp = np.zeros(params['bVecs'], dtype=complex)

        for i in range(params['bVecs']):
    
            temp[i] = complex(C_interp[ifo][i,0], C_interp[ifo][i,1])
        
        #-- calculating quantities dependent on extrinsic parameters ---
        detector = params['det'][ifo]
        del_t = detector.time_delay_from_earth_center(ra, dec, tc)
        fp, fc = detector.antenna_pattern(ra, dec, pol, tc)
        A = (((1 + np.cos(iota)**2)/2)*fp - 1j*np.cos(iota)*fc)/distance

        #-- index corresponding to proposed tc + del_t (ra, dec, tc) ---

        k = int(np.floor((tc + del_t - times[0])*params['sampling_frequency']))
        val = 4
        zc_prime = np.array(np.dot(temp, basisVecs[:params['bVecs'],k-val:k+val]))
        
        zc_R_interpolant = CubicSpline(times[k-val:k+val], zc_prime.real)
        zc_I_interpolant = CubicSpline(times[k-val:k+val], zc_prime.imag)

        dh_interp += A.conj() * complex(zc_R_interpolant(tc + del_t), zc_I_interpolant(tc + del_t))

        hh += A*A.conj()*sigmasq_interp
    
    llr = np.log(special.i0e(abs(dh_interp))) + abs(dh_interp) - 0.5*hh  # marginalized phase likelihood
                
    return llr[0].real

# -- helper subroutines for dynesty sampling ---

# -- function to generate samples of mass_ratio from a distribution which is uniform in mass1 and mass2 \
# constrained by mass_ratio ---

def cdfinv_q(mass_ratio_min, mass_ratio_max, value):

    mass_ratio_array = np.linspace(mass_ratio_min, mass_ratio_max, num=1000, endpoint=True)
    mass_ratio_invcdf_interp = interp1d(cdf_param(mass_ratio_array),
                               mass_ratio_array, kind='cubic',
                               bounds_error=True)
    
    return mass_ratio_invcdf_interp((cdf_param(mass_ratio_max) - cdf_param(mass_ratio_min)) * value + cdf_param(mass_ratio_min))

def cdf_param(mass_ratio):

    return -5. * mass_ratio**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -mass_ratio)
        

def prior_transform(cube, boundary, boundary_dL_tc):
    
    """
    Function to generate a set of parameters from specified distribtutions (used in dynesty sampler)
    
    Parameters
    ----------
    cube: a unit hypercube
    boundary: a dictionary containing the boundaries of the intrinsic parameter space
    boundary_dL_tc: a dictionary containing the boundaries of dL and tc
    
    Returns
    --------
    cube: a cuboid in physical parameters
    
    """
    cube[0] = np.power((boundary['Mc'].max()**2-boundary['Mc'].min()**2)*cube[0]+boundary['Mc'].min()**2,1./2)      # chirpmass: power law mc**1
    cube[1] = cdfinv_q(boundary['mass_ratio'].min(), boundary['mass_ratio'].max(), cube[1])                         # mass_ratio: unifrom in m1 and m2 constrained by q
    cube[2] = boundary['chi1z'].min() + (boundary['chi1z'].max() - boundary['chi1z'].min()) * cube[2]               # s1z: uniform prior
    cube[3] = boundary['chi2z'].min() + (boundary['chi2z'].max() - boundary['chi2z'].min()) * cube[3]               # s2z: uniform prior
    cube[4] = 2*np.pi*cube[4]              # ra: uniform sky
    cube[5] = np.arcsin(2*cube[5] - 1)     # dec: uniform sky
    cube[6] = np.arccos(2*cube[6] - 1)     # iota: sin angle
    cube[7] = 2*np.pi*cube[7]              # pol: uniform angle
    cube[8] = np.power((boundary_dL_tc['dL'].max()**3 - boundary_dL_tc['dL'].min()**3)*cube[8] + boundary_dL_tc['dL'].min()**3, 1./3) # distance: unifrom prior in dL**3
    cube[9] = boundary_dL_tc['tc'].min() + (boundary_dL_tc['tc'].max() - boundary_dL_tc['tc'].min()) * cube[9]                   # tc: uniform prior

    return cube

def calc_mode(samps):
    
    """Function to calculate the mode
    
    Parameters
    -----------
    samps: samples
    bins: number of bins for np.histogram
    
    Returns
    --------
    map_val: mode/MAP value"""
    
    
    kernel = gaussian_kde(samps)
    map_val = samps[np.argmax(kernel(samps))]
    
    return map_val

def pycbc_samps_mode_vals(filename):
    
    file = h5py.File(filename, 'r')
    mchirp, mass_ratio, chi1z = np.array(file['Mc']), np.array(file['mass_ratio']), np.array(file['chi1z'])    
    chi2z, iota, ra, dec = np.array(file['chi2z']), np.array(file['iota']), np.array(file['ra']), np.array(file['dec'])
    dL =  np.array(file['dL'])  
    
    m1, m2 = mass1_from_mchirp_q(mchirp, mass_ratio), mass2_from_mchirp_q(mchirp, mass_ratio)
    eta = eta_from_mass1_mass2(m1, m2)
        
    return np.column_stack([mchirp, eta, chi1z, chi2z, iota, ra, dec, dL])

def pycbc_log_likelihood(q, model):
        
    """Function to calculate pycbc log-likelihood
    
    Parameters
    -----------
    q: parameters
    model: pycbc object containg the information regarding noise and waveform model
    PSD: power spectral density
    data: strain data
    
    Returns
    --------
    model.loglr: marginalized phase likelihood
    
    For more details visit: https://pycbc.org/pycbc/latest/html/_modules/pycbc/inference/models/
    marginalized_gaussian_noise.html#MarginalizedPhaseGaussianNoise"""
    
    mchirp, mass_ratio, s1z, s2z, ra, dec, iota, pol, distance, tc = q 
    m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
    m2 = mass2_from_mchirp_q(mchirp, mass_ratio)
    
    model.update(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, tc=tc, distance=distance, ra=ra, dec=dec, inclination=iota, polarization=pol)

    return model.loglr

def relbin_log_likelihood(q, model):
        
    """Function to calculate pycbc log-likelihood
    
    Parameters
    -----------
    q: parameters
    model: pycbc object containg the information regarding noise and waveform model
    PSD: power spectral density
    data: strain data
    
    Returns
    --------
    model.loglr: marginalized phase likelihood
    
    For more details visit: https://pycbc.org/pycbc/latest/html/_modules/pycbc/inference/models/
    marginalized_gaussian_noise.html#MarginalizedPhaseGaussianNoise"""
    
    mchirp, mass_ratio, s1z, s2z, ra, dec, iota, pol, distance, tc = q 
    m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
    m2 = mass2_from_mchirp_q(mchirp, mass_ratio)
    
    model.update(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, tc=tc, distance=distance, ra=ra, dec=dec, inclination=iota, polarization=pol)

    return model.loglr