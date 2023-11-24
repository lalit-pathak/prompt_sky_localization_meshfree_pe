import sys
import os
import time
import glob
import h5py
import psutil
import pickle
import random
import dynesty
import cProfile
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import special
from scipy.linalg import svd
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator

import multiprocessing as mp
from numpy.random import Generator, PCG64

from rbf.poly import mvmonos
from rbf.interpolate import RBFInterpolant

import pycbc
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.filter import sigmasq
from pycbc.catalog import Merger
from pycbc.detector import Detector
from gwosc.datasets import event_gps
from pycbc.frame.frame import read_frame
from pycbc.pnutils import get_final_freq
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, \
                                    tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3, \
                                    mass1_from_mchirp_eta, mass2_from_mchirp_eta, eta_from_mass1_mass2, q_from_mass1_mass2

from rbf_pe_utils import cdfinv_q, start_up, get_cleaned_data_psd, prior_transform, \
                                                            RBFInterpolatedLikelihood, calc_mode, pycbc_samps_mode_vals, hdf_to_txt
from pe_sampler import dynesty_sampler, generate_postSamps

#-- section: data ----
event = 'GW170817'
analysis_start_time = 342  #(before trigTime)
analysis_end_time = 30     #(after trigTime)
sampling_frequency = 4096  #Hz
low_frequency_cutoff = 20  #Hz
high_frequency_cutoff = 1600 #Hz ~ fISCO corresponding to M = 2.76 M_solar
high_pass_cutoff = 18 #Hz
filter_order = 4 # filter order for highpass
seg_len = 2 # segment length for psd estimation
trigTime = 1187008882.4 # trigger time of the event (Geocentric GPS time)
psd_estimate_method = 'median-mean' # psd estimation method
window = 'hann' # window for inverse psd truncation 
ifos = ['L1', 'H1', 'V1'] # list of available detectors for the given event

#--- checking for a .hdf file containg the data and psd in the current directory -----
filename_prefix = '../{}_strain'.format(event)
directory = '.'
choices = glob.glob(os.path.join(directory, '{}*.hdf'.format(filename_prefix)))

if any(choices):
    
    print('File with', filename_prefix, 'found!')
    
    data = {}
    psd = {}
    
    for ifo in ifos:
        
        data[ifo] = load_frequencyseries(filename_prefix + '.hdf', group='/data/strain_{}'.format(ifo))
        psd[ifo] = load_frequencyseries(filename_prefix + '.hdf', group='/data/psd_{}'.format(ifo))
        
    data_dict = {'psd': psd, 'data': data}
     
else:
    
    print('cleaning the data and generating the psd......')
    data, psd = get_cleaned_data_psd(event, ifos, analysis_start_time, analysis_end_time, high_pass_cutoff, \
                                            low_frequency_cutoff, filter_order, seg_len, psd_estimate_method, window, save_to_hdf=True)

    data_dict = {'psd': psd, 'data': data}
    

nodes_seed = 9
seed_PE = 0

st_start_up = time.time()

#-- Event params ---
trigger_params = {}  # Add an optmization algorithm later on to reach an appropriate 
# center from the trigger and calcuate the covarince matrix at the center

#-- fiducial center params ---
#-- found using the optimization routine starting from best-matched template from search pipelines-----

m1_sol, m2_sol, chi1z_sol, chi2z_sol =  np.loadtxt('points.txt')[:,0], np.loadtxt('points.txt')[:,1], \
                                                                        np.loadtxt('points.txt')[:,2], np.loadtxt('points.txt')[:,3]

mchirp_sol, mass_ratio_sol = mchirp_from_mass1_mass2(m1_sol, m2_sol), q_from_mass1_mass2(m1_sol, m2_sol)
params_list = ['Mc', 'eta', 'chi1z', 'chi2z', 'iota', 'phi', 'theta', 'dL']

fiducial_params = {}
fiducial_params['Mc'] = mchirp_sol[-1]
fiducial_params['eta'] = eta_from_mass1_mass2(m1_sol[-1], m2_sol[-1])
fiducial_params['chi1z'] = chi1z_sol[-1]
fiducial_params['chi2z'] = chi2z_sol[-1]
fiducial_params['iota'] = np.pi/4
fiducial_params['phi'] = np.pi/4
fiducial_params['theta'] = np.pi/4
fiducial_params['dL'] = 37.6
fiducial_params.update(dict(psi=0, tGPS=trigTime))  

#-- Boundary for rbf interpolation ---
mchirp_min, mchirp_max = mchirp_sol[-1] - 0.0001, mchirp_sol[-1] + 0.0001
mass_ratio_min, mass_ratio_max = mass_ratio_sol[-1] - 0.07, mass_ratio_sol[-1] + 0.07
chi1z_min, chi1z_max = chi1z_sol[-1] - 0.0025, chi1z_sol[-1] + 0.0025
chi2z_min, chi2z_max = chi2z_sol[-1] - 0.0025, chi2z_sol[-1] + 0.0025
lambda1_min, lambda1_max = 100, 400 # 500-700
lambda2_min, lambda2_max = 100, 400 # 500-700
boundary = {'Mc': np.array([mchirp_min, mchirp_max]), 'mass_ratio': np.array([mass_ratio_min, mass_ratio_max]), 'chi1z': np.array([chi1z_min, chi1z_max]), \
            'chi2z': np.array([chi2z_min, chi2z_max]), 'lambda1': np.array([lambda1_min, lambda1_max]), 'lambda2': np.array([lambda2_min, lambda2_max])}

#-- Section: start_up analysis --------------

#-- waveform params ----
approximant = 'IMRPhenomD_NRTidalv2'
waveform_params = {'approximant': approximant, 'fLow': low_frequency_cutoff, 'fHigh': high_frequency_cutoff, 'sampling_frequency': sampling_frequency}

#-- rbf params---
tau = 0.15
Nbasis = 20
phi = 'ga'
order = 7 
eps = 10 # 10 works fine #30 for the best result
rbf_params = {'Nbasis': Nbasis, 'phi': phi, 'order': order, 'eps': eps, 'tau': tau}

#-- Nodes params ---
Nnodes = 2500
nodes_gauss_num = 0.2
nodes_params = {'boundary': boundary, 'Nnodes': Nnodes, 'nodes_gauss_num': nodes_gauss_num, 'nodes_seed': nodes_seed}

print('start-up stage starts........')

#-- startup stage ---
nProcs = int(sys.argv[1])
filename = '{}_strain.hdf'.format(event)    
hdf_to_txt(event, filename, ifos) # function to convert 

interp_data = start_up(event, fiducial_params, data_dict, waveform_params, rbf_params, nodes_params, \
                                               ifos, nProcs, savetrainingdata=True, saveinterpolants=True)
interpolants = interp_data['interpolants']
basis_vectors = interp_data['basis_vectors']
times = interp_data['times']
interp_data.update(dict(nodes_params=nodes_params))

et_start_up = time.time()

print('start-up stage is completed......')
print('start-up stage took %.2f seconds....'%(et_start_up-st_start_up))

# writing startup time and nodes seed in .txt file for future references
with open('seeds_and_timings_rbf.txt', 'a') as f:

    f.write('nodes_seed: %d, start_up_time: %.2f minutes'%(nodes_seed, (et_start_up-st_start_up)/60))

#-- Sampling using dynesty ---
bVecs = 20
det = {}

for ifo in ifos:

    det[ifo] = Detector(ifo)

#-- defining prior tranform ---
dL_min, dL_max = 10, 60
tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
boundary_dL_tc = {'dL': np.array([dL_min, dL_max]), 'tc': np.array([tc_min, tc_max])}

nLive = 500
nWalks = 100
nDims = 12
dlogz = 0.1
sample = 'rwalk'

sampler_params = dict(nLive = nLive, nWalks=nWalks, nDims=nDims, dlogz=dlogz, \
                                          sample=sample, seed_PE=seed_PE, nProcs=nProcs)

params = {'det': det, 'low_frequency_cutoff': low_frequency_cutoff, \
                          'sampling_frequency': sampling_frequency, 'ifos': ifos, 'bVecs': bVecs}

st_PE = time.time()
raw_samples = dynesty_sampler(interp_data, boundary, boundary_dL_tc, sampler_params=sampler_params, save_to_hdf=False, **params)
eff_samps_dict = generate_postSamps(raw_samples, save_to_hdf=True)
et_PE = time.time()

print('sampling completed and took %.2f seconds....'%(et_PE-st_PE))

with open('seeds_and_timings_rbf.txt', 'a') as f:

    f.write('\n' + 'rbf_PE_seed: %s, rbf_PE_time: %.2f minutes'%(seed_PE, (et_PE-st_PE)/60))
    
