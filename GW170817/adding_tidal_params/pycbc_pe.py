import sys
import time
import h5py
import pickle
import random
import dynesty
import matplotlib
import numpy as np
from tqdm import tqdm
import cProfile
import matplotlib.pyplot as plt

from scipy import special
from scipy.interpolate import interp1d
from scipy.special import hyp2f1
from scipy.linalg import svd
from scipy.interpolate import CubicSpline, PchipInterpolator

from multiprocessing import Process
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from rbf.poly import mvmonos
from rbf.interpolate import RBFInterpolant

import pycbc
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.filter import sigmasq
from pycbc.catalog import Merger
from gwosc.datasets import event_gps
from pycbc.detector import Detector
from pycbc.frame.frame import read_frame
from pycbc.pnutils import get_final_freq
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.types.array import complex_same_precision_as
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta,\
                                                    tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2, eta_from_mass1_mass2, q_from_mass1_mass2, \
                                                    mass1_from_tau0_tau3, mass2_from_tau0_tau3, mass1_from_mchirp_q, mass2_from_mchirp_q
from pe_sampler import dynesty_sampler_pycbc, generate_postSamps_pycbc


approximant = 'IMRPhenomD_NRTidalv2'
fLow = 20
fHigh = 1600
sampling_frequency = 4096 

psd = {}
data = {}
event = 'GW170817'
ifos = ['L1', 'H1', 'V1']
trigTime = 1187008882.4
low_frequency_cutoff = {}
high_frequency_cutoff = {}

for ifo in ifos:
    
    data[ifo] = load_frequencyseries('../GW170817_strain.hdf', group='/data/strain_{}'.format(ifo))
    psd[ifo] = load_frequencyseries('../GW170817_strain.hdf', group='/data/psd_{}'.format(ifo))
    low_frequency_cutoff[ifo] = fLow
    high_frequency_cutoff[ifo] = fHigh

seed_PE = 0

#-- setting fixed parameters and factors ---
st_PE = time.time()

static_params = dict(approximant=approximant, f_lower=fLow)
variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'lambda1', 'lambda2', 'distance', 'tc', 'ra', 'dec', 'inclination', 'polarization']

model = MarginalizedPhaseGaussianNoise(variable_params, data, low_frequency_cutoff, psd, high_frequency_cutoff, \
                              static_params=static_params)

nLive = 500
nWalks = 100
nDims = 12
dlogz = 0.1
sample = 'rwalk' 
nProcs = int(sys.argv[1])

sampler_params = dict(nLive = nLive, nWalks=nWalks, nDims=nDims, dlogz=dlogz, \
                                                      sample=sample, seed_PE=seed_PE, nProcs=nProcs)
m1_sol, m2_sol=  np.loadtxt('points.txt')[:,0], np.loadtxt('points.txt')[:,1]
chi1z_sol, chi2z_sol = np.loadtxt('points.txt')[:,2], np.loadtxt('points.txt')[:,3]

mchirp_sol, mass_ratio_sol = mchirp_from_mass1_mass2(m1_sol, m2_sol), q_from_mass1_mass2(m1_sol, m2_sol)

params_list = ['Mc', 'eta', 'chi1z', 'chi2z', 'iota', 'phi', 'theta', 'dL']

fiducial_params = {}
fiducial_params['Mc'] = mchirp_sol[-1]
fiducial_params['eta'] = eta_from_mass1_mass2(m1_sol, m2_sol)[-1]
fiducial_params['chi1z'] = chi1z_sol[-1]
fiducial_params['chi2z'] = chi2z_sol[-1]
fiducial_params['iota'] = np.pi/4
fiducial_params['phi'] = np.pi/4
fiducial_params['theta'] = np.pi/4
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

#-- defining prior tranform ---
dL_min, dL_max = 10, 60
tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
boundary_dL_tc = {'dL': np.array([dL_min, dL_max]), 'tc': np.array([tc_min, tc_max])}

raw_samples = dynesty_sampler_pycbc(model, boundary, boundary_dL_tc, sampler_params=sampler_params, save_to_hdf=False)
eff_samps_dict = generate_postSamps_pycbc(raw_samples, save_to_hdf=True)
et_PE = time.time()
print('sampling completed and took %.2f seconds....'%(et_PE-st_PE))

with open('seeds_and_timings_pycbc.txt', 'w') as f:

    f.write('\n' + 'pycbc_PE_seed: %d pycbc_pe_time: %.2f minutes'%(seed_PE, (et_PE-st_PE)/60))
