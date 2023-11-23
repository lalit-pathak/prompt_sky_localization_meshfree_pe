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
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q,\
                                                    tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2, eta_from_mass1_mass2, \
                                                    mass1_from_tau0_tau3, mass2_from_tau0_tau3, mass1_from_mchirp_q, mass2_from_mchirp_q
from pe_sampler import dynesty_sampler_pycbc, generate_postSamps_pycbc


approximant = 'IMRPhenomD'
fLow = 20
fHigh = 1024 
sampling_frequency = 4096 

psd = {}
data = {}
event = 'GW200115_042309-v2'
ifos = ['L1', 'H1', 'V1']
trigTime = 1263097407.7
low_frequency_cutoff = {}
high_frequency_cutoff = {}

for ifo in ifos:
    
    data[ifo] = load_frequencyseries('GW200115_042309-v2_strain.hdf', group='/data/strain_{}'.format(ifo))
    psd[ifo] = load_frequencyseries('GW200115_042309-v2_strain.hdf', group='/data/psd_{}'.format(ifo))
    low_frequency_cutoff[ifo] = fLow
    high_frequency_cutoff[ifo] = fHigh

st_PE = time.time()

#-- setting fixed parameters and factors ---
static_params = dict(approximant=approximant, f_lower=fLow)
variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'tc', 'ra', 'dec', 'inclination', 'polarization']

model = MarginalizedPhaseGaussianNoise(variable_params, data, low_frequency_cutoff, psd, \
                                                                   high_frequency_cutoff, static_params=static_params)
nLive = 2500
nWalks = 350
nDims = 10
dlogz = 0.1
sample = 'rwalk'
seed_PE = 4 
nProcs = int(sys.argv[1])

sampler_params = dict(nLive = nLive, nWalks=nWalks, nDims=nDims, dlogz=dlogz, \
                                                      sample=sample, seed_PE=seed_PE, nProcs=nProcs)
#-- fiducial center params ---
Mc_fid, mass_ratio_fid, chi1z_fid, chi2z_fid, dL_fid = np.loadtxt('map_vals_bilby.txt')
m1_fid, m2_fid = mass1_from_mchirp_q(Mc_fid, mass_ratio_fid), mass2_from_mchirp_q(Mc_fid, mass_ratio_fid)
mass_ratio_fid = m1_fid / m2_fid

fiducial_params = {}

fiducial_params['Mc'] = Mc_fid
fiducial_params['chi1z'] = chi1z_fid
fiducial_params['chi2z'] = chi2z_fid
fiducial_params['dL'] = dL_fid
fiducial_params['iota'] = np.pi/4
fiducial_params['phi'] = np.pi/4
fiducial_params['theta'] = np.pi/4
fiducial_params.update(dict(psi=0, tGPS=trigTime))

#-- Boundary for rbf interpolation ---
mchirp_min, mchirp_max = fiducial_params['Mc'] - 0.0011, fiducial_params['Mc'] + 0.0011
mass_ratio_min, mass_ratio_max = mass_ratio_fid - 0.11, mass_ratio_fid + 0.11
chi1z_min, chi1z_max = fiducial_params['chi1z'] - 0.0025, fiducial_params['chi1z'] + 0.0025
chi2z_min, chi2z_max = fiducial_params['chi2z'] - 0.0025, fiducial_params['chi2z'] + 0.0025
boundary = {'Mc': np.array([mchirp_min, mchirp_max]), 'mass_ratio': np.array([mass_ratio_min, mass_ratio_max]), \
                                                     'chi1z': np.array([chi1z_min, chi1z_max]), 'chi2z': np.array([chi2z_min, chi2z_max])}
#-- defining prior tranform ---
dL_min, dL_max = 70, 750
tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
boundary_dL_tc = {'dL': np.array([dL_min, dL_max]), 'tc': np.array([tc_min, tc_max])}
    
sampler_params = dict(nLive = nLive, nWalks=nWalks, nDims=nDims, dlogz=dlogz, \
                                          sample=sample, seed_PE=seed_PE, nProcs=nProcs)

st_PE = time.time()
raw_samples = dynesty_sampler_pycbc(model, boundary, boundary_dL_tc, sampler_params=sampler_params, save_to_hdf=False)
eff_samps_dict = generate_postSamps_pycbc(raw_samples, save_to_hdf=True)
print('generating effective posterior samples from raw samples......')
et_PE = time.time()

print('sampling completed and took %.2f seconds....'%(et_PE-st_PE))

with open('seeds_and_timings_pycbc.txt', 'a') as f:

    f.write('rbf_PE_seed: %s, rbf_PE_time: %.2f minutes'%(seed_PE, (et_PE-st_PE)/60))
